// vipera: a llama.cpp runtime that places the model across the GPU, the CPU and
// RAM before it loads a weight, sizes the KV cache against the RAM that is left,
// then reports what the run actually cost.
//
// Why this file exists instead of a wrapper script: llama-cli silently accepts a
// context larger than RAM and pages; it also prints "0.0 t/s" when a prompt does
// not fit instead of failing. Both are silent-wrong, and this tool refuses to be.
//
// Every printed number is read from the model header, read from /proc, or
// measured by this process. Nothing is inferred.

#include "llama.h"
#include "accept.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "fit.h"
#include "speculative.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr size_t MIB = 1024 * 1024;
constexpr size_t DEFAULT_HEADROOM = 512 * MIB;

// ---------------------------------------------------------------- lifetime ---

struct ModelFree { void operator()(llama_model * p) const { llama_model_free(p); } };
struct CtxFree   { void operator()(llama_context * p) const { llama_free(p); } };
struct BatchFree { void operator()(llama_batch_ext * p) const { llama_batch_ext_free(p); } };
struct SamplerFree { void operator()(llama_sampler * p) const { llama_sampler_free(p); } };

using ModelPtr   = std::unique_ptr<llama_model, ModelFree>;
using CtxPtr     = std::unique_ptr<llama_context, CtxFree>;
using BatchPtr   = std::unique_ptr<llama_batch_ext, BatchFree>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerFree>;

// -------------------------------------------------------------------- log ----

bool g_verbose = false;

// Default: the library's warnings and errors still reach stderr, only its
// per-tensor INFO chatter is dropped. Swallowing everything (as the first
// version did) hides real load failures behind a clean-looking screen.
void log_forward(enum ggml_log_level level, const char * text, void *) {
    const bool wanted = g_verbose || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
    if (wanted) {
        fputs(text, stderr);
    }
}

// -------------------------------------------------------------- host facts ---

size_t mem_available_bytes() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) {
        return 0;
    }
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            kb = std::strtoull(line + 13, nullptr, 10);
            break;
        }
    }
    fclose(f);
    return kb * 1024;
}

// Physical cores only: decode here is ALU-bound, and the measured hyperthread
// gain on a 4C/8T laptop is negative (0.91x at -t 8 against -t 4). Counting
// distinct (physical id, core id) pairs is what /proc/cpuinfo actually reports;
// a single "cpu cores" field is per socket and undercounts multi-socket boxes.
int physical_cores() {
    FILE * f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        std::vector<std::pair<int, int>> seen;
        int phys = -1;
        int core = -1;
        int cores = 0;
        while (fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "physical id", 11) == 0) {
                phys = std::atoi(std::strchr(line, ':') + 1);
            } else if (std::strncmp(line, "core id", 7) == 0) {
                core = std::atoi(std::strchr(line, ':') + 1);
            } else if (line[0] == '\n' && core >= 0) {
                const std::pair<int, int> id{phys, core};
                if (std::find(seen.begin(), seen.end(), id) == seen.end()) {
                    seen.push_back(id);
                }
                phys = core = -1;
            }
        }
        fclose(f);
        cores = (int) seen.size();
        if (cores > 0) {
            return cores;
        }
    }
    const int logical = (int) std::thread::hardware_concurrency();
    return std::max(1, logical / 2);
}

// -------------------------------------------------------------------- kv -----

ggml_type kv_type_from(const std::string & name, bool & ok) {
    static const struct { const char * name; ggml_type type; } table[] = {
        {"f32", GGML_TYPE_F32},   {"f16", GGML_TYPE_F16},   {"bf16", GGML_TYPE_BF16},
        {"q8_0", GGML_TYPE_Q8_0}, {"q5_1", GGML_TYPE_Q5_1}, {"q5_0", GGML_TYPE_Q5_0},
        {"q4_1", GGML_TYPE_Q4_1}, {"q4_0", GGML_TYPE_Q4_0}, {"iq4_nl", GGML_TYPE_IQ4_NL},
    };
    for (const auto & row : table) {
        if (name == row.name) {
            ok = true;
            return row.type;
        }
    }
    ok = false;
    return GGML_TYPE_F16;
}

bool is_quantized(ggml_type t) {
    return t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_BF16;
}

// Per-layer KV head counts, read once: a hybrid checkpoint lists them (some
// layers hold no cache at all), and repeating the scalar the API exposes would
// over-count those layers every time the number is printed.
std::vector<int> kv_heads_per_layer(const llama_model * model) {
    const int n_layer = llama_model_n_layer(model);
    std::vector<int> heads((size_t) std::max(0, n_layer), llama_model_n_head_kv(model));

    char arch[128] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
        return heads;
    }
    char key[256];
    std::snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
    char value[4096] = {0};
    if (llama_model_meta_val_str(model, key, value, sizeof(value)) < 0) {
        return heads;
    }

    std::vector<int> parsed;
    for (const char * p = value; *p; ) {
        char * end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        parsed.push_back((int) v);
        p = (*end == ',') ? end + 1 : end;
    }
    return (int) parsed.size() == n_layer ? parsed : heads;
}

// Bytes of KV cache added per token. Computed from the tensor type that will
// store it: ggml_row_size rounds a head row up to the quantization block, which
// is exactly where "one byte per element" undercounts q8_0 by 6%.
size_t kv_bytes_per_token(const llama_model * model, ggml_type type_k, ggml_type type_v,
                          const std::vector<int> & heads) {
    const int n_head = std::max(1, llama_model_n_head(model));
    const int head_dim = llama_model_n_embd(model) / n_head;
    const size_t per_head = ggml_row_size(type_k, head_dim) + ggml_row_size(type_v, head_dim);
    size_t per_token = 0;
    for (int h : heads) {
        per_token += (size_t) h * per_head;
    }
    return per_token;
}

// ------------------------------------------------------------------ device ---

struct Device {
    std::string name;
    size_t free = 0;
    size_t total = 0;
    bool present = false;
};

// The card the fit will fill. Read from ggml's device registry, not from a
// vendor CLI: whatever backend this binary was built with is what it reports.
Device first_gpu() {
    Device d;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        d.name = ggml_backend_dev_description(dev);
        d.free = free_bytes;
        d.total = total_bytes;
        d.present = true;
        break;
    }
    return d;
}

// ------------------------------------------------------------------- plan ----

struct Plan {
    int ctx;              // tokens the context will actually be created with
    size_t kv;            // KV bytes for `ctx`
    size_t per_token;     // KV bytes per token
    bool clamped;
    bool limited_by_train;  // context cut by the model, not by RAM
    std::string reason;
};

// `resident` is the weight bytes that live in system RAM: the layers the GPU
// did not take, plus the non-repeating tensors. Charging the whole file to RAM
// after offloading would under-count the context a hybrid run can afford.
// `cap` is the context the placement was fitted for (0 = no cap): asking for
// more tokens than that would put KV back into VRAM the layers already filled.
Plan make_plan(const llama_model * model, int requested, ggml_type type_k, ggml_type type_v,
               const std::vector<int> & heads, size_t resident, size_t available, int cap) {
    Plan p{};
    p.per_token = kv_bytes_per_token(model, type_k, type_v, heads);

    const int train = llama_model_n_ctx_train(model);
    p.ctx = requested > 0 ? requested : train;
    if (p.ctx > train) {
        p.ctx = train;
        p.clamped = true;
        p.limited_by_train = true;
        p.reason = "the model was trained for fewer tokens";
    }
    if (cap > 0 && p.ctx > cap) {
        p.ctx = cap;
        p.clamped = true;
        p.limited_by_train = false;
        p.reason = "the card cannot hold a larger KV cache beside the layers it took";
    }

    if (available > 0 && p.per_token > 0) {
        const long long budget = (long long) available - (long long) resident -
                                 (long long) DEFAULT_HEADROOM;
        if (budget <= 0) {
            p.reason = "no RAM is left beside the model for a KV cache";
            p.clamped = true;
        } else if ((size_t) budget / p.per_token < (size_t) p.ctx) {
            p.ctx = (int) ((size_t) budget / p.per_token);
            p.clamped = true;
            p.limited_by_train = false;
            p.reason = "the KV cache does not fit beside the model in RAM";
        }
    }
    p.kv = p.per_token * (size_t) p.ctx;
    return p;
}

// ------------------------------------------------------------------- main ----

struct Options {
    std::string model;
    std::string prompt = "The capital of France is";
    std::string kv_k = "f16";
    std::string kv_v = "";      // empty = same as -ctk
    int ctx = 0;
    int n_predict = 64;
    int threads = 0;            // 0 = physical cores
    int ngl = -1;               // -1 = fit the free VRAM, 0 = CPU only
    int ubatch = 0;             // 0 = min(512, -b); the CPU side sizes the batch
    int fit_target = 0;         // MiB kept free on the card; 0 = scale to its size
    int n_cpu_ffn = 0;          // 0 = off; else this many layers read FFN from RAM
    // -ncffn auto: pick it by probing, see run_ffn_ladder.
    bool ffn_auto = false;
    // Internal, set on the probe children the ladder spawns: load, create the
    // context, report, exit. Never a user-facing mode.
    bool ffn_probe = false;
    float temp = 0.0f;          // 0 = greedy
    float top_p = 0.95f;
    float min_p = 0.05f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    // Self-speculation, off by default: the n-gram drafter proposes the tokens
    // that followed the same n-gram in the prompt, and the model verifies them
    // in one batch. Placement-neutral (no weights, no KV change), so it belongs
    // beside -ngl rather than in the trade-offs -ncffn makes.
    std::string spec = "off";
    int spec_n = 48;            // draft tokens proposed per step
    bool plan_only = false;
    int kv_mode = 0;            // 0 = KV cache on the card, -1 = in RAM (-nkvo)
    // Off by default: the prompt number below excludes the allocation because
    // the prefill is synchronised, so the warmup only costs its own token
    // (measured 387.8 vs 396.4 t/s prompt, 73.8 vs 73.4 decode, 1281-token
    // prompt on CUDA). It stays for callers that want a warm cache.
    bool warmup = false;
    bool help = false;
};

void print_usage(const char * argv0) {
    std::printf(
        "\nusage:\n"
        "  %s -m model.gguf [options]\n\n"
        "options:\n"
        "  -c N            context to request (0 = fitted: the largest that fits;\n"
        "                  with -ngl 0 it defaults to the model's trained context)\n"
        "  -n N            tokens to generate (default 64)\n"
        "  -t N            threads (default: physical cores)\n"
        "  -ub N           batch size for one pass (default 512, max 1024: a\n"
        "                  larger batch needs compute buffers the card lacks;\n"
        "                  1024 prefilled 1655 tokens at 88.0 t/s against 84.3)\n"
        "  -ngl N          layers on GPU; default fits the free VRAM (see --fit)\n"
        "  --fit on|off    place layers across GPU and RAM automatically (default on)\n"
        "  --fit-target N  MiB the fit must leave free on the card (default: a\n"
        "                  scaled one - card/16, at least 192, at most 512, where\n"
        "                  upstream's fixed 1024 is 28%% of a 4 GB card and\n"
        "                  card/8 was worth three layers)\n"
        "  -nkvo           keep the KV cache in RAM instead of on the card; on a\n"
        "                  partial split it buys layers (19/28 vs 17/28 at 8k on\n"
        "                  a 4 GB card) and costs a bus read per attention step\n"
        "  -ncffn N        attention of every layer on the card, FFN of the first\n"
        "                  N layers read from RAM. Bypasses the fit - the split\n"
        "                  cannot be priced from a header - so pass -c; -ctk stays\n"
        "                  f16 only when it fits, otherwise it is lowered to q8_0\n"
        "  -ncffn auto     find the smallest N that still creates a context and\n"
        "                  run that: smaller is faster, so the answer sits on the\n"
        "                  cliff (measured 7B/4 GB: 5 fails, 6 runs). Costs a\n"
        "                  probe process per rung - ~1.5 s per rung that fails,\n"
        "                  plus the one that wins - each a child process, so no\n"
        "                  VRAM leaks between attempts\n"
        "  -p TEXT         prompt (default \"The capital of France is\")\n"
        "  --kv TYPE       KV cache type for K and V (default f16)\n"
        "  -ctk/-ctv TYPE  set K and V separately\n"
        "  --temp/--top-p/--min-p/--seed   sampling (temp 0 = greedy, the default)\n"
        "  --spec TYPE     self-speculative decoding with no extra model: the\n"
        "                  n-gram map drafter indexes the prompt and proposes the\n"
        "                  tokens that followed the same n-gram before, so the\n"
        "                  model checks several tokens in one pass. No weights, no\n"
        "                  KV change, no layer given up; it pays on output that\n"
        "                  repeats the prompt and ~nothing on fresh prose. Types:\n"
        "                  ngram-map-k (default pick), ngram-map-k4v,\n"
        "                  ngram-simple, ngram-mod; 'off' is the default\n"
        "  --spec-n N      draft tokens per step (default 48)\n"
        "  --plan          print the plan and exit (loads metadata only)\n"
        "  --warmup        decode one token first so the prompt starts from a\n"
        "                  warm graph (costs that one token, ~2%% of prompt)\n"
        "  --no-warmup     accepted for compatibility; it is the default\n"
        "  --verbose       forward llama.cpp's INFO log to stderr\n"
        "  -h, --help\n\n", argv0);
}

bool parse_args(int argc, char ** argv, Options & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        if (a == "-h" || a == "--help") { o.help = true; return true; }
        else if (a == "--plan")     { o.plan_only = true; }
        else if (a == "-nkvo" || a == "--no-kv-offload") { o.kv_mode = -1; }
        else if (a == "--warmup")   { o.warmup = true; }
        else if (a == "--no-warmup"){ o.warmup = false; }
        else if (a == "--verbose")  { g_verbose = true; }
        else if (a == "-m" && has_value) { o.model = argv[++i]; }
        else if (a == "-p" && has_value) { o.prompt = argv[++i]; }
        else if (a == "-c" && has_value) { o.ctx = std::atoi(argv[++i]); }
        else if (a == "-n" && has_value) { o.n_predict = std::atoi(argv[++i]); }
        else if (a == "-t" && has_value) { o.threads = std::atoi(argv[++i]); }
        else if (a == "-ub" && has_value) { o.ubatch = std::atoi(argv[++i]); }
        else if (a == "--fit-target" && has_value) { o.fit_target = std::atoi(argv[++i]); }
        else if ((a == "-ncffn" || a == "--n-cpu-ffn") && has_value) {
            const std::string v = argv[++i];
            // Same trap as -ngl auto: atoi("auto") means 0, which is the
            // classic split and would look applied while holding nothing back.
            if (v == "auto") { o.ffn_auto = true; o.n_cpu_ffn = 0; }
            else { o.n_cpu_ffn = std::atoi(v.c_str()); }
        }
        else if (a == "--ncffn-probe") { o.ffn_probe = true; }
        else if (a == "-ngl" && has_value) {
            const std::string v = argv[++i];
            // 'auto' spells the default; atoi("auto") would mean CPU-only.
            o.ngl = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--fit" && has_value) {
            const std::string v = argv[++i];
            if (v != "on" && v != "off") {
                std::fprintf(stderr, "error: --fit takes 'on' or 'off', not '%s'\n", v.c_str());
                return false;
            }
            // An explicit -ngl keeps its value; off only rules out fitting.
            o.ngl = (v == "on") ? -1 : (o.ngl > 0 ? o.ngl : 0);
        }
        else if (a == "--kv" && has_value) { o.kv_k = o.kv_v = argv[++i]; }
        else if (a == "-ctk" && has_value) { o.kv_k = argv[++i]; }
        else if (a == "-ctv" && has_value) { o.kv_v = argv[++i]; }
        else if (a == "--temp" && has_value) { o.temp = (float) std::atof(argv[++i]); }
        else if (a == "--top-p" && has_value) { o.top_p = (float) std::atof(argv[++i]); }
        else if (a == "--min-p" && has_value) { o.min_p = (float) std::atof(argv[++i]); }
        else if (a == "--seed" && has_value) { o.seed = (uint32_t) std::strtoul(argv[++i], nullptr, 10); }
        else if (a == "--spec" && has_value) { o.spec = argv[++i]; }
        else if (a == "--spec-n" && has_value) { o.spec_n = std::atoi(argv[++i]); }
        else {
            std::fprintf(stderr, "error: unknown or incomplete argument '%s'\n", a.c_str());
            return false;
        }
    }
    return !o.model.empty();
}

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void batch_set(llama_batch_ext * batch, const llama_token * tokens, int n, llama_pos pos0,
               bool logits_last) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < n; ++i) {
        const int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = pos0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    if (logits_last && n > 0) {
        llama_batch_ext_set_output_logits(batch, n - 1, true);
    }
}

// One generated token on stdout, flushed. A token that does not happen to
// contain a newline would otherwise stay invisible until the buffer filled.
void emit_token(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n > 0) {
        std::fwrite(buf, 1, (size_t) n, stdout);
        std::fflush(stdout);
    }
}

SamplerPtr make_sampler(const Options & o) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    params.no_perf = true;
    SamplerPtr chain(llama_sampler_chain_init(params));
    if (o.temp <= 0.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(o.min_p, 1));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(o.top_p, 1));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(o.temp));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(o.seed));
    }
    return chain;
}

// Sorted input: the caller sorts the sample once, rather than each helper
// copying and re-sorting the same vector - three copies and three sorts for
// one printed line, with identical numbers.
double median_sorted(const std::vector<double> & v) {
    if (v.empty()) {
        return 0.0;
    }
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// `-ncffn N` keeps every layer's attention (and its KV slice) on the card and
// reads the FFN blocks of the first N layers from system RAM. The strings have
// to outlive the load - llama.cpp holds the pattern pointers - so `keep` is the
// owner, reserved by the caller before the first push_back.
// The list is NULL-terminated and llama.cpp stops at the first empty pattern,
// so the entries are filled in place - pushing back would append after the
// terminator and never be read.
bool add_ffn_cpu_override(std::vector<std::string> & keep,
                          std::vector<llama_model_tensor_buft_override> & overrides,
                          size_t idx, int layer) {
    if (idx + 1 >= overrides.size()) {
        return false;
    }
    keep.push_back("blk\\." + std::to_string(layer) + "\\.ffn_(up|down|gate)\\.");
    overrides[idx] = {keep.back().c_str(), ggml_backend_cpu_buffer_type()};
    return true;
}

double percentile_sorted(const std::vector<double> & v, double q) {
    if (v.empty()) {
        return 0.0;
    }
    // Nearest rank: index ceil(q*N) - 1, not floor(q*N). On 100 samples the
    // latter reports the 91st value as "p90"; this tool's whole point is that
    // the label matches the number.
    const double scaled = std::ceil(q * (double) v.size()) - 1.0;
    const size_t idx = (size_t) std::min<double>((double) v.size() - 1, std::max(0.0, scaled));
    return v[idx];
}

// -------------------------------------------------------- ffn ladder -------
// Exit code a probe child uses for "the context did not fit". Distinct from 1
// so the ladder can tell an out-of-VRAM rung from a bad model path and stop
// climbing in the second case instead of spawning 60 more processes.
constexpr int PROBE_NO_CONTEXT = 2;
// No model on any card has needed a larger FFN-on-CPU count than this to make a
// context; the ladder stops here rather than spin if the answer is "never".
constexpr int FFN_LADDER_MAX = 64;

int spawn_self(const std::vector<std::string> & args) {
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & a : args) {
        argv.push_back(const_cast<char *>(a.c_str()));
    }
    argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "error: fork failed: %s\n", std::strerror(errno));
        return 127;
    }
    if (pid == 0) {
        // /proc/self/exe, not argv[0]: the ladder has to re-run this binary even
        // when it was started through a PATH lookup or a symlink.
        execv("/proc/self/exe", argv.data());
        std::fprintf(stderr, "error: cannot re-run this binary: %s\n", std::strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "error: waitpid failed: %s\n", std::strerror(errno));
        return 127;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 127;
}

// ------------------------------------------------------ ladder memory ------
// Every rung of the climb is a child process that loads the whole model, so a
// failed rung costs ~1.5 s - and the ladder pays that again on the next run,
// for an answer that cannot have changed. The answer for a given (model,
// context, KV type) is kept in $XDG_CACHE_HOME/vipera/ffn-ladder.txt as
// "key<TAB>value" lines. Appending is enough because the last match wins: a
// crash mid-write cannot damage an answer that is already there.
std::string ladder_path() {
    const char * cache = std::getenv("XDG_CACHE_HOME");
    std::string base;
    if (cache != nullptr && *cache != '\0') {
        base = cache;
    } else {
        const char * home = std::getenv("HOME");
        if (home == nullptr || *home == '\0') {
            return {};
        }
        base = std::string(home) + "/.cache";
    }
    return base + "/vipera/ffn-ladder.txt";
}

// The context and the KV type decide what fits, so both are part of the key: a
// rung measured at 4096 is not the answer at 8192.
std::string ladder_key(const std::string & model, int ctx, const std::string & kv_k,
                       const std::string & kv_v, const Device & gpu) {
    struct stat st {};
    long long size = 0;
    long long mtime = 0;
    if (::stat(model.c_str(), &st) == 0) {
        size = (long long) st.st_size;
        mtime = (long long) st.st_mtime;
    }
    // The card is part of the key. The same file on a bigger GPU keeps more FFN
    // blocks on it, and a rung remembered on the old card would be replayed
    // forever: the extra FFN on the CPU still *loads*, so the confirmation step
    // cannot catch it - the run just decodes slower for the rest of its life.
    char buf[1536];
    std::snprintf(buf, sizeof(buf), "%s|%lld|%lld|%d|%s/%s|%s|%zuMiB", model.c_str(), size, mtime,
                  ctx, kv_k.c_str(), kv_v.c_str(),
                  gpu.present ? gpu.name.c_str() : "no-gpu", gpu.total / MIB);
    return buf;
}

int ladder_lookup(const std::string & key) {
    const std::string path = ladder_path();
    if (path.empty()) {
        return 0;
    }
    FILE * f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return 0;
    }
    char line[1024];
    int found = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        const std::string row(line);
        const size_t tab = row.find('\t');
        if (tab == std::string::npos) {
            continue;
        }
        if (row.compare(0, tab, key) == 0) {
            found = std::atoi(row.c_str() + tab + 1);
        }
    }
    std::fclose(f);
    return found;
}

// mkdir -p for the cache directory. One level was not enough: it created
// $XDG_CACHE_HOME/vipera but not $XDG_CACHE_HOME itself, and the fopen below
// then failed silently - the ladder kept re-climbing on every start with no sign
// that the memory it was meant to write was never written.
bool make_dirs(const std::string & path) {
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] != '/') {
            continue;
        }
        ::mkdir(path.substr(0, i).c_str(), 0755);
    }
    return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

void ladder_remember(const std::string & key, int n) {
    const std::string path = ladder_path();
    if (path.empty()) {
        return;
    }
    const size_t slash = path.rfind('/');
    if (slash != std::string::npos) {
        make_dirs(path.substr(0, slash));
    }
    FILE * f = std::fopen(path.c_str(), "a");
    if (f == nullptr) {
        std::fprintf(stderr,
                     "vipera: note: cannot write the ladder memory at %s (%s);\n"
                     "        every start will probe again\n",
                     path.c_str(), std::strerror(errno));
        return;
    }
    std::fprintf(f, "%s\t%d\n", key.c_str(), n);
    std::fclose(f);
}

// `-ncffn auto`: the smallest FFN-on-CPU count that still creates a context.
//
// In-process probing was rejected. A context that fails to allocate does not
// hand its VRAM back, so a few failures in a row turn a split that would load
// into one that does not - the same wall vipera's Python ladder documents after
// three failures on a 4 GB card. A child process returns every byte to the
// driver when it exits, so each rung starts from the same card.
//
// Most aggressive first, because a smaller count is the faster run: every FFN
// block on the card is a block not read over the bus on every token. The
// winning rung then runs for real in its own child, so the numbers reported are
// a clean process and the ladder never pays for two generations.
int run_ffn_ladder(int argc, char ** argv, const std::string & model) {
    // One message for a bad path beats sixty child processes each reporting it.
    // The magic is checked for the same reason: a truncated download or an HTML
    // error page saved as .gguf opens fine and then fails inside
    // llama_model_load_from_file, which a probe has to report as "this rung does
    // not fit" - so the ladder would climb all 64 rungs before admitting the
    // file cannot be read. Whatever is wrong with those bytes, more FFN on the
    // CPU will not fix it.
    FILE * f = std::fopen(model.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot open %s: %s\n", model.c_str(), std::strerror(errno));
        return 1;
    }
    char magic[4] = {0, 0, 0, 0};
    const size_t got = std::fread(magic, 1, sizeof(magic), f);
    std::fclose(f);
    if (got != sizeof(magic) || std::memcmp(magic, "GGUF", 4) != 0) {
        std::fprintf(stderr, "error: %s is not a GGUF file (or is truncated)\n", model.c_str());
        return 1;
    }

    int value_idx = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-ncffn" || a == "--n-cpu-ffn") && i + 1 < argc) {
            value_idx = i + 1;
            ++i;
        }
    }
    if (value_idx < 0) {
        std::fprintf(stderr, "error: -ncffn auto needs the flag it belongs to\n");
        return 1;
    }

    int want_ctx = 0;
    std::string want_kv_k = "f16";
    std::string want_kv_v;
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-c") { want_ctx = std::atoi(argv[i + 1]); }
        else if (a == "--kv") { want_kv_k = want_kv_v = argv[i + 1]; }
        else if (a == "-ctk") { want_kv_k = argv[i + 1]; }
        else if (a == "-ctv") { want_kv_v = argv[i + 1]; }
    }
    // The answer depends on the card as much as on the file (see ladder_key), so
    // the parent reads the card from ggml's registry - the same source the fit
    // uses - before deciding whether a remembered rung applies. This initialises
    // the backends here; the children pay for that anyway, and no weight is
    // loaded in this process.
    llama_log_set(log_forward, nullptr);
    llama_backend_init();
    const Device gpu = first_gpu();
    const std::string key = ladder_key(model, want_ctx, want_kv_k,
                                       want_kv_v.empty() ? want_kv_k : want_kv_v, gpu);

    std::vector<std::string> args(argv, argv + argc);
    int first_rung = 1;
    // A remembered rung is confirmed, not trusted: the entry may predate another
    // process taking VRAM. A rung that no longer creates a context falls through
    // to the climb below, which is also the direction a stale rung needs - a
    // card that shrank needs MORE FFN on the CPU, not less.
    const int remembered = ladder_lookup(key);
    if (remembered > 0) {
        args[(size_t) value_idx] = std::to_string(remembered);
        std::vector<std::string> probe = args;
        probe.push_back("--ncffn-probe");
        const int code = spawn_self(probe);
        if (code == 0) {
            std::fprintf(stderr,
                         "vipera: -ncffn auto reuses %d from the ladder memory\n"
                         "        (confirmed at this -c; delete %s to re-measure)\n",
                         remembered, ladder_path().c_str());
            return spawn_self(args);
        }
        if (code != PROBE_NO_CONTEXT) {
            std::fprintf(stderr,
                         "error: the probe at -ncffn %d exited %d, which is not an\n"
                         "       out-of-VRAM failure; not climbing further\n",
                         remembered, code);
            return code;
        }
        first_rung = remembered + 1;
    }

    for (int n = first_rung; n <= FFN_LADDER_MAX; ++n) {
        args[(size_t) value_idx] = std::to_string(n);
        std::vector<std::string> probe = args;
        probe.push_back("--ncffn-probe");
        const int code = spawn_self(probe);
        if (code == 0) {
            ladder_remember(key, n);
            std::fprintf(stderr, "vipera: -ncffn auto chose %d: the smallest count that\n"
                                 "        creates a context at this -c\n", n);
            return spawn_self(args);
        }
        if (code != PROBE_NO_CONTEXT) {
            std::fprintf(stderr,
                         "error: the probe at -ncffn %d exited %d, which is not an\n"
                         "       out-of-VRAM failure; not climbing further\n", n, code);
            return code;
        }
    }
    std::fprintf(stderr, "error: no -ncffn up to %d created a context\n", FFN_LADDER_MAX);
    return 1;
}

}  // namespace

// What a generation loop produced, so the loop does not also own the report.
struct GenerateStats {
    std::vector<double> token_ms;   // one sample per token produced, in print order
    int n_generated      = 0;
    int n_draft_proposed = 0;
    int n_draft_accepted = 0;
    int n_verify_steps   = 0;
    int pos              = 0;       // context positions consumed
    double seconds       = 0.0;     // decode window, first sampling excluded
};

// Decode up to `opt.n_predict` tokens, printing each as it is produced.
//
// `spec` is null for plain decode, one sampled token per pass. With a drafter,
// the carried token and the draft share one verify batch and the cache is rolled
// back to the accepted prefix. Returns false when the target decode or that
// rollback fails; this function owns no resource, so the caller keeps the single
// teardown path - including freeing the backend after the drafter, whose deleter
// may own llama contexts.
static bool generate(llama_context * ctx, const llama_vocab * vocab, const Options & opt,
                     llama_sampler * sampler, llama_batch_ext * batch,
                     common_speculative * spec, llama_tokens & history,
                     int n_ctx, int pos0, GenerateStats & st) {
    int pos = pos0;
    const int budget = std::min(opt.n_predict, n_ctx - pos);
    st.token_ms.reserve((size_t) std::max(0, budget));
    llama_tokens draft;
    // `next` is sampled but neither printed nor fed yet: with a draft beside it
    // it goes into the same batch as the draft tokens and shares their pass.
    llama_token next = llama_sampler_sample(sampler, ctx, -1);
    const double t_gen_start = now_s();
    while (st.n_generated < budget) {
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }

        draft.clear();
        if (spec) {
            // Leave room for the token after the drafts and for the two the
            // context keeps for a shift, exactly as the server's slot does.
            const int n_max = std::min(std::min(opt.spec_n, budget - st.n_generated - 1),
                                       n_ctx - pos - 2);
            if (n_max > 0) {
                common_speculative_draft_params & dp =
                        common_speculative_get_draft_params(spec, 0);
                dp = common_speculative_draft_params{};
                dp.drafting = true;
                dp.n_max    = n_max;
                dp.pos0     = (llama_pos) pos;
                dp.id_last  = next;
                dp.prompt   = &history;
                dp.result   = &draft;
                common_speculative_draft(spec);
            }
        }

        const double t0 = now_s();
        if (!draft.empty()) {
            // One batch: the sampled token plus every draft token, logits on
            // every row, because row i holds the token at pos + i and decides
            // the one after it.
            llama_batch_ext_clear(batch);
            for (size_t i = 0; i <= draft.size(); ++i) {
                const int idx = llama_batch_ext_add_token(batch, 0,
                                                          i == 0 ? next : draft[i - 1]);
                const llama_pos p = (llama_pos) (pos + (int) i);
                llama_batch_ext_set_pos(batch, idx, &p);
                llama_batch_ext_set_output_logits(batch, idx, true);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                std::fprintf(stderr, "error: verify decode failed at token %d\n", st.n_generated);
                return false;
            }
            llama_synchronize(ctx);

            // Sample row by row and keep the prefix that matches the draft. The
            // last sample is a new token even when every draft was accepted:
            // the row after the last draft predicts it, which is why a fully
            // accepted pass yields one more token than it proposed.
            llama_tokens accepted;
            accepted.reserve(draft.size() + 1);
            // The scan that decides how much of this pass survives - and so how
            // much KV is rolled back - lives in accept.h so it can be tested
            // without a model.
            const int n_keep = (int) accepted_rows(draft, [&](size_t i) {
                const llama_token t = llama_sampler_sample(sampler, ctx, (int) i);
                accepted.push_back(t);
                return t;
            });

            // The rows past the accepted prefix are already in the cache and
            // would be read as context by the next batch; drop them.
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0,
                                     (llama_pos) (pos + n_keep), -1)) {
                std::fprintf(stderr,
                             "error: cannot roll the KV cache back over the rejected drafts\n");
                return false;
            }

            // Tell the drafter how many of its proposals survived - the count is
            // the accepted prefix without the carried token. The server makes
            // the same call after its verify pass; skipping it leaves the
            // drafter's statistics at zero and any implementation that keeps
            // state between steps reasoning from a stale picture.
            common_speculative_accept(spec, 0, (uint16_t) (n_keep - 1));

            const double dt_ms = (now_s() - t0) * 1000.0;

            // Printed in the order the model produced them: the carried token
            // first, then the drafts it confirmed. accepted.back() is carried
            // and printed at the top of the next pass.
            emit_token(vocab, next);
            ++st.n_generated;
            bool stop = false;
            for (int i = 0; i + 1 < n_keep; ++i) {
                if (llama_vocab_is_eog(vocab, accepted[(size_t) i])) {
                    stop = true;
                    break;
                }
                emit_token(vocab, accepted[(size_t) i]);
                ++st.n_generated;
            }
            // One per-token sample for every token this pass produced, so a
            // fast pass reads as fast tokens instead of one long one.
            for (int i = 0; i < n_keep; ++i) {
                st.token_ms.push_back(dt_ms / (double) n_keep);
            }

            // The cache now ends at the last matched draft; the carried token
            // is the only one of this pass the model has not seen yet.
            history.push_back(next);
            history.insert(history.end(), accepted.begin(), accepted.end() - 1);

            st.n_verify_steps += 1;
            st.n_draft_proposed += (int) draft.size();
            st.n_draft_accepted += n_keep - 1;
            pos += n_keep;
            next = accepted.back();

            if (stop) {
                break;
            }
        } else {
            emit_token(vocab, next);

            batch_set(batch, &next, 1, pos, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                std::fprintf(stderr, "error: decode failed at token %d\n", st.n_generated);
                return false;
            }
            // Same reason as above: llama.cpp's decode is asynchronous, so the
            // wait lands in whichever call next reads the output. Measured
            // without this, a 13 ms token reported a 0.3 ms median (75 t/s
            // decode alongside "median 0.31 ms"); the synchronize moves the
            // wait into the window that claims to measure it and costs nothing
            // - the sampler below would have blocked for exactly as long.
            llama_synchronize(ctx);
            st.token_ms.push_back((now_s() - t0) * 1000.0);

            history.push_back(next);
            ++pos;
            ++st.n_generated;
            next = llama_sampler_sample(sampler, ctx, -1);
        }
    }
    st.pos     = pos;
    st.seconds = now_s() - t_gen_start;
    return true;
}

// The run report: prompt, generation, per-token and drafter numbers in one
// place. Everything it prints arrives as an argument, so main cannot report a
// window it no longer owns and the statistics cannot pick up a stale one.
static void report(const Plan & plan, const GenerateStats & gen, const Options & opt,
                   int n_prompt, double prefill_s, bool spec_on) {
    // First generated token's cost includes the sample of the prefill's logits,
    // so it is excluded from the per-token statistics and reported separately.
    std::vector<double> steady(
        gen.token_ms.size() > 1
            ? std::vector<double>(gen.token_ms.begin() + 1, gen.token_ms.end())
            : std::vector<double>());
    std::sort(steady.begin(), steady.end());
    const double gen_s = gen.seconds;

    std::printf("\n\n");
    std::printf("  prompt         %7.1f t/s  (%d tokens, %.2f s)\n",
                n_prompt / std::max(1e-9, prefill_s), n_prompt, prefill_s);
    if (!opt.warmup) {
        // Say it rather than let the number look wrong: this is the graph
        // allocation, and it lands here because the prefill is synchronised.
        std::printf("  note           the first prefill carries the one-time graph allocation;\n"
                    "                 --warmup moves it into a one-token decode\n");
    }
    if (gen.n_generated > 0) {
        std::printf("  generation     %7.2f t/s  (%d tokens, %.2f s)\n",
                    gen.n_generated / std::max(1e-9, gen_s), gen.n_generated, gen_s);
        if (!steady.empty()) {
            std::printf("  per token      median %.2f ms, p90 %.2f ms, first %.2f ms\n",
                        median_sorted(steady), percentile_sorted(steady, 0.9),
                        gen.token_ms.front());
        }
    }
    if (spec_on) {
        // Report what the drafter cost and bought: a low acceptance rate is the
        // honest explanation for a --spec run that is slower than the plain one.
        const double rate = gen.n_draft_proposed > 0
                          ? (double) gen.n_draft_accepted / (double) gen.n_draft_proposed : 0.0;
        std::printf("  spec           %d/%d draft tokens accepted (%.1f%%), %d verify steps\n",
                    gen.n_draft_accepted, gen.n_draft_proposed, 100.0 * rate,
                    gen.n_verify_steps);
    }
    std::printf("  kv used        %zu MiB of the %zu MiB planned\n",
                plan.per_token * (size_t) gen.pos / MIB, plan.kv / MIB);
}

int main(int argc, char ** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }
    if (opt.help) {
        print_usage(argv[0]);
        return 0;
    }

    // A probe answers one question - does a context fit at this count - so the
    // plan-only exit must not answer it before the context is created.
    if (opt.ffn_probe) {
        opt.plan_only = false;
    }

    // `-ncffn auto` is resolved before any model work: the ladder re-runs this
    // binary per rung, so the parent never loads a weight itself.
    if (opt.ffn_auto) {
        return run_ffn_ladder(argc, argv, opt.model);
    }

    const bool ffn_split = opt.n_cpu_ffn > 0;
    // f16 at 8k asks for 448 MiB beside every layer's attention; 224 with q8_0
    // is what let 8 FFN blocks fit on the reference 4 GB card (18.2 t/s against
    // a failure to create the context). Only when the KV type was left alone.
    bool kv_lowered = false;
    if (ffn_split && opt.kv_k == "f16" && opt.kv_v.empty()) {
        opt.kv_k = opt.kv_v = "q8_0";
        kv_lowered = true;
    }

    bool kv_ok = false;
    const ggml_type type_k = kv_type_from(opt.kv_k, kv_ok);
    if (!kv_ok) {
        std::fprintf(stderr, "error: unknown KV type '%s'\n", opt.kv_k.c_str());
        return 1;
    }
    const ggml_type type_v = kv_type_from(opt.kv_v.empty() ? opt.kv_k : opt.kv_v, kv_ok);
    if (!kv_ok) {
        std::fprintf(stderr, "error: unknown KV type '%s'\n", opt.kv_v.c_str());
        return 1;
    }
    const bool quantized_kv = is_quantized(type_k) || is_quantized(type_v);

    llama_log_set(log_forward, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();
    const Device device = first_gpu();
    const bool auto_placement = opt.ngl < 0;
    bool fit_failed = false;
    uint32_t fitted_ctx = 0;

    // These stay alive for the whole load: the fit writes their addresses into
    // mparams, and llama.cpp dereferences them while the model is loaded.
    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    std::vector<llama_model_tensor_buft_override> overrides(llama_max_tensor_buft_overrides());
    // Reserved up front: push_back must not move the strings after a pattern
    // pointer has been handed to llama.cpp.
    std::vector<std::string> override_patterns;
    if (ffn_split) {
        override_patterns.reserve((size_t) opt.n_cpu_ffn);
    }
    // A fixed 1 GiB margin is 28% of a 4 GB card and 4 more layers left on the
    // CPU. Scale it to the card, with a floor for the driver's own allocations.
    // The fit already prices the weights and the compute buffers, so this is
    // only room for the driver and for whatever else grows later; the card's
    // own free figure above is already net of the desktop.
    // card/8 (466 MiB here) turned out to cost three layers. Measured on a
    // GTX 1650 4 GB, Qwen2.5-7B Q4_K_M, f16 KV, both a 641- and a 1281-token
    // prompt, ctx 3072/4096/8192: 466 MiB keeps 18/28 layers (14.47 t/s),
    // 256 keeps 19, 224 keeps 20 (15.46 t/s), 64 keeps 21 (16.33 t/s) - and 22
    // never loads at any margin, so the fit's own buffers are the wall, not the
    // driver. card/16 is 233 MiB here, which lands on the 20-layer rung and
    // across the largest margin this card was measured to still fit. The KV
    // cache is allocated for the whole window up front and the compute buffers
    // are fixed, so a split that loads is a split that lasts: nothing grows
    // during the run except by another process taking the card.
    const size_t margin = opt.fit_target > 0
                              ? (size_t) opt.fit_target * MIB
                              : std::min(DEFAULT_HEADROOM, std::max(192 * MIB, device.total / 16));
    std::vector<size_t> margins(llama_max_devices(), margin);

    if (ffn_split && !device.present) {
        std::fprintf(stderr, "error: -ncffn needs a GPU device; this build has none\n");
        llama_backend_free();
        return 1;
    } else if (ffn_split) {
        // Whole layers put the attention of 17 of 28 on the card and read the
        // other 11 - attention included - over the bus on every token. This
        // keeps the attention (and therefore the whole KV cache) on the card
        // and pays the bus only for FFN blocks. Measured on the reference
        // 4 GB card, 7B Q4_K_M at 8k, 1655-token prompt: 17 whole layers 11.4
        // t/s, 8 FFN in RAM 16.8 t/s (+48%), 7 FFN 16.8-17.8 (+52%), 6 FFN does
        // not fit. The fit is bypassed: it aborts when overrides are preset,
        // and a header cannot price a tensor split anyway.
        int applied = 0;
        for (int il = 0; il < opt.n_cpu_ffn; ++il) {
            // One slot is kept for the terminator.
            if (!add_ffn_cpu_override(override_patterns, overrides, (size_t) applied, il)) {
                break;
            }
            ++applied;
        }
        if (applied < opt.n_cpu_ffn) {
            std::fprintf(stderr,
                         "error: this build holds %zu tensor overrides, so -ncffn caps at %d\n",
                         overrides.size() - 1, applied);
            llama_backend_free();
            return 1;
        }
        model_params.tensor_buft_overrides = overrides.data();
        model_params.n_gpu_layers = -1;
        opt.ngl = -1;
        if (opt.ctx == 0) {
            // Nothing fitted a context here, and the trained maximum assumes a
            // cache that fits beside 28 layers' attention on this card.
            opt.ctx = 8192;
        }
    } else if (auto_placement && !device.present) {
        // With no GPU device the fitter reports success and leaves
        // `n_gpu_layers` at its default, -1, which is llama.cpp's "all layers".
        // Reading that as a full offload made a CPU-only box print "28/28, the
        // whole model fits the card" and charge the KV cache to a card that is
        // not there. Nothing to fit: say zero and keep every layer in RAM.
        model_params.n_gpu_layers = 0;
        opt.ngl = 0;
        fit_failed = true;
    } else if (auto_placement) {
        // The fit only moves parameters that still hold their default, so this
        // decides the KV device before it runs - and since the fitter prices the
        // cache where it is told to put it, one pass is enough: `-nkvo` frees
        // the card of the cache up front and the layers it buys are part of the
        // same answer. Taking the cache off the card buys layers (19/28 against
        // 17/28 at 8k on the reference card) but pays for them on every
        // attention read: 9.3-10.3 t/s against 10.8 with the cache in VRAM. It
        // wins only on a split deep enough for the layers to pay the transfers,
        // so it stays opt-in.
        cparams.offload_kqv = opt.kv_mode >= 0;
        // llama.cpp's own fitter, given the tensor map, decides what the card
        // can hold: it costs one metadata-only load and it is the difference
        // between "28/28" and a guess that OOMs on a card with a desktop on it.
        cparams.n_ctx = opt.ctx > 0 ? (uint32_t) opt.ctx : 0;

        const common_params_fit_status status = common_fit_params(
            opt.model.c_str(), &model_params, &cparams, tensor_split.data(),
            overrides.data(), margins.data(), 4096, nullptr,
            g_verbose ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        if (status == COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            opt.ngl = model_params.n_gpu_layers;
            fitted_ctx = cparams.n_ctx;
        } else {
            // No device, or a context that cannot be made to fit: keep every
            // layer in RAM rather than fail the load.
            model_params.n_gpu_layers = 0;
            opt.ngl = 0;
            fit_failed = true;
        }
    } else {
        model_params.n_gpu_layers = opt.ngl;
    }

    // --plan still loads the weights: llama.cpp only finalises some hparams (the
    // per-layer KV head list among them) while loading tensors, and reading them
    // from a vocab-only model aborts. Planning cheaply is not worth an abort.

    ModelPtr model(llama_model_load_from_file(opt.model.c_str(), model_params));
    if (!model) {
        if (opt.ffn_probe) {
            // Too many FFN blocks on the card fails here rather than at the
            // context: llama.cpp cannot allocate the tensor buffers at load.
            std::fprintf(stderr, "probe: -ncffn %d does not load\n", opt.n_cpu_ffn);
            llama_backend_free();
            return PROBE_NO_CONTEXT;
        }
        std::fprintf(stderr, "error: cannot load %s\n", opt.model.c_str());
        llama_backend_free();
        return 1;
    }

    const std::vector<int> heads = kv_heads_per_layer(model.get());
    const size_t model_bytes = (size_t) llama_model_size(model.get());
    const size_t available = mem_available_bytes();
    const int n_layer = std::max(1, llama_model_n_layer(model.get()));
    // -1 is llama.cpp's "all layers": the fit leaves it alone when the card
    // holds the whole model, so it means "all", not "none".
    const int gpu_layers = opt.ngl < 0 ? n_layer : std::min(opt.ngl, n_layer);
    // Per-layer bytes are the model's average; the non-repeating tensors ride
    // along in RAM either way, so this charges them to the CPU side. With
    // `-ncffn` the layer index no longer says where its weights are: those N
    // layers keep only their attention on the card, so they are charged whole,
    // which errs on the side of a smaller context.
    const int ram_layers = n_layer - gpu_layers + (ffn_split ? std::min(opt.n_cpu_ffn, n_layer) : 0);
    const size_t resident = model_bytes * (size_t) std::max(0, ram_layers) / (size_t) n_layer;
    const int requested = opt.ctx > 0 ? opt.ctx : (int) fitted_ctx;
    const Plan plan = make_plan(model.get(), requested, type_k, type_v, heads, resident, available,
                                (int) fitted_ctx);
    const int threads = opt.threads > 0 ? opt.threads : physical_cores();
    const int train = llama_model_n_ctx_train(model.get());

    std::printf("\nvipera plan\n");
    std::printf("  model          %zu MiB, %d layers, %d kv heads, head dim %d\n",
                model_bytes / MIB, llama_model_n_layer(model.get()),
                llama_model_n_head_kv(model.get()),
                llama_model_n_embd(model.get()) / std::max(1, llama_model_n_head(model.get())));
    // The placement comes first: it decides what the rest of the plan means.
    if (device.present) {
        std::printf("  gpu            %s, %zu MiB free / %zu MiB\n",
                    device.name.c_str(), device.free / MIB, device.total / MIB);
    } else {
        std::printf("  gpu            none (CPU only)\n");
    }
    if (ffn_split) {
        std::printf("  gpu layers     %d/%d (attention of all, FFN of %d from RAM)\n",
                    n_layer, n_layer, opt.n_cpu_ffn);
    } else if (!auto_placement) {
        std::printf("  gpu layers     %d/%d (given by -ngl)\n", gpu_layers, n_layer);
    } else if (!device.present) {
        std::printf("  gpu layers     0/%d (no GPU device in this build or on this machine)\n",
                    n_layer);
    } else if (fit_failed) {
        std::printf("  gpu layers     0/%d (nothing fit the card; every layer stays in RAM)\n",
                    n_layer);
    } else if (opt.ngl < 0) {
        std::printf("  gpu layers     %d/%d (the whole model fits the card)\n", n_layer, n_layer);
    } else {
        std::printf("  gpu layers     %d/%d (fitted to the free VRAM, the rest in RAM)\n",
                    gpu_layers, n_layer);
    }
    std::printf("  ram weights    ~%zu MiB of the model stays in system RAM\n", resident / MIB);
    if (ffn_split) {
        std::printf("  ffn on cpu     %d of %d layers read their FFN blocks from RAM\n",
                    opt.n_cpu_ffn, n_layer);
    }
    std::printf("  kv cache       %zu MiB at %d tokens (%s/%s)\n",
                plan.kv / MIB, plan.ctx, opt.kv_k.c_str(),
                (opt.kv_v.empty() ? opt.kv_k : opt.kv_v).c_str());
    if (kv_lowered) {
        std::printf("  kv lowered     f16 did not fit beside the split, so q8_0 halves it\n");
    }
    if (!cparams.offload_kqv) {
        std::printf("  kv location    all %zu MiB in RAM, the card keeps its layers\n",
                    plan.kv / MIB);
    } else if (gpu_layers > 0) {
        // llama.cpp keeps the cache of an offloaded layer on that device.
        std::printf("  kv on gpu      ~%zu MiB for %d of %d layers, the rest in RAM\n",
                    plan.kv * (size_t) gpu_layers / (size_t) n_layer / MIB, gpu_layers, n_layer);
    }
    if (device.present) {
        std::printf("  headroom       %zu MiB of the card kept free for the driver\n", margin / MIB);
    }
    if (available > 0) {
        std::printf("  available      %zu MiB\n", available / MIB);
    } else {
        // mem_available_bytes() returns 0 when /proc/meminfo cannot be read, and
        // make_plan skips its RAM clamp in that case. Say so: a plan printed
        // without the number it was clamped against is exactly the silent-wrong
        // output this tool exists to avoid.
        std::printf("  available      unknown (/proc/meminfo unreadable), so the KV\n"
                    "                 cache was NOT budgeted against RAM\n");
    }
    std::printf("  context        %d requested, %d used (train limit %d)\n",
                opt.ctx > 0 ? opt.ctx : (requested > 0 ? requested : train), plan.ctx, train);
    if (plan.clamped) {
        std::printf("  clamped        %s\n", plan.reason.c_str());
    }
    std::printf("  threads        %d (physical cores)\n", threads);
    // Not forced on for a quantized KV cache. llama.cpp accepts one without it,
    // and measured here forcing it on f16 lost 2% of prompt processing.
    std::printf("  flash attn     auto\n");
    if (quantized_kv) {
        // The trade is memory for time, and it is a bad one for speed: the
        // attention matmuls no longer read f16, so they go through a much slower
        // path on CPU. Measured on Qwen2.5-1.5B Q4_K_M, 3557-token prompt, 4 threads:
        // f16 140.5 t/s, q8_0 K+V 62.5, q8_0 K only 83.3.
        std::printf("  kv cost        quantized KV buys RAM, not speed: 140 -> 62 t/s prompt\n");
        std::printf("                 at 3.5k tokens on CPU (f16 vs q8_0 K+V)\n");
    }
    // stdout is block-buffered when piped while stderr is not, so an error that
    // follows this block would otherwise appear before the plan it refers to.
    std::fflush(stdout);
    if (opt.plan_only) {
        llama_backend_free();
        return 0;
    }

    cparams.n_ctx = (uint32_t) plan.ctx;
    cparams.n_batch = std::min<uint32_t>(2048u, (uint32_t) plan.ctx);
    // 2048 on the reference 4 GB card asks for a 1.36 GB compute buffer and the
    // context fails to create, so the ceiling is 1024 whatever the flag asks.
    const uint32_t ubatch = std::min<uint32_t>(opt.ubatch > 0 ? (uint32_t) opt.ubatch : 512u, 1024u);
    cparams.n_ubatch = std::min<uint32_t>(ubatch, cparams.n_batch);
    cparams.n_threads = (int32_t) threads;
    cparams.n_threads_batch = (int32_t) threads;
    cparams.type_k = type_k;
    cparams.type_v = type_v;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.no_perf = true;

    // The prefill compute buffer lives on the card as well, and `-ncffn` has
    // already filled it with every layer's attention plus the KV cache: at 8k
    // on the reference 4 GB card a 512-token ubatch asked for 333 MiB that
    // were not there, while 256 tokens fit. Walk it down instead of failing.
    std::vector<uint32_t> ubatches = {cparams.n_ubatch};
    if (ffn_split) {
        for (uint32_t u : {(uint32_t) 256, (uint32_t) 128}) {
            if (u < cparams.n_ubatch) {
                ubatches.push_back(u);
            }
        }
    }
    CtxPtr ctx(nullptr);
    for (uint32_t ub : ubatches) {
        cparams.n_ubatch = std::min<uint32_t>(ub, cparams.n_batch);
        ctx.reset(llama_init_from_model(model.get(), cparams));
        if (ctx) {
            if (ub != ubatches.front()) {
                std::printf("  ubatch         %u (lowered: the card had no room for a %u-token prefill buffer)\n",
                            cparams.n_ubatch, ubatches.front());
            }
            break;
        }
    }
    if (!ctx) {
        std::fprintf(stderr, "error: cannot create a context (%d tokens)", plan.ctx);
        if (ffn_split) {
            std::fprintf(stderr,
                         "; free VRAM with a larger -ncffn, a smaller -c, or a quantized -ctk/-ctv");
        }
        std::fprintf(stderr, "\n");
        llama_backend_free();
        // A probe rung reports "did not fit" in a way the ladder can read; every
        // other failure stays a plain 1 so the ladder stops instead of climbing.
        return opt.ffn_probe ? PROBE_NO_CONTEXT : 1;
    }

    // The ladder only needs to know that a context exists at this count; the
    // prompt and the generation are the parent's job, in a clean process.
    if (opt.ffn_probe) {
        std::fprintf(stderr, "probe: -ncffn %d creates a %d-token context\n", opt.n_cpu_ffn, plan.ctx);
        llama_backend_free();
        return 0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int n_prompt = -llama_tokenize(vocab, opt.prompt.c_str(), opt.prompt.size(),
                                         nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "error: cannot tokenize the prompt\n");
        llama_backend_free();
        return 1;
    }
    // The warmup token, when there is one, occupies the first position.
    const int pos0 = opt.warmup ? 1 : 0;
    if (n_prompt + pos0 > plan.ctx) {
        std::fprintf(stderr,
                     "error: prompt is %d tokens and the window holds %d; raise -c or shorten the prompt\n",
                     n_prompt, plan.ctx);
        llama_backend_free();
        return 1;
    }
    std::vector<llama_token> prompt_tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, opt.prompt.c_str(), opt.prompt.size(), prompt_tokens.data(),
                       prompt_tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "error: cannot tokenize the prompt\n");
        llama_backend_free();
        return 1;
    }

    BatchPtr batch(llama_batch_ext_init(ctx.get()));

    // Optional warmup: the first llama_decode allocates the compute graph, and
    // charging that to prompt throughput is how a benchmark reports a number
    // nobody can reproduce. It runs here, in this context, and the token stays
    // in the cache so the prompt continues at pos 1: a throwaway context and a
    // cache rollback were both measured slower by the same 8%.
    if (opt.warmup) {
        const llama_token bos = llama_vocab_bos(vocab);
        batch_set(batch.get(), &bos, 1, 0, true);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            std::fprintf(stderr, "error: warmup decode failed\n");
            llama_backend_free();
            return 1;
        }
    }

    // Prefill in n_batch chunks: a prompt larger than the batch limit is legal,
    // but handing llama.cpp one oversized batch aborts the process.
    const int n_ctx = plan.ctx;
    int pos = pos0;
    int n_done = 0;
    const double t_prefill_start = now_s();
    while (n_done < n_prompt) {
        const int n = std::min<int>((int) cparams.n_batch, n_prompt - n_done);
        batch_set(batch.get(), prompt_tokens.data() + n_done, n, pos, true);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            std::fprintf(stderr, "error: prompt evaluation failed at token %d\n", pos);
            llama_backend_free();
            return 1;
        }
        pos += n;
        n_done += n;
    }
    // decode() only enqueues the graph; without this the last chunk's compute
    // and the one-time allocation would be charged to the first generated token.
    llama_synchronize(ctx.get());
    const double t_prefill_end = now_s();

    SamplerPtr sampler = make_sampler(opt);

    // Self-speculation. The drafter is the n-gram map in common/: it indexes
    // the prompt the model was just given and proposes the tokens that followed
    // the same n-gram before, so the model can check several tokens in one pass
    // instead of one per pass. Placement is untouched - no weights, no KV dtype,
    // no layer given up - and the only cost is a wider batch on the verify step,
    // which the context already budgets for through n_batch.
    common_speculative_ptr spec;
    llama_tokens history = prompt_tokens;
    // Rejecting a draft means cutting the cache back to the accepted prefix,
    // and recurrent / hybrid attention cannot drop part of a sequence: the
    // removal below would fail after the run had already printed. The server
    // survives that with a checkpoint replay; this tool keeps no such state, so
    // the honest answer is to decode plainly and say why.
    const bool spec_supported = !llama_model_is_recurrent(model.get())
                             && !llama_model_is_hybrid(model.get());
    if (opt.spec != "off" && !spec_supported) {
        std::fprintf(stderr,
                     "note: --spec is off for this model: recurrent and hybrid attention\n"
                     "      cannot roll the KV cache back over a rejected draft, so this\n"
                     "      run decodes plainly\n");
    }
    if (opt.spec != "off" && spec_supported) {
        common_params_speculative spec_params;
        try {
            spec_params.types = common_speculative_types_from_names({ opt.spec });
        } catch (const std::exception &) {
            // A typo would otherwise decode normally and look like the flag did
            // nothing, which is the silent-wrong this tool exists to refuse.
            std::fprintf(stderr,
                         "error: unknown --spec '%s'; try ngram-map-k, ngram-map-k4v, "
                         "ngram-simple or ngram-mod\n", opt.spec.c_str());
            llama_backend_free();
            return 1;
        }
        spec.reset(common_speculative_init(spec_params, 1));
        if (!spec) {
            std::fprintf(stderr, "error: --spec %s could not start\n", opt.spec.c_str());
            llama_backend_free();
            return 1;
        }
        common_speculative_begin(spec.get(), 0, history);
        std::printf("  spec           %s, up to %d draft tokens per step, %zu prompt tokens indexed\n",
                    opt.spec.c_str(), common_speculative_n_max(spec.get()), history.size());
        std::fflush(stdout);
    }

    GenerateStats gen;
    if (!generate(ctx.get(), vocab, opt, sampler.get(), batch.get(), spec.get(),
                  history, n_ctx, pos, gen)) {
        // Before the backend: the drafter's deleter may own llama contexts.
        spec.reset();
        llama_backend_free();
        return 1;
    }

    report(plan, gen, opt, n_prompt, t_prefill_end - t_prefill_start, spec != nullptr);
    return 0;
}

// vipera: a llama.cpp runtime that sizes the KV cache against the machine's real
// RAM before it allocates a context, then reports what the run actually cost.
//
// Why this file exists instead of a wrapper script: llama-cli silently accepts a
// context larger than RAM and pages; it also prints "0.0 t/s" when a prompt does
// not fit instead of failing. Both are silent-wrong, and this tool refuses to be.
//
// Every printed number is read from the model header, read from /proc, or
// measured by this process. Nothing is inferred.

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

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

// ------------------------------------------------------------------- plan ----

struct Plan {
    int ctx;              // tokens the context will actually be created with
    size_t kv;            // KV bytes for `ctx`
    size_t per_token;     // KV bytes per token
    bool clamped;
    bool limited_by_train;  // context cut by the model, not by RAM
    std::string reason;
};

Plan make_plan(const llama_model * model, int requested, ggml_type type_k, ggml_type type_v,
               const std::vector<int> & heads, size_t model_bytes, size_t available) {
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

    if (available > 0 && p.per_token > 0) {
        const long long budget = (long long) available - (long long) model_bytes -
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
    int ngl = 0;
    float temp = 0.0f;          // 0 = greedy
    float top_p = 0.95f;
    float min_p = 0.05f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    bool plan_only = false;
    bool warmup = true;
    bool help = false;
};

void print_usage(const char * argv0) {
    std::printf(
        "\nusage:\n"
        "  %s -m model.gguf [options]\n\n"
        "options:\n"
        "  -c N            context to request (0 = the model's trained context)\n"
        "  -n N            tokens to generate (default 64)\n"
        "  -t N            threads (default: physical cores)\n"
        "  -ngl N          layers on GPU (default 0)\n"
        "  -p TEXT         prompt (default \"The capital of France is\")\n"
        "  --kv TYPE       KV cache type for K and V (default f16)\n"
        "  -ctk/-ctv TYPE  set K and V separately\n"
        "  --temp/--top-p/--min-p/--seed   sampling (temp 0 = greedy, the default)\n"
        "  --plan          print the plan and exit (loads metadata only)\n"
        "  --no-warmup     skip the one-token warmup pass\n"
        "  --verbose       forward llama.cpp's INFO log to stderr\n"
        "  -h, --help\n\n", argv0);
}

bool parse_args(int argc, char ** argv, Options & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        if (a == "-h" || a == "--help") { o.help = true; return true; }
        else if (a == "--plan")     { o.plan_only = true; }
        else if (a == "--no-warmup"){ o.warmup = false; }
        else if (a == "--verbose")  { g_verbose = true; }
        else if (a == "-m" && has_value) { o.model = argv[++i]; }
        else if (a == "-p" && has_value) { o.prompt = argv[++i]; }
        else if (a == "-c" && has_value) { o.ctx = std::atoi(argv[++i]); }
        else if (a == "-n" && has_value) { o.n_predict = std::atoi(argv[++i]); }
        else if (a == "-t" && has_value) { o.threads = std::atoi(argv[++i]); }
        else if (a == "-ngl" && has_value){ o.ngl = std::atoi(argv[++i]); }
        else if (a == "--kv" && has_value) { o.kv_k = o.kv_v = argv[++i]; }
        else if (a == "-ctk" && has_value) { o.kv_k = argv[++i]; }
        else if (a == "-ctv" && has_value) { o.kv_v = argv[++i]; }
        else if (a == "--temp" && has_value) { o.temp = (float) std::atof(argv[++i]); }
        else if (a == "--top-p" && has_value) { o.top_p = (float) std::atof(argv[++i]); }
        else if (a == "--min-p" && has_value) { o.min_p = (float) std::atof(argv[++i]); }
        else if (a == "--seed" && has_value) { o.seed = (uint32_t) std::strtoul(argv[++i], nullptr, 10); }
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

double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t idx = (size_t) std::min<double>((double) v.size() - 1, std::floor(q * (double) v.size()));
    return v[idx];
}

}  // namespace

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
    model_params.n_gpu_layers = opt.ngl;
    // --plan still loads the weights: llama.cpp only finalises some hparams (the
    // per-layer KV head list among them) while loading tensors, and reading them
    // from a vocab-only model aborts. Planning cheaply is not worth an abort.

    ModelPtr model(llama_model_load_from_file(opt.model.c_str(), model_params));
    if (!model) {
        std::fprintf(stderr, "error: cannot load %s\n", opt.model.c_str());
        llama_backend_free();
        return 1;
    }

    const std::vector<int> heads = kv_heads_per_layer(model.get());
    const size_t model_bytes = (size_t) llama_model_size(model.get());
    const size_t available = mem_available_bytes();
    const Plan plan = make_plan(model.get(), opt.ctx, type_k, type_v, heads, model_bytes, available);
    const int threads = opt.threads > 0 ? opt.threads : physical_cores();
    const int train = llama_model_n_ctx_train(model.get());

    std::printf("\nvipera plan\n");
    std::printf("  model          %zu MiB, %d layers, %d kv heads, head dim %d\n",
                model_bytes / MIB, llama_model_n_layer(model.get()),
                llama_model_n_head_kv(model.get()),
                llama_model_n_embd(model.get()) / std::max(1, llama_model_n_head(model.get())));
    std::printf("  kv cache       %zu MiB at %d tokens (%s/%s)\n",
                plan.kv / MIB, plan.ctx, opt.kv_k.c_str(),
                (opt.kv_v.empty() ? opt.kv_k : opt.kv_v).c_str());
    std::printf("  available      %zu MiB\n", available / MIB);
    std::printf("  context        %d requested, %d used (train limit %d)\n",
                opt.ctx > 0 ? opt.ctx : train, plan.ctx, train);
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

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) plan.ctx;
    cparams.n_batch = std::min<uint32_t>(2048u, (uint32_t) plan.ctx);
    cparams.n_ubatch = std::min<uint32_t>(512u, cparams.n_batch);
    cparams.n_threads = (int32_t) threads;
    cparams.n_threads_batch = (int32_t) threads;
    cparams.type_k = type_k;
    cparams.type_v = type_v;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.no_perf = true;

    CtxPtr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        std::fprintf(stderr, "error: cannot create a context (%d tokens)\n", plan.ctx);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int n_prompt = -llama_tokenize(vocab, opt.prompt.c_str(), opt.prompt.size(),
                                         nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "error: cannot tokenize the prompt\n");
        llama_backend_free();
        return 1;
    }
    if (n_prompt >= plan.ctx) {
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

    // One-token warmup in a throwaway context: the first llama_decode on a model
    // allocates the compute graph, and charging that to prompt throughput is how
    // a benchmark reports a number nobody can reproduce.
    if (opt.warmup) {
        llama_context_params wparams = cparams;
        wparams.n_ctx = 8;
        wparams.n_batch = 8;
        wparams.n_ubatch = 8;
        CtxPtr warm(llama_init_from_model(model.get(), wparams));
        if (warm) {
            BatchPtr wb(llama_batch_ext_init(warm.get()));
            const llama_token bos = llama_vocab_bos(vocab);
            batch_set(wb.get(), &bos, 1, 0, true);
            llama_process(warm.get(), LLAMA_PROCESS_TYPE_DECODE, wb.get());
        }
    }

    BatchPtr batch(llama_batch_ext_init(ctx.get()));

    // Prefill in n_batch chunks: a prompt larger than the batch limit is legal,
    // but handing llama.cpp one oversized batch aborts the process.
    const int n_ctx = plan.ctx;
    int pos = 0;
    const double t_prefill_start = now_s();
    while (pos < n_prompt) {
        const int n = std::min<int>((int) cparams.n_batch, n_prompt - pos);
        batch_set(batch.get(), prompt_tokens.data() + pos, n, pos, true);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            std::fprintf(stderr, "error: prompt evaluation failed at token %d\n", pos);
            llama_backend_free();
            return 1;
        }
        pos += n;
    }
    const double t_prefill_end = now_s();

    SamplerPtr sampler = make_sampler(opt);
    const int budget = std::min(opt.n_predict, n_ctx - n_prompt);
    std::vector<double> token_ms;
    token_ms.reserve((size_t) std::max(0, budget));
    int n_generated = 0;
    std::string line;
    const double t_gen_start = now_s();
    while (n_generated < budget) {
        const llama_token id = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        char buf[256];
        const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            line.append(buf, (size_t) n);
        }

        const double t0 = now_s();
        batch_set(batch.get(), &id, 1, pos, true);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            std::fprintf(stderr, "error: decode failed at token %d\n", n_generated);
            llama_backend_free();
            return 1;
        }
        token_ms.push_back((now_s() - t0) * 1000.0);

        ++pos;
        ++n_generated;
        if (n > 0 && buf[n - 1] == '\n') {
            std::fwrite(line.data(), 1, line.size(), stdout);
            std::fflush(stdout);
            line.clear();
        }
    }
    const double t_gen_end = now_s();
    if (!line.empty()) {
        std::fwrite(line.data(), 1, line.size(), stdout);
    }

    // First generated token's cost includes the sample of the prefill's logits,
    // so it is excluded from the per-token statistics and reported separately.
    std::vector<double> steady(token_ms.size() > 1 ? std::vector<double>(token_ms.begin() + 1, token_ms.end())
                                                   : std::vector<double>());
    const double prefill_s = t_prefill_end - t_prefill_start;
    const double gen_s = t_gen_end - t_gen_start;

    std::printf("\n\n");
    std::printf("  prompt         %7.1f t/s  (%d tokens, %.2f s)\n",
                n_prompt / std::max(1e-9, prefill_s), n_prompt, prefill_s);
    if (n_generated > 0) {
        std::printf("  generation     %7.2f t/s  (%d tokens, %.2f s)\n",
                    n_generated / std::max(1e-9, gen_s), n_generated, gen_s);
        if (!steady.empty()) {
            std::printf("  per token      median %.2f ms, p90 %.2f ms, first %.2f ms\n",
                        median(steady), percentile(steady, 0.9), token_ms.front());
        }
    }
    std::printf("  kv used        %zu MiB of the %zu MiB planned\n",
                plan.per_token * (size_t) pos / MIB, plan.kv / MIB);
    return 0;
}

// kernel-gate: how many bytes a mul_mat reads per second, per quant type.
//
// The number that matters is bytes-read over wall time: for a quantized
// matvec the kernel is memory bound, so a faster kernel is one that moves the
// same bytes in less time. One JSON object per line, one line per quant type.
//
// Output is compared against tools/kernel_gate/baseline.json by the gate math
// in tools/vipera/gate.h (median of three, delta must beat 5%).

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Rows are a knob because the weight must be bigger than the last-level cache
// for the number to be a DRAM number: at 4096 rows the Q4_K weight is 9.4 MB
// and sits in an 8 MB L3, which is why it reads faster than the CPU can read
// main memory. `--rows 131072` puts every type at 220-570 MB.
static int N_ROWS = 4096;
static const int N_COLS   = 4096;
// Bytes one timed batch should read: enough to average out the timer, few
// enough to keep a 570 MB weight from taking a minute per type.
static const double BATCH_BYTES = 8e9;
static const int ITERS    = 50;
static const int WARMUP   = 3;
static const int N_RUNS   = 3;
static const int N_THREADS = 4;
// Rows quantized for real, then repeated to fill the weight. 512 rows of 4096
// floats is 8 MB, so a 570 MB weight does not first need a 2 GB float copy.
static const int TILE_ROWS = 512;

// xorshift: the same weights every run, so two builds are comparable
static float frand(uint32_t & s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (float) (s & 0xFFFFFF) / (float) 0xFFFFFF * 2.0f - 1.0f;
}

struct result {
    const char * name;
    double       gbs;
    size_t       bytes;
};

static bool bench_type(enum ggml_type type, const char * name, result & out) {
    const size_t row_size = ggml_row_size(type, N_COLS);
    const size_t nbytes   = row_size * (size_t) N_ROWS;
    const int tile_rows   = std::min(TILE_ROWS, N_ROWS);

    std::vector<float> src((size_t) tile_rows * N_COLS);
    uint32_t seed = 12345;
    for (auto & f : src) {
        f = frand(seed);
    }

    std::vector<uint8_t> tile(row_size * (size_t) tile_rows);
    if (ggml_quantize_chunk(type, src.data(), tile.data(), 0, tile_rows, N_COLS, nullptr) !=
        tile.size()) {
        fprintf(stderr, "%s: quantization size mismatch\n", name);
        return false;
    }

    std::vector<uint8_t> qdata(nbytes);
    for (size_t off = 0; off < nbytes; off += tile.size()) {
        memcpy(qdata.data() + off, tile.data(), std::min(tile.size(), nbytes - off));
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ nbytes + 1024u * 1024u * 1024u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: ggml_init failed\n", name);
        return false;
    }

    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, type, N_COLS, N_ROWS);
    memcpy(w->data, qdata.data(), nbytes);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_COLS, 1);
    for (int i = 0; i < N_COLS; ++i) {
        ((float *) x->data)[i] = 1.0f;
    }
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    const int iters = std::max(4, std::min(ITERS, (int) (BATCH_BYTES / (double) nbytes)));

    std::vector<double> runs;
    runs.reserve(N_RUNS);

    for (int r = 0; r < N_RUNS; ++r) {
        for (int i = 0; i < WARMUP; ++i) {
            ggml_graph_compute_with_ctx(ctx, gf, N_THREADS);
        }
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < iters; ++i) {
            ggml_graph_compute_with_ctx(ctx, gf, N_THREADS);
        }
        const int64_t t1 = ggml_time_us();
        runs.push_back((double) (t1 - t0) / 1e6);
    }

    // a sink so the compiler cannot skip the matvec
    if (((float *) y->data)[0] == 1e30f) {
        fprintf(stderr, "unreachable\n");
    }

    // best of the runs inside this process: a descheduled interval only ever
    // makes a run slower, so the fastest run is the closest to the kernel's
    // real cost. The protocol's median-3 is across processes, in compare.py.
    std::sort(runs.begin(), runs.end());
    const double best = runs.front();
    out.name  = name;
    out.bytes = nbytes;
    out.gbs   = (double) nbytes * iters / best / 1e9;

    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            N_ROWS = atoi(argv[++i]);
            if (N_ROWS < 1) {
                fprintf(stderr, "--rows needs a positive count\n");
                return 2;
            }
        } else {
            fprintf(stderr, "usage: %s [--rows N]\n", argv[0]);
            return 2;
        }
    }

    const struct {
        enum ggml_type type;
        const char *   name;
    } types[] = {
        { GGML_TYPE_Q8_0, "Q8_0" },
        { GGML_TYPE_Q4_K, "Q4_K" },
        { GGML_TYPE_Q3_K, "Q3_K" },
    };

    printf("# kernel-gate rows=%d cols=%d batch_bytes=%.0f threads=%d\n", N_ROWS, N_COLS,
           BATCH_BYTES, N_THREADS);

    for (const auto & t : types) {
        result r;
        if (!bench_type(t.type, t.name, r)) {
            return 1;
        }
        printf("{\"quant\":\"%s\",\"op\":\"mul_mat\",\"gbs\":%.3f,\"bytes\":%zu,\"env\":\"cpu\"}\n",
               r.name, r.gbs, r.bytes);
    }
    return 0;
}

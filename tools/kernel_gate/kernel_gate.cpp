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

static const int N_ROWS   = 4096;
static const int N_COLS   = 4096;
static const int ITERS    = 50;
static const int WARMUP   = 3;
static const int N_RUNS   = 3;
static const int N_THREADS = 4;

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
    const size_t nbytes   = row_size * N_ROWS;

    std::vector<float> src((size_t) N_ROWS * N_COLS);
    uint32_t seed = 12345;
    for (auto & f : src) {
        f = frand(seed);
    }

    std::vector<uint8_t> qdata(nbytes);
    if (ggml_quantize_chunk(type, src.data(), qdata.data(), 0, N_ROWS, N_COLS, nullptr) != nbytes) {
        fprintf(stderr, "%s: quantization size mismatch\n", name);
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024u * 1024u * 1024u,
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

    std::vector<double> runs;
    runs.reserve(N_RUNS);

    for (int r = 0; r < N_RUNS; ++r) {
        for (int i = 0; i < WARMUP; ++i) {
            ggml_graph_compute_with_ctx(ctx, gf, N_THREADS);
        }
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < ITERS; ++i) {
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
    out.gbs   = (double) nbytes * ITERS / best / 1e9;

    ggml_free(ctx);
    return true;
}

int main() {
    ggml_time_init();

    const struct {
        enum ggml_type type;
        const char *   name;
    } types[] = {
        { GGML_TYPE_Q8_0, "Q8_0" },
        { GGML_TYPE_Q4_K, "Q4_K" },
        { GGML_TYPE_Q3_K, "Q3_K" },
    };

    printf("# kernel-gate rows=%d cols=%d iters=%d threads=%d\n", N_ROWS, N_COLS, ITERS, N_THREADS);

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

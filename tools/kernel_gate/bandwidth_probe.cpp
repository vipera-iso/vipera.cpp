// bandwidth-probe (CPU): the number kernel-gate has to be compared against.
//
// A quantized matvec is memory bound, so "65 GB/s" means nothing until the
// machine's own achievable bandwidth is known. This measures the two shapes
// that matter:
//
//   read  -- sum every byte once, the closest CPU analogue of a decode step
//            streaming weights
//   copy  -- read one buffer and write another, the number people quote as
//            "memory bandwidth"
//
// One JSON object per line on stdout, same style as kernel-gate, so a run can
// be committed next to the kernel numbers it judges.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static const size_t BUFFER_BYTES = (size_t) 1 << 30;  // 1 GiB
static const int    N_RUNS       = 5;
static const int    WARMUP       = 1;

static double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static int default_threads() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? (int) hw : 4;
}

// The sink is written through a volatile pointer so the summation cannot be
// dropped, and is never read: that keeps the loop free of observable output
// while still being real work.
static volatile uint64_t g_sink = 0;

static void read_slice(const uint64_t * src, size_t count, uint64_t * out) {
    uint64_t acc = 0;
    for (size_t i = 0; i < count; ++i) {
        acc += src[i];
    }
    *out = acc;
}

static void copy_slice(uint8_t * dst, const uint8_t * src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

static double run_read(const std::vector<uint8_t> & buf, int threads) {
    // byte buffer read as words: vector<uint8_t> is aligned for any scalar, and
    // the probe wants the read and copy passes over the same allocation.
    const uint64_t *         words = reinterpret_cast<const uint64_t *>(buf.data());
    const size_t             count = buf.size() / sizeof(uint64_t);
    const size_t             per   = (count + (size_t) threads - 1) / (size_t) threads;
    std::vector<uint64_t>    partial((size_t) threads, 0);
    std::vector<std::thread> pool;

    const double t0 = now_s();
    pool.reserve((size_t) threads);
    for (int t = 0; t < threads; ++t) {
        const size_t begin = (size_t) t * per;
        const size_t end   = begin + per < count ? begin + per : count;
        if (begin >= end) {
            continue;
        }
        pool.emplace_back(read_slice, words + begin, end - begin, &partial[(size_t) t]);
    }
    for (auto & th : pool) {
        th.join();
    }
    const double elapsed = now_s() - t0;

    uint64_t total = 0;
    for (uint64_t v : partial) {
        total += v;
    }
    g_sink = total;
    return elapsed;
}

static double run_copy(std::vector<uint8_t> & dst, const std::vector<uint8_t> & src, int threads) {
    const size_t             count = src.size();
    const size_t             per   = (count + (size_t) threads - 1) / (size_t) threads;
    std::vector<std::thread> pool;

    const double t0 = now_s();
    pool.reserve((size_t) threads);
    for (int t = 0; t < threads; ++t) {
        const size_t begin = (size_t) t * per;
        const size_t end   = begin + per < count ? begin + per : count;
        if (begin >= end) {
            continue;
        }
        pool.emplace_back(copy_slice, dst.data() + begin, src.data() + begin, end - begin);
    }
    for (auto & th : pool) {
        th.join();
    }
    return now_s() - t0;
}

static double best_of(const std::vector<double> & runs) {
    double best = runs.front();
    for (double v : runs) {
        if (v < best) {
            best = v;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    int threads = default_threads();
    if (argc > 1) {
        const int parsed = std::atoi(argv[1]);
        if (parsed > 0) {
            threads = parsed;
        }
    }

    std::printf("# bandwidth-probe buffer=%.2f GiB runs=%d threads=%d\n", (double) BUFFER_BYTES / (double) (1u << 30),
                N_RUNS, threads);

    std::vector<uint8_t> src(BUFFER_BYTES);
    std::vector<uint8_t> dst(BUFFER_BYTES);
    // Deterministic fill: a bandwidth number that depends on the data pattern
    // is not a bandwidth number.
    uint64_t *           words = reinterpret_cast<uint64_t *>(src.data());
    uint64_t             seed  = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < BUFFER_BYTES / sizeof(uint64_t); ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        words[i] = seed;
    }

    std::vector<double> reads;
    for (int r = -WARMUP; r < N_RUNS; ++r) {
        const double elapsed = run_read(src, threads);
        if (r >= 0) {
            reads.push_back(elapsed);
        }
    }
    const double best_read = best_of(reads);
    std::printf("{\"probe\":\"cpu-read\",\"threads\":%d,\"bytes\":%zu,\"gbs\":%.3f}\n", threads, BUFFER_BYTES,
                (double) BUFFER_BYTES / best_read / 1e9);

    std::vector<double> copies;
    for (int r = -WARMUP; r < N_RUNS; ++r) {
        const double elapsed = run_copy(dst, src, threads);
        if (r >= 0) {
            copies.push_back(elapsed);
        }
    }
    const double best_copy = best_of(copies);
    // A copy touches every byte twice; report the moved total so the number is
    // comparable with a card's "GB/s" figure.
    const double moved     = (double) BUFFER_BYTES * 2.0;
    std::printf("{\"probe\":\"cpu-copy\",\"threads\":%d,\"bytes\":%zu,\"gbs\":%.3f}\n", threads, (size_t) moved,
                moved / best_copy / 1e9);

    std::printf("# cpu-cores=%u\n", (unsigned int) std::thread::hardware_concurrency());
    return 0;
}

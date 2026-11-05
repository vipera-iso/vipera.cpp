// bandwidth-probe (CUDA): the ceiling a GPU decode kernel is judged against.
//
// Decode streams every weight once per token, so its speed is the card's
// achievable read bandwidth divided by the model bytes. The spec-sheet number
// (192 GB/s for a GTX 1650) is not that ceiling: no kernel reaches it. This
// measures what a plain read and a plain copy actually achieve.
//
// JSON on stdout, in the same shape as the CPU probe, so both sit in one file.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const size_t BUFFER_BYTES = (size_t) 1 << 30;  // 1 GiB
static const int    N_RUNS       = 10;

#define CHECK_CUDA(call)                                                               \
    do {                                                                               \
        const cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err__)); \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

__global__ void read_kernel(const uint4 * __restrict__ src, size_t n, unsigned long long * __restrict__ sink) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    uint4        acc    = make_uint4(0, 0, 0, 0);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint4 v = src[i];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    // Never true for the data this probe uses, so the writes stay out of the
    // timing while the loads cannot be optimised away.
    if ((acc.x | acc.y | acc.z | acc.w) == 0xFFFFFFFFu) {
        sink[0] = (unsigned long long) acc.x;
    }
}

__global__ void copy_kernel(uint4 * __restrict__ dst, const uint4 * __restrict__ src, size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

static int blocks_for(size_t n) {
    const int per_block = 256;
    const int wanted    = (int) ((n + (size_t) per_block - 1) / (size_t) per_block);
    return std::min(std::max(wanted, 1024), 65535);
}

static float time_ms(void (*launch)(const uint4 *, uint4 *, size_t, unsigned long long *, int),
                     const uint4 *        src,
                     uint4 *              dst,
                     size_t               n,
                     unsigned long long * sink,
                     int                  reps) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int r = 0; r < reps; ++r) {
        launch(src, dst, n, sink, r);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static void launch_read(const uint4 * src, uint4 *, size_t n, unsigned long long * sink, int) {
    read_kernel<<<blocks_for(n), 256>>>(src, n, sink);
}

static void launch_copy(const uint4 * src, uint4 * dst, size_t n, unsigned long long *, int) {
    copy_kernel<<<blocks_for(n), 256>>>(dst, src, n);
}

int main() {
    int            device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    uint4 *              d_src  = nullptr;
    uint4 *              d_dst  = nullptr;
    unsigned long long * d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, BUFFER_BYTES));
    CHECK_CUDA(cudaMalloc(&d_dst, BUFFER_BYTES));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_src, 0x5A, BUFFER_BYTES));

    const size_t n_vec = BUFFER_BYTES / sizeof(uint4);

    std::vector<float> reads;
    for (int r = -1; r < N_RUNS; ++r) {
        const float ms = time_ms(launch_read, d_src, d_dst, n_vec, d_sink, 5);
        if (r >= 0) {
            reads.push_back(ms / 5.0f);
        }
    }
    const float best_read = *std::min_element(reads.begin(), reads.end());
    std::printf("{\"probe\":\"gpu-read\",\"device\":\"%s\",\"bytes\":%zu,\"gbs\":%.3f}\n", prop.name, BUFFER_BYTES,
                (double) BUFFER_BYTES / (double) best_read / 1e6);

    std::vector<float> copies;
    for (int r = -1; r < N_RUNS; ++r) {
        const float ms = time_ms(launch_copy, d_src, d_dst, n_vec, d_sink, 5);
        if (r >= 0) {
            copies.push_back(ms / 5.0f);
        }
    }
    const float  best_copy = *std::min_element(copies.begin(), copies.end());
    const double moved     = (double) BUFFER_BYTES * 2.0;
    std::printf("{\"probe\":\"gpu-copy\",\"device\":\"%s\",\"bytes\":%.0f,\"gbs\":%.3f}\n", prop.name, moved,
                moved / (double) best_copy / 1e6);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_sink);
    return 0;
}

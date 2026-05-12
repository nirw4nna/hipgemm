#include "fp8_4wave.hpp"
#include "utils.hpp"

#ifndef MDIM
#   define MDIM 8192
#endif

#ifndef NDIM
#   define NDIM 8192
#endif

#ifndef KDIM
#   define KDIM 8192
#endif

int main() {
    int warmup_iters = 1000;
    int iters = 1000;

    constexpr int M = MDIM;
    constexpr int N = NDIM;
    constexpr int K = KDIM;

    int ROTATING_BUFFER_COUNT = std::max((512 * 1024 * 1024) / (2 * M * K), (512 * 1024 * 1024) / (2 * N * K)); // Rotating Buffers for A,B

    size_t A_size = M * K, B_size = K * N, C_size = M * N;
    size_t A_size_bytes = A_size * sizeof(fp8);
    size_t B_size_bytes = B_size * sizeof(fp8);
    size_t C_size_bytes = C_size * sizeof(bf16);

    fp8* A_host = (fp8*)hip_malloc_host(A_size_bytes);
    fp8* B_host = (fp8*)hip_malloc_host(B_size_bytes);

    fp8* A_device = (fp8*)hip_malloc_device(ROTATING_BUFFER_COUNT * A_size_bytes);
    fp8* B_device = (fp8*)hip_malloc_device(ROTATING_BUFFER_COUNT * B_size_bytes);
    bf16* C_device = (bf16*)hip_malloc_device(C_size_bytes);

    printf("Initializing %i rotating buffer blocks\n", ROTATING_BUFFER_COUNT);
    for (int i = 0; i < ROTATING_BUFFER_COUNT; i++) {
        init_random(A_host, A_size, 2 * i);
        init_random(B_host, B_size, 2 * i + 1);
        CHECK_HIP_ERROR(hipMemcpy(A_device + A_size * i, A_host, A_size_bytes, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(B_device + B_size * i, B_host, B_size_bytes, hipMemcpyHostToDevice));
        printf("Buffer block #%i created\n", i);
    }
    printf("Buffer initialization finished\n");

    printf("\n%i warmup iterations...\n", warmup_iters);
    for (int i = 0; i < warmup_iters; i++) {
        int block_idx = i % ROTATING_BUFFER_COUNT;
        fp8* A_current_device = A_device + block_idx * A_size;
        fp8* B_current_device = B_device + block_idx * B_size;
        CHECK_HIP_ERROR(hipMemset(C_device, 0, C_size_bytes));

        fp8_gemm_4wave_256x256x128<M, N, K><<<(M * N) / (256 * 256), 256>>>(A_current_device, B_current_device, C_device);

        CHECK_HIP_ERROR(hipDeviceSynchronize());
        CHECK_HIP_ERROR(hipGetLastError());
    }
    printf("Warmup finished\n");

    hipEvent_t start_event, stop_event;
    CHECK_HIP_ERROR(hipEventCreate(&start_event));
    CHECK_HIP_ERROR(hipEventCreate(&stop_event));

    std::vector<float> times_ms;
    times_ms.reserve(iters);
    float ms = 0.0f;

    printf("\n%i benchmark iterations...\n", iters);
    for (int i = 0; i < iters; i++) {
        int block_idx = (i + warmup_iters) % ROTATING_BUFFER_COUNT;
        fp8* A_current_device = A_device + block_idx * A_size;
        fp8* B_current_device = B_device + block_idx * B_size;
        CHECK_HIP_ERROR(hipMemset(C_device, 0, C_size_bytes));

        CHECK_HIP_ERROR(hipEventRecord(start_event, 0));
        fp8_gemm_4wave_256x256x128<M, N, K><<<(M * N) / (256 * 256), 256>>>(A_current_device, B_current_device, C_device);

        CHECK_HIP_ERROR(hipEventRecord(stop_event, 0));
        CHECK_HIP_ERROR(hipEventSynchronize(stop_event));
        CHECK_HIP_ERROR(hipEventElapsedTime(&ms, start_event, stop_event));

        times_ms.push_back(ms);
        CHECK_HIP_ERROR(hipGetLastError());
    }
    printf("Benchmark finished\n");

    float sum_ms = 0.f, best_ms = 1e30f;
    for (float t : times_ms) {
        sum_ms += t;
        best_ms = std::min(best_ms, t);
    }
    float avg_ms = sum_ms / times_ms.size();
    double flop = 2.0 * M * N * K;
    double best_tflops = (flop / (best_ms * 1e-3)) / 1e12;
    double avg_tflops = (flop / (avg_ms * 1e-3)) / 1e12;


    CHECK_HIP_ERROR(hipEventDestroy(start_event));
    CHECK_HIP_ERROR(hipEventDestroy(stop_event));
    CHECK_HIP_ERROR(hipGetLastError());

    CHECK_HIP_ERROR(hipFreeHost(A_host));
    CHECK_HIP_ERROR(hipFreeHost(B_host));
    CHECK_HIP_ERROR(hipFree(A_device));
    CHECK_HIP_ERROR(hipFree(B_device));
    CHECK_HIP_ERROR(hipFree(C_device));
    CHECK_HIP_ERROR(hipGetLastError());

    printf("\n=== PERFORMANCE RESULTS ===\n");
    printf("  Kernel time (best): %.5f ms,  TFLOPS: %.2f\n", best_ms, best_tflops);
    printf("  Kernel time (avg ): %.5f ms,  TFLOPS: %.2f\n", avg_ms, avg_tflops);

    return 0;
}
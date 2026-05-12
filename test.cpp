#include "fp8_4wave.hpp"
#include "utils.hpp"
#include <omp.h>


#ifndef MDIM
#   define MDIM 4096
#endif

#ifndef NDIM
#   define NDIM 4096
#endif

#ifndef KDIM
#   define KDIM 4096
#endif


void gemm_cpu(fp8* A, fp8* B, bf16* C, int M, int N, int K) {
#pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += float(A[row * K + k]) * float(B[col * K + k]); // col-major B
            }
            C[row * N + col] = bf16(acc);
        }
    }
}

bool validate_matrix(bf16* C, bf16* C_ref, int M, int N) {
    bool success = true;
    for (int row = 0; row < M && success; ++row) {
        for (int col = 0; col < N && success; ++col) {
            float c_val = float(C[row * N + col]);
            float c_ref_val = float(C_ref[row * N + col]);
            float diff = std::abs(c_val - c_ref_val);
            float threshold = c_ref_val * 0.01f;
            if (diff > threshold || std::isnan(c_val) || std::isnan(c_ref_val)) {
                printf("Mismatch at (row=%d, col=%d): c_host = %f, c_ref = %f, diff = %f\n", row, col, c_val, c_ref_val, diff);
                success = false;
            }
        }
    }
    return success;
}

int main() {
    constexpr int M = MDIM;
    constexpr int N = NDIM;
    constexpr int K = KDIM;

    size_t A_size = M * K, B_size = K * N, C_size = M * N;


    size_t A_size_bytes = A_size * sizeof(fp8);
    size_t B_size_bytes = B_size * sizeof(fp8);
    size_t C_size_bytes = C_size * sizeof(bf16);

    fp8* A_host = (fp8*)hip_malloc_host(A_size_bytes);
    fp8* B_host = (fp8*)hip_malloc_host(B_size_bytes);
    bf16* C_host = (bf16*)hip_malloc_host(C_size_bytes);
    bf16* C_ref = (bf16*)hip_malloc_host(C_size_bytes);

    fp8* A_device = (fp8*)hip_malloc_device(A_size_bytes);
    fp8* B_device = (fp8*)hip_malloc_device(B_size_bytes);
    bf16* C_device = (bf16*)hip_malloc_device(C_size_bytes);

    init_random(A_host, A_size);
    init_random(B_host, B_size);

    CHECK_HIP_ERROR(hipMemcpy(A_device, A_host, A_size_bytes, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_device, B_host, B_size_bytes, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(C_device, C_host, C_size_bytes, hipMemcpyHostToDevice));

    gemm_cpu(A_host, B_host, C_ref, M, N, K);

    fp8_gemm_4wave_256x256x128<M, N, K><<<(M * N) / (256 * 256), 256>>>(A_device, B_device, C_device);

    CHECK_HIP_ERROR(hipMemcpy(C_host, C_device, C_size_bytes, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipGetLastError());
    bool success = validate_matrix(C_host, C_ref, M, N);
    printf("Test passed: %i\n", success);

    CHECK_HIP_ERROR(hipFreeHost(A_host));
    CHECK_HIP_ERROR(hipFreeHost(B_host));
    CHECK_HIP_ERROR(hipFreeHost(C_host));
    CHECK_HIP_ERROR(hipFreeHost(C_ref));
    CHECK_HIP_ERROR(hipFree(A_device));
    CHECK_HIP_ERROR(hipFree(B_device));
    CHECK_HIP_ERROR(hipFree(C_device));
    CHECK_HIP_ERROR(hipGetLastError());

    return 0;
}
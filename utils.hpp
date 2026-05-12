#pragma once

#include <cstdio>
#include <hip/hip_runtime.h>
#include <random>

#define CHECK_HIP_ERROR(error)                                                                                  \
    if (error != hipSuccess) {                                                                                  \
        fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                                                     \
    }

void*
hip_malloc_host(size_t size) {
    void* ptr;
    CHECK_HIP_ERROR(hipHostMalloc(&ptr, size));
    return ptr;
}

void*
hip_malloc_device(size_t size) {
    void* ptr;
    CHECK_HIP_ERROR(hipMalloc(&ptr, size));
    return ptr;
}

void
init_random(fp8* array, size_t size, size_t seed = 0) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        array[i] = fp8(dis(gen));
    }
}
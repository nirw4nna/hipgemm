#pragma once

#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <bit>

using fp8 = __hip_fp8_e4m3;
using fp8x4 = __hip_fp8x4_e4m3;
using bf16 = __hip_bfloat16;
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void
llvm_amdgcn_raw_buffer_load_lds(i32x4 rsrc, as3_uint32_ptr lds_ptr, int size, int voffset, int soffset, int offset, int aux)
                                __asm("llvm.amdgcn.raw.buffer.load.lds");

struct buffer_resource {
    const void* ptr;
    int32_t range;
    int32_t config;
};

__device__ i32x4
make_buffer_resource(const void* ptr, int32_t range) {
    buffer_resource res{ptr, range, 0x110000};
    return __builtin_bit_cast(i32x4, res);
}

template <int ld>
__device__ inline void
global_to_lds(const fp8* global_ptr, fp8* lds_ptr, const int& waveid, uint32_t (&swizzled_offsets)[2]) {
    i32x4 srsrc = make_buffer_resource(global_ptr, 128 * ld);
    lds_ptr = lds_ptr + waveid * 1024;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        lds_ptr = lds_ptr + (i * 8192);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);
        llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, 16, swizzled_offsets[i], 0, 0, 0);
    }
}

struct coord {
    int row;
    int col;
};

__device__ inline coord
swizzle_lds_coord(int row, int col) {
    const uint32_t offset = (row * 128 + col);
    const int swizzle = ((offset % (16 * 128)) >> 8) << 4;
    const int swizzled_offset = offset ^ swizzle;
    int swizzled_row = swizzled_offset / 128;
    int swizzled_col = swizzled_offset % 128;
    return {swizzled_row, swizzled_col};
}

template <int ld>
__device__ inline void
calculate_swizzled_offsets(uint32_t (&swizzled_offsets)[2], const int& waveid, const int& laneid) {
    int wave_offset = (waveid / 2) * 16 * ld;
    int row = ((waveid % 2) * 64 + laneid) / 8;
    int col = (laneid % 8) * 16;
    coord dst_coord = swizzle_lds_coord(row, col);
    int swizzled_row = dst_coord.row;
    int swizzled_col = dst_coord.col;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        swizzled_offsets[i] = wave_offset + (i * 64 + swizzled_row) * ld + swizzled_col;
    }
}

__device__ inline static void
lds_to_a_reg(fp8* lds_ptr, fp8x4 (&a_reg)[4][8], const int& laneid) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
        int row = laneid % 16;
        int col = (laneid / 16) * 16 + k * 64;
        const uint32_t lds_uint32_ptr = reinterpret_cast<uintptr_t>(lds_ptr);
        const uint32_t offset = lds_uint32_ptr + row * 128 + col;
        const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
        const int idx = k * 4;

#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int offset = i * 2048;
            asm volatile("ds_read_b128 %0, %1 offset:%2\n" : "=v"(*reinterpret_cast<float4*>(&a_reg[i][idx])) : "v"(addr), "i"(offset) : "memory");
        }
    }
}

__device__ inline static void
lds_to_b_reg(fp8* lds_ptr, fp8x4 (&b_reg)[2][8], const int& laneid) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
        int row = laneid % 16;
        int col = (laneid / 16) * 16 + k * 64;
        const uint32_t lds_uint32_ptr = reinterpret_cast<uintptr_t>(lds_ptr);
        const uint32_t offset = lds_uint32_ptr + row * 128 + col;
        const uint32_t addr = offset ^ (((offset % (16 * 128)) >> 8) << 4);
        const int idx = k * 4;

#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int offset = i * 2048;
            asm volatile("ds_read_b128 %0, %1 offset:%2\n" : "=v"(*reinterpret_cast<float4*>(&b_reg[i][idx])) : "v"(addr), "i"(offset) : "memory");
        }
    }
}

__device__ static inline void
mfma_16x16x128_f8f8(const fp8x4 (&a_reg)[8], const fp8x4 (&b_reg)[8], float2 (&c_reg)[2]) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    *(floatx4_t*)c_reg = {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(*(intx8_t*)a_reg, *(intx8_t*)b_reg, *(floatx4_t*)c_reg, 0, 0, 0, 0, 0, 0)};
}

__device__ static inline void
mfma(const fp8x4 (&a_reg)[4][8], const fp8x4 (&b_reg)[2][8], float2 (&c_reg)[4][2][2]) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            mfma_16x16x128_f8f8(a_reg[i], b_reg[j], c_reg[i][j]);
        }
    }
}

static __host__ __device__ inline bf16
float_to_bf16_custom(const float& val) {
    return std::bit_cast<bf16>(static_cast<uint16_t>(std::bit_cast<uint32_t>(val) >> 16));
}

template <int ld>
__device__ inline void
store_accumulator(float2 (&c_reg)[4][2][2], bf16* global_ptr, const int& laneid) {
    const int row_offset = 4 * (laneid / 16);
    const int col_offset = laneid % 16;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = i * 16 + row_offset;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int col = j * 16 + col_offset;
#pragma unroll
            for (int k = 0; k < 2; k++) {
                global_ptr[(row + k * 2) * ld + col] = float_to_bf16_custom(c_reg[i][j][k].x);
                global_ptr[(row + k * 2 + 1) * ld + col] = float_to_bf16_custom(c_reg[i][j][k].y);
            }
        }
    }
}

template <int M, int N, int K>
__global__ void
__launch_bounds__(512, 2) fp8_gemm_256x256x128(fp8* A, fp8* B, bf16* C) {
    __shared__ fp8 A_lds[2][2][128 * 128];
    __shared__ fp8 B_lds[2][2][128 * 128];

    fp8x4 a_reg[4][8];
    fp8x4 b_reg0[2][8];
    fp8x4 b_reg1[2][8];

    float2 c_reg0[4][2][2]{};
    float2 c_reg1[4][2][2]{};
    float2 c_reg2[4][2][2]{};
    float2 c_reg3[4][2][2]{};

    const int waveid = threadIdx.x / 64;
    const int laneid = threadIdx.x % 64;

    uint32_t swizzled_offsets[2];
    calculate_swizzled_offsets<K>(swizzled_offsets, waveid, laneid);

    constexpr int num_blocks_k = K / 128;
    constexpr int num_blocks_n = N / 256;

    const int block_m = blockIdx.x / num_blocks_n;
    const int block_n = blockIdx.x % num_blocks_n;

    const int wave_m = waveid / 4;
    const int wave_n = waveid % 4;

    int db_idx0 = 0, db_idx1 = 1;

    fp8* global_load_B0 = B + (2 * block_n * 128 * K);
    fp8* global_load_A0 = A + (2 * block_m * 128 * K);
    fp8* global_load_B1 = B + ((2 * block_n + 1) * 128 * K);
    fp8* global_load_A1 = A + ((2 * block_m + 1) * 128 * K);

    global_to_lds<K>(global_load_B0, B_lds[db_idx0][0], waveid, swizzled_offsets);
    global_to_lds<K>(global_load_A0, A_lds[db_idx0][0], waveid, swizzled_offsets);
    global_to_lds<K>(global_load_B1, B_lds[db_idx0][1], waveid, swizzled_offsets);
    global_to_lds<K>(global_load_A1, A_lds[db_idx0][1], waveid, swizzled_offsets);

    if (wave_m == 1) { __builtin_amdgcn_s_barrier(); }

    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    global_to_lds<K>(global_load_B0 + 128, B_lds[db_idx1][0], waveid, swizzled_offsets);
    global_to_lds<K>(global_load_A0 + 128, A_lds[db_idx1][0], waveid, swizzled_offsets);
    global_to_lds<K>(global_load_B1 + 128, B_lds[db_idx1][1], waveid, swizzled_offsets);

    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

#pragma unroll 2
    for (int k = 0; k < (num_blocks_k - 2); k++) {
        fp8* B_lds_wave0 = B_lds[db_idx0][0] + wave_n * 32 * 128;
        lds_to_b_reg(B_lds_wave0, b_reg0, laneid);
        fp8* A_lds_wave0 = A_lds[db_idx0][0] + wave_m * 64 * 128;
        lds_to_a_reg(A_lds_wave0, a_reg, laneid);
        global_to_lds<K>(global_load_A1 + (k + 1) * 128, A_lds[db_idx1][1], waveid, swizzled_offsets);
        // asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg0, c_reg0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        fp8* B_lds_wave1 = B_lds[db_idx0][1] + wave_n * 32 * 128;
        lds_to_b_reg(B_lds_wave1, b_reg1, laneid);
        global_to_lds<K>(global_load_B0 + (k + 2) * 128, B_lds[db_idx0][0], waveid, swizzled_offsets);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg1, c_reg1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fp8* A_lds_wave1 = A_lds[db_idx0][1] + wave_m * 64 * 128;
        lds_to_a_reg(A_lds_wave1, a_reg, laneid);
        global_to_lds<K>(global_load_A0 + (k + 2) * 128, A_lds[db_idx0][0], waveid, swizzled_offsets);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg0, c_reg2);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        global_to_lds<K>(global_load_B1 + (k + 2) * 128, B_lds[db_idx0][1], waveid, swizzled_offsets);
        asm volatile("s_waitcnt vmcnt(6)");
        __builtin_amdgcn_s_barrier();

        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg1, c_reg3);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        db_idx0 ^= 1;
        db_idx1 ^= 1;
    }

    {
        constexpr int k = num_blocks_k - 2;
        fp8* B_lds_wave0 = B_lds[db_idx0][0] + wave_n * 32 * 128;
        lds_to_b_reg(B_lds_wave0, b_reg0, laneid);
        fp8* A_lds_wave0 = A_lds[db_idx0][0] + wave_m * 64 * 128;
        lds_to_a_reg(A_lds_wave0, a_reg, laneid);
        global_to_lds<K>(global_load_A1 + (k + 1) * 128, A_lds[db_idx1][1], waveid, swizzled_offsets);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg0, c_reg0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        fp8* B_lds_wave1 = B_lds[db_idx0][1] + wave_n * 32 * 128;
        lds_to_b_reg(B_lds_wave1, b_reg1, laneid);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg1, c_reg1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        fp8* A_lds_wave1 = A_lds[db_idx0][1] + wave_m * 64 * 128;
        lds_to_a_reg(A_lds_wave1, a_reg, laneid);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg0, c_reg2);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        B_lds_wave0 = B_lds[db_idx1][0] + wave_n * 32 * 128;
        lds_to_b_reg(B_lds_wave0, b_reg0, laneid);
        __builtin_amdgcn_s_barrier();

        // asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg1, c_reg3);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        db_idx0 ^= 1, db_idx1 ^= 1;
    }

    {
        fp8* A_lds_wave0 = A_lds[db_idx0][0] + wave_m * 64 * 128;
        lds_to_a_reg(A_lds_wave0, a_reg, laneid);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg0, c_reg0);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        bf16* global_store0 = C + (2 * block_m * 128 + wave_m * 64) * N + 2 * block_n * 128 + wave_n * 32;
        store_accumulator<N>(c_reg0, global_store0, laneid);

        fp8* B_lds_wave1 = B_lds[db_idx0][1] + wave_n * 32 * 128;
        lds_to_b_reg(B_lds_wave1, b_reg1, laneid);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg1, c_reg1);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        bf16* global_store1 = C + (2 * block_m * 128 + wave_m * 64) * N + (2 * block_n + 1) * 128 + wave_n * 32;
        store_accumulator<N>(c_reg1, global_store1, laneid);

        fp8* A_lds_wave1 = A_lds[db_idx0][1] + wave_m * 64 * 128;
        lds_to_a_reg(A_lds_wave1, a_reg, laneid);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mfma(a_reg, b_reg0, c_reg2);
        mfma(a_reg, b_reg1, c_reg3);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    if (wave_m == 0) { __builtin_amdgcn_s_barrier(); }

    bf16* global_store2 = C + ((2 * block_m + 1) * 128 + wave_m * 64) * N + 2 * block_n * 128 + wave_n * 32;
    store_accumulator<N>(c_reg2, global_store2, laneid);
    bf16* global_store3 = C + ((2 * block_m + 1) * 128 + wave_m * 64) * N + (2 * block_n + 1) * 128 + wave_n * 32;
    store_accumulator<N>(c_reg3, global_store3, laneid);
}
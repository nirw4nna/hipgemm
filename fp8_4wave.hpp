#pragma once

// 4-Wave interleave GEMM kernel for AMD GPUs (CDNA4).
// The code is derived from HipKittens's FP8_4Wave matmul_device.
// Source: https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L1

#include <bit>
#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

#define RESTRICT __restrict
#define INLINE __forceinline__

using fp8 = __hip_fp8_e4m3;
using bf16 = __hip_bfloat16;
using fp32x4 = float __attribute__((__vector_size__(4 * sizeof(float))));
using i32x8 = int __attribute__((__vector_size__(8 * sizeof(int)))); // 256-bit type
using i32x4 = int __attribute__((__vector_size__(4 * sizeof(int)))); // 128-bit type
using as3_uint32_ptr = uint32_t __attribute__((address_space(3))) *;
using LDS_Swizzle = int2[4];


union RT_Frag {
    i32x8 full;
    i32x4 half[2];
};

// Each A/B register tile is 64x128 (4 i32x8 fragments)
struct RT_ABt {
    RT_Frag tiles[4];
};

// Each C register tile is 64x64
struct RT_C {
    fp32x4 tiles[4][4];
};

struct Coordinates2D {
    int row, col;
};

struct PrecomputedAddresses {
    i32x4 buffer_descriptor;
    uintptr_t lds_base;
};


// ====================================================================
// ======================== MFMA Helpers
// ====================================================================

static __device__ INLINE void mfma_16x16x128(fp32x4 &d, i32x8 &a, i32x8 &b, fp32x4 &c) {
    // V_MFMA_F32_16x16x128_F8F6F4 with FP8 inputs
    // M = 16, N = 16, K = 128
    // K_L = 128 / (64 / 16) = 32
    // Each lane holds 32 elements of A/B (8 VGPRs)
    d = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 0, 0, 0);
}

static __device__ INLINE void
mfma_ABt(RT_C &d, RT_ABt &a, RT_ABt &b, RT_C &c, const int m, const int n) {
    mfma_16x16x128(d.tiles[m][n], a.tiles[m].full, b.tiles[n].full, c.tiles[m][n]);
}

static __device__ INLINE void mfma_ABt_all(RT_C &d, RT_ABt &a, RT_ABt &b, RT_C &c) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            mfma_ABt(d, a, b, c, i, j);
        }
    }
}

// ====================================================================
// ======================== Global --> LDS Helpers
// ====================================================================

extern "C" __device__ void
llvm_amdgcn_raw_buffer_load_lds(i32x4 rsrc,
                                as3_uint32_ptr lds_ptr,
                                int size,
                                int voffset,
                                int soffset,
                                int offset,
                                int aux) __asm("llvm.amdgcn.raw.buffer.load."
                                               "lds");

static __device__ INLINE i32x4 make_buffer_resource(const void *RESTRICT ptr,
                                                    const int32_t range) {
    struct BufferResource {
        const void *ptr;
        int32_t range;
        int32_t config;
    };

    BufferResource res{ ptr, range, 0x110000 };
    return std::bit_cast<i32x4>(res);
}


template <int K>
static __device__ INLINE void
load_lds(fp8 *RESTRICT shmem, const fp8 *RESTRICT gmem_base, const int k_offset, const int4 &offsets) {
    const int wave_id = threadIdx.x / 64;
    auto lds_ptr = shmem + wave_id * 1024;

    auto buffer_dsc = make_buffer_resource(gmem_base + k_offset, K * 128);
    for (int step = 0; step < 4; ++step) {
        auto lds_ptr_raw = (as3_uint32_ptr)((uintptr_t)lds_ptr + step * 4096);
        llvm_amdgcn_raw_buffer_load_lds(buffer_dsc, lds_ptr_raw, 16,
                                        offsets[step], 0, 0, 0);
    }
}

template <int step>
static __device__ INLINE void load_one_lds(PrecomputedAddresses &addr, const int4 &offsets) {
    static_assert(step < 4);

    auto lds_ptr_raw = (as3_uint32_ptr)(addr.lds_base + step * 4096);
    llvm_amdgcn_raw_buffer_load_lds(addr.buffer_descriptor, lds_ptr_raw, 16,
                                    offsets[step], 0, 0, 0);
}

template <int K>
static __device__ INLINE PrecomputedAddresses
precompute_addresses(fp8 *RESTRICT lds_dst, const fp8 *RESTRICT gl_src, const int k_offset) {
    const int wave_id = threadIdx.x / 64;
    auto lds_ptr = (uintptr_t)(lds_dst + wave_id * 1024);

    auto buffer_dsc = make_buffer_resource(gl_src + k_offset, K * 128);
    return { buffer_dsc, lds_ptr };
}

static __device__ INLINE Coordinates2D swizzle_128(const int row, const int col) {
    const int offset = row * 128 + col;
    const int swizzle = ((offset % (16 * 128)) >> 8) << 4;
    const int swizzled_offset = offset ^ swizzle;

    return { swizzled_offset / 128, swizzled_offset % 128 };
}

template <int K> static __device__ INLINE int4 compute_global_swizzle() {
    const int wave_id = threadIdx.x / 64;
    const int lane_id = threadIdx.x % 64;
    // For this kernel assume always 4 rounds
    int4 offsets;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = lane_id / 8 + wave_id * 8 + i * 32;
        const int col = (lane_id % 8) * 16;
        const auto swizzled = swizzle_128(row, col);
        offsets[i] = swizzled.row * K + swizzled.col;
    }
    return offsets;
}

// ====================================================================
// ======================== LDS --> Registers Helpers
// ====================================================================

static __device__ INLINE void
precompute_swizzle_lds(LDS_Swizzle &lds_swizzle, const int wave_idx) {
    const int lane_id = threadIdx.x % 64;

#pragma unroll
    for (int row_offset = 0; row_offset < 4; ++row_offset) {
        const int row = wave_idx * 64 + row_offset * 16 + lane_id % 16;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int col = (lane_id / 16) * 16 + i * 64;
            const auto swizzle = swizzle_128(row, col);
            lds_swizzle[row_offset][i] = swizzle.row * 128 + swizzle.col;
        }
    }
}

// Load half of a 16x128 tile from LDS to registers
template <int register_row, int k>
static __device__ INLINE void
load_one_rt(const fp8 *RESTRICT lds_src, RT_ABt &dst, const LDS_Swizzle &lds_swizzle) {
    static_assert(k < 2);
    static_assert(register_row < 4);

    const auto lds_addr =
    (uint32_t)(((uintptr_t)lds_src) + lds_swizzle[register_row][k]);
    asm volatile("ds_read_b128 %0, %1\n"
                 : "=v"(dst.tiles[register_row].half[k])
                 : "v"(lds_addr)
                 : "memory");
}

// Load an entire 64x128 tile from LDS to registers
static __device__ INLINE void load_rt(const fp8 *lds_src, RT_ABt &dst, const int wave_idx) {
    const int lane_id = threadIdx.x % 64;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = wave_idx * 64 + i * 16 + lane_id % 16;
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int col = (lane_id / 16) * 16 + step * 64;
            const auto swizzle = swizzle_128(row, col);
            const auto lds_addr =
            (uint32_t)(((uintptr_t)lds_src) + (swizzle.row * 128 + swizzle.col));
            asm volatile("ds_read_b128 %0, %1\n"
                         : "=v"(dst.tiles[i].half[step])
                         : "v"(lds_addr)
                         : "memory");
        }
    }
}

static __device__ INLINE bf16 cvt_f32_bf16(const float x) {
    // This is marginally better than the __float2bfloat16 builtin
    return std::bit_cast<bf16>((uint16_t)(std::bit_cast<uint32_t>(x) >> 16));
}

template <int N>
static __device__ INLINE void
store_rt(bf16 *RESTRICT gl_dst, const RT_C &rt_src, const int base_row, const int base_col) {
    const int lane_id = threadIdx.x % 64;
    const int group = lane_id / 16;
    const int col_lane = lane_id % 16;

#pragma unroll
    for (int ti = 0; ti < 4; ++ti) {
        const int row = base_row + ti * 16 + group * 4;
#pragma unroll
        for (int tj = 0; tj < 4; ++tj) {
            const int col = base_col + tj * 16 + col_lane;
            gl_dst[(row + 0) * N + col] = cvt_f32_bf16(rt_src.tiles[ti][tj][0]);
            gl_dst[(row + 1) * N + col] = cvt_f32_bf16(rt_src.tiles[ti][tj][1]);
            gl_dst[(row + 2) * N + col] = cvt_f32_bf16(rt_src.tiles[ti][tj][2]);
            gl_dst[(row + 3) * N + col] = cvt_f32_bf16(rt_src.tiles[ti][tj][3]);
        }
    }
}

// ====================================================================
// ======================== Kernel Code
// ====================================================================

template <int K>
static __device__ INLINE void interleaved_cluster(fp8 *RESTRICT lds_dst,
                                                  const fp8 *RESTRICT gl_src,
                                                  const int k_offset,
                                                  const int4 &global_offsets,
                                                  const int wave_idx,
                                                  RT_ABt &rt_dst,
                                                  const fp8 *RESTRICT lds_src,
                                                  RT_ABt &a,
                                                  RT_ABt &b,
                                                  RT_C &c) {
    // This function computes a 64x64 output tile using 4x4 16x128 MFMA
    // instructions The compute is interleaved with memory load operations (from
    // global to LDS and from LDS to registers) that will be used later. The
    // operands of the MFMA are a, b, c and when this function is called are
    // already in registers ready to be used.
    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 0, 0);
    __builtin_amdgcn_sched_barrier(0);

    auto addresses = precompute_addresses<K>(lds_dst, gl_src, k_offset);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 0, 1);
    __builtin_amdgcn_sched_barrier(0);

    // Note: this could be pre-computed once since there are just 2 possible
    // wave ids however, I tried this and it crushed the perfs.
    LDS_Swizzle lds_swizzle;
    precompute_swizzle_lds(lds_swizzle, wave_idx);

    load_one_lds<0>(addresses, global_offsets);
    load_one_rt<0, 0>(lds_src, rt_dst, lds_swizzle);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 0, 2);
    __builtin_amdgcn_sched_barrier(0);

    load_one_rt<0, 1>(lds_src, rt_dst, lds_swizzle);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 0, 3);
    __builtin_amdgcn_sched_barrier(0);

    load_one_lds<1>(addresses, global_offsets);
    load_one_rt<1, 0>(lds_src, rt_dst, lds_swizzle);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 1, 0);
    mfma_ABt(c, a, b, c, 1, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one_rt<1, 1>(lds_src, rt_dst, lds_swizzle);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 1, 2);
    mfma_ABt(c, a, b, c, 1, 3);
    __builtin_amdgcn_sched_barrier(0);

    load_one_lds<2>(addresses, global_offsets);
    load_one_rt<2, 0>(lds_src, rt_dst, lds_swizzle);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 2, 0);
    mfma_ABt(c, a, b, c, 2, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one_rt<2, 1>(lds_src, rt_dst, lds_swizzle);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 2, 2);
    mfma_ABt(c, a, b, c, 2, 3);
    __builtin_amdgcn_sched_barrier(0);

    load_one_lds<3>(addresses, global_offsets);
    load_one_rt<3, 0>(lds_src, rt_dst, lds_swizzle);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 3, 0);
    mfma_ABt(c, a, b, c, 3, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one_rt<3, 1>(lds_src, rt_dst, lds_swizzle);

    __builtin_amdgcn_sched_barrier(0);
    mfma_ABt(c, a, b, c, 3, 2);
    mfma_ABt(c, a, b, c, 3, 3);
    __builtin_amdgcn_sched_barrier(0);
}

template <int M, int N, int K>
__global__
__launch_bounds__(256, 1) void fp8_gemm_4wave_256x256x128(const fp8 *RESTRICT A,
                                                          const fp8 *RESTRICT B_T,
                                                          bf16 *RESTRICT C) {
    constexpr int BLOCK_M = 256;
    constexpr int BLOCK_N = 256;
    constexpr int BLOCK_K = 128;

    __shared__ fp8 A_lds[2][2][128 * BLOCK_K];
    __shared__ fp8 B_lds[2][2][128 * BLOCK_K];

    constexpr int k_step = BLOCK_K;
    constexpr int k_iters = K / BLOCK_K;

    const int wave_id = threadIdx.x / 64;

    // Note: uncomment these two lines to use normal block assignment
    // const int tile_i = blockIdx.x / (N / BLOCK_N);
    // const int tile_j = blockIdx.x % (N / BLOCK_N);
    const int wave_i = wave_id / 2; // 0..1
    const int wave_j = wave_id % 2; // 0..1
    
    // Note: L2 swizzle
    const int global_block_id = blockIdx.x;
    // Original WGID.
    int wgid = global_block_id;
    const int NUM_WGS = gridDim.x;
    constexpr int NUM_XCDS = 8;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    // Swizzle for better L2 within the same XCD.
    constexpr int WGM = 4;
    constexpr int num_pid_m = (M + BLOCK_M - 1) / BLOCK_M;
    constexpr int num_pid_n = (N + BLOCK_N - 1) / BLOCK_N;
    constexpr int num_wgid_in_group = WGM * num_pid_n;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * WGM;
    const int group_size_m = min(num_pid_m - first_pid_m, WGM);
    const int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    const int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    const int tile_i = pid_m; // blockIdx.x
    const int tile_j = pid_n; // blockIdx.y

    // Compute swizzling offsets once
    const int4 global_offset = compute_global_swizzle<K>();

    RT_C c[2][2]{};

    // Ping-pong schedule: compute on cur load on next
    int cur = 0, next = 1;
    {
        __builtin_amdgcn_sched_barrier(0);
        // In HK, RT_A/B hold an entire 64x128 tile so whenever a load LDS->reg is invoked the entire tile is loaded
        RT_ABt a[2];
        RT_ABt b[2];

        const fp8 *RESTRICT gl_A0_ptr = &A[(tile_i * BLOCK_M) * K];
        const fp8 *RESTRICT gl_A128_ptr = &A[(tile_i * BLOCK_M + 128) * K];
        const fp8 *RESTRICT gl_B0_ptr = &B_T[(tile_j * BLOCK_N) * K];
        const fp8 *RESTRICT gl_B128_ptr = &B_T[(tile_j * BLOCK_N + 128) * K];

        // Prologue
        // Pre-load a 256x128 tile of A/B
        // Each one of these issue 4 global -> LDS instructions so after these
        // two blocks I'll have 32 operations in flight before loading the first
        // fragment of A I need the first one to be done so 32-4 -> 28
        load_lds<K>(A_lds[cur][0], gl_A0_ptr, 0 * k_step, global_offset); // wait 28
        load_lds<K>(B_lds[cur][0], gl_B0_ptr, 0 * k_step, global_offset); // wait 24
        load_lds<K>(B_lds[cur][1], gl_B128_ptr, 0 * k_step, global_offset); // wait 20
        load_lds<K>(A_lds[cur][1], gl_A128_ptr, 0 * k_step, global_offset); // wait 16

        // Issue also the load of the next tile
        load_lds<K>(A_lds[next][0], gl_A0_ptr, 1 * k_step, global_offset); // wait 12
        load_lds<K>(B_lds[next][0], gl_B0_ptr, 1 * k_step, global_offset); // wait 8
        load_lds<K>(B_lds[next][1], gl_B128_ptr, 1 * k_step, global_offset); // wait 4
        load_lds<K>(A_lds[next][1], gl_A128_ptr, 1 * k_step, global_offset); // wait 0

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(28)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Load the first 64x128 subtile of A from LDS to registers
        load_rt(A_lds[cur][0], a[0], wave_i);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(24)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_rt(B_lds[cur][0], b[0], wave_j);

// Main loop
#pragma unroll
        for (int k = 0; k < k_iters - 2; ++k, cur ^= 1, next ^= 1) {
            const int k_offset = (k + 2) * k_step;
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(16)"); // Wait for A/B cur
            asm volatile("s_waitcnt lgkmcnt(0)"); // Wait for the registers to be loaded entirely
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Issue global load for A[cur] 1st half
            // Issue register load for B[cur] 2nd half
            interleaved_cluster<K>(A_lds[cur][0], gl_A0_ptr, k_offset, global_offset,
                                   wave_j, b[1], B_lds[cur][1], a[0], b[0], c[0][0]);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)"); // Wait for b1
            __builtin_amdgcn_sched_barrier(0);

            // Issue global load for B[cur] 1st half
            // Issue register load for A[cur] 2nd half
            interleaved_cluster<K>(B_lds[cur][0], gl_B0_ptr, k_offset, global_offset,
                                   wave_i, a[1], A_lds[cur][1], a[0], b[1], c[0][1]);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(16)"); // wait for A/B next first half
            __builtin_amdgcn_s_barrier();
            asm volatile("s_waitcnt lgkmcnt(0)"); // Wait for a1
            __builtin_amdgcn_sched_barrier(0);

            interleaved_cluster<K>(B_lds[cur][1], gl_B128_ptr, k_offset, global_offset,
                                   wave_i, a[0], A_lds[next][0], a[1], b[0], c[1][0]);

            interleaved_cluster<K>(A_lds[cur][1], gl_A128_ptr, k_offset, global_offset,
                                   wave_j, b[0], B_lds[next][0], a[1], b[1], c[1][1]);
        }

        // Epilogue: k = k_iters - 2
        {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(16)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);

            load_rt(B_lds[cur][1], b[1], wave_j);

            __builtin_amdgcn_sched_barrier(0);
            mfma_ABt_all(c[0][0], a[0], b[0], c[0][0]);
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);

            load_rt(A_lds[cur][1], a[1], wave_i);

            __builtin_amdgcn_sched_barrier(0);
            mfma_ABt_all(c[0][1], a[0], b[1], c[0][1]);
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(8)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);

            load_rt(A_lds[next][0], a[0], wave_i);

            __builtin_amdgcn_sched_barrier(0);
            mfma_ABt_all(c[1][0], a[1], b[0], c[1][0]);
            __builtin_amdgcn_sched_barrier(0);

            load_rt(B_lds[next][0], b[0], wave_j);

            __builtin_amdgcn_sched_barrier(0);
            mfma_ABt_all(c[1][1], a[1], b[1], c[1][1]);
            __builtin_amdgcn_sched_barrier(0);

            cur ^= 1;
            next ^= 1;
        }
        // Epilogue: k = k_iters - 1
        {
            const int base_row = tile_i * BLOCK_M + wave_i * 64;
            const int base_col = tile_j * BLOCK_N + wave_j * 64;
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);

            load_rt(B_lds[cur][1], b[1], wave_j);

            __builtin_amdgcn_sched_barrier(0);
            mfma_ABt_all(c[0][0], a[0], b[0], c[0][0]);
            __builtin_amdgcn_sched_barrier(0);

            store_rt<N>(C, c[0][0], base_row + 0, base_col + 0);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);

            load_rt(A_lds[cur][1], a[1], wave_i);

            __builtin_amdgcn_sched_barrier(0);
            mfma_ABt_all(c[0][1], a[0], b[1], c[0][1]);
            __builtin_amdgcn_sched_barrier(0);

            store_rt<N>(C, c[0][1], base_row + 0, base_col + 128);

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
            mfma_ABt_all(c[1][0], a[1], b[0], c[1][0]);
            __builtin_amdgcn_sched_barrier(0);

            store_rt<N>(C, c[1][0], base_row + 128, base_col + 0);

            __builtin_amdgcn_sched_barrier(0);
            mfma_ABt_all(c[1][1], a[1], b[1], c[1][1]);
            __builtin_amdgcn_sched_barrier(0);

            store_rt<N>(C, c[1][1], base_row + 128, base_col + 128);
        }
        __builtin_amdgcn_sched_barrier(0);
    }
}

// ====================================================================
// ======================== Reference 4wave kernel
// ====================================================================

// 256 threads load one 128x128 FP8 tile from global to LDS without touching the
// registers K is the global memory stride.
template <int K>
static __device__ INLINE void
cooperative_load(fp8 *shmem, const fp8 *gmem_base, const int4 &offsets) {
    // 128x128 -> 16K
    // Eeach lane loads 16bytes/round
    // 4 waves --> 4 rounds/wave
    const int wave_id = threadIdx.x / 64;
    // Interleave waves: wave 0 loads the first 1k elements (8 rows of 128), wave 1 the second...
    auto lds_ptr = shmem + wave_id * 1024;

    auto buffer_dsc = make_buffer_resource(gmem_base, K * 128);
    for (int round = 0; round < 4; ++round) {
        // Each wave reads 1K bytes, there are 4 waves so the offset in LDS for each round is 4K
        auto lds_ptr_raw = (as3_uint32_ptr)((uintptr_t)lds_ptr + round * 4096);
        llvm_amdgcn_raw_buffer_load_lds(buffer_dsc, lds_ptr_raw, 16,
                                        offsets[round], 0, 0, 0);
    }
}

template <int M, int N, int K>
__global__
__launch_bounds__(256, 1) void fp8_gemm_4wave_256x256x128_ref(const fp8 *RESTRICT A,
                                                              const fp8 *RESTRICT B_T,
                                                              bf16 *RESTRICT C) {
    constexpr int BLOCK_M = 256;
    constexpr int BLOCK_N = 256;
    constexpr int BLOCK_K = 128;

    __shared__ fp8 A_lds[2][128 * BLOCK_K];
    __shared__ fp8 B_lds[2][128 * BLOCK_K];

    const int k_step = BLOCK_K;

    const int lane_id = threadIdx.x % 64;
    const int wave_id = threadIdx.x / 64;

    const int tile_i = blockIdx.x / (N / BLOCK_N);
    const int tile_j = blockIdx.x % (N / BLOCK_N);
    const int wave_i = wave_id / 2; // 0..1
    const int wave_j = wave_id % 2; // 0..1
    const int4 global_swizzle = compute_global_swizzle<K>();

    fp32x4 accum_00[4][4]{};
    fp32x4 accum_01[4][4]{};
    fp32x4 accum_10[4][4]{};
    fp32x4 accum_11[4][4]{};

    auto compute = [&](fp32x4(&acc)[4][4], fp8 *RESTRICT A_lds, fp8 *RESTRICT B_lds) {
#pragma unroll
        for (int ti = 0; ti < 4; ++ti) {
            RT_Frag a_frag{};
            const int row = wave_i * 64 + ti * 16 + lane_id % 16;
            // Each wave has to load 32 FP8 elements for both A and B
            // With a single ds_read_b128 I can load 16 of them

#pragma unroll
            for (int ld_step = 0; ld_step < 2; ++ld_step) {
                const int col = (lane_id / 16) * 16 + ld_step * 64;
                const auto swizzle = swizzle_128(row, col);
                const uint32_t lds_addr =
                (uint32_t)(uintptr_t)A_lds + (swizzle.row * BLOCK_K + swizzle.col);
                asm volatile("ds_read_b128 %0, %1\n"
                             : "=v"(a_frag.half[ld_step])
                             : "v"(lds_addr)
                             : "memory");
            }

#pragma unroll
            for (int tj = 0; tj < 4; ++tj) {
                RT_Frag b_frag{};
#pragma unroll
                for (int ld_step = 0; ld_step < 2; ++ld_step) {
                    const int row = wave_j * 64 + tj * 16 + (lane_id % 16);
                    const int col = (lane_id / 16) * 16 + ld_step * 64;
                    const auto swizzle = swizzle_128(row, col);
                    const uint32_t lds_addr = (uint32_t)(uintptr_t)B_lds +
                                              (swizzle.row * BLOCK_K + swizzle.col);
                    asm volatile("ds_read_b128 %0, %1\n"
                                 : "=v"(b_frag.half[ld_step])
                                 : "v"(lds_addr)
                                 : "memory");
                }

                asm volatile("s_waitcnt lgkmcnt(0)");

                __builtin_amdgcn_sched_barrier(0);
                mfma_16x16x128(acc[ti][tj], a_frag.full, b_frag.full, acc[ti][tj]);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    };

    for (int k = 0; k < K; k += k_step) {
        cooperative_load<K>(A_lds[0], &A[(tile_i * BLOCK_M) * K + k], global_swizzle);
        cooperative_load<K>(A_lds[1], &A[(tile_i * BLOCK_M + 128) * K + k], global_swizzle);
        cooperative_load<K>(B_lds[0], &B_T[(tile_j * BLOCK_N) * K + k], global_swizzle);
        cooperative_load<K>(B_lds[1], &B_T[(tile_j * BLOCK_N + 128) * K + k], global_swizzle);

        asm volatile("s_waitcnt vmcnt(0)");

        compute(accum_00, A_lds[0], B_lds[0]);
        compute(accum_01, A_lds[0], B_lds[1]);
        compute(accum_10, A_lds[1], B_lds[0]);
        compute(accum_11, A_lds[1], B_lds[1]);

        __builtin_amdgcn_s_barrier();
    }

    const int group = lane_id / 16;
    const int col_lane = lane_id % 16;

    auto store_tile = [&](fp32x4(&acc)[4][4], const int base_row, const int base_col) {
#pragma unroll
        for (int ti = 0; ti < 4; ++ti) {
#pragma unroll
            for (int tj = 0; tj < 4; ++tj) {
                const int row = base_row + ti * 16 + group * 4;
                const int col = base_col + tj * 16 + col_lane;
                C[(row + 0) * N + col] = __float2bfloat16(acc[ti][tj][0]);
                C[(row + 1) * N + col] = __float2bfloat16(acc[ti][tj][1]);
                C[(row + 2) * N + col] = __float2bfloat16(acc[ti][tj][2]);
                C[(row + 3) * N + col] = __float2bfloat16(acc[ti][tj][3]);
            }
        }
    };

    const int base_row = tile_i * BLOCK_M + wave_i * 64;
    const int base_col = tile_j * BLOCK_N + wave_j * 64;

    store_tile(accum_00, base_row + 0, base_col + 0);
    store_tile(accum_01, base_row + 0, base_col + 128);
    store_tile(accum_10, base_row + 128, base_col + 0);
    store_tile(accum_11, base_row + 128, base_col + 128);
}
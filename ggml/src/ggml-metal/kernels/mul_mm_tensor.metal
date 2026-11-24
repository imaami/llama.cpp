#include "common.h"
#include "dequantize.h"

#ifdef GGML_METAL_HAS_TENSOR

constant bool    FC_mul_mm_bc_inp       [[function_constant(FC_MUL_MM + 0)]];
constant bool    FC_mul_mm_bc_out       [[function_constant(FC_MUL_MM + 1)]];
constant short   FC_mul_mm_ne12         [[function_constant(FC_MUL_MM + 2)]];
constant short   FC_mul_mm_ne13         [[function_constant(FC_MUL_MM + 3)]];
constant short   FC_mul_mm_r2           [[function_constant(FC_MUL_MM + 4)]];
constant short   FC_mul_mm_r3           [[function_constant(FC_MUL_MM + 5)]];
constant int32_t FC_mul_mm_id_map0_ne02 [[function_constant(FC_MUL_MM + 7)]];

// Q1_0 products made only of full tiles (M % 64, N % 128, K % 32 all zero): kernel_mul_mm with a static K32 extent and no bounds handling
kernel void kernel_mul_mm_q1_0_f32_k32(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
    constexpr int NK  = N_MM_NK_TOTAL;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;
    static_assert(NUM_THREADS == NRA * N_MM_NK, "one 16-weight chunk per thread");

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    const int im  = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // same work mapping as kernel_mul_mm: row = tiitg / N_MM_NK, chunk = tiitg % N_MM_NK
    const int   row    = tiitg / N_MM_NK;
    const short k_base = (tiitg % N_MM_NK) * 16;

    threadgroup half * sa = (threadgroup half *) shmem;

    device const block_q1_0 * row_ptr = (device const block_q1_0 *)(srcA + args.nb01 * (ra + row) + offset0);
    device float * ptrB = (device float *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(float);

    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NRA));
    auto tB = tensor(ptrB + rb * strideB, dextents<int32_t, 2>(NK, NRB), array<int, 2>({1, strideB}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        const int k_pos = loop_k + k_base;

        half4x4 temp_a;
        dequantize_q1_0(row_ptr + k_pos / QK1_0, (k_pos / 16) % (QK1_0 / 16), temp_a);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            sa[row * NK + k_base + i] = temp_a[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(NK, NRB), array<int, 2>({1, strideB}));

        mm.run(tBv, tA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;

    auto tD = tensor(dstTile, dextents<int32_t, 2>(NRA, NRB), array<int, 2>({1, M}));
    cT.store(tD);
}

// PQ2_0 x F32 for a few src1 rows (speculative verify widths). One simdgroup owns a
// 16 x 32 output tile (16 src1 rows by 32 weight rows) in a matmul2d destination
// cooperative tensor for its whole K range. Weights are decoded straight into the
// right-hand cooperative tensor, so the K loop has no threadgroup staging and no
// barriers; the generic tensor mul_mm tile is 64 x 128 and would compute 8x more
// columns than a 16-row verify needs. NCB simdgroups cover NCB column blocks and NKS
// simdgroups split K for each block; the split partials are summed once at the end.
//
// Operand layout (probed on M5 with get_multidimensional_index): lane L holds A rows
// fm, fm+8 at k fn..fn+3, B (transposed) weight rows fm+8j (j = 0..3) at the same k,
// and C rows fm, fm+8 at columns fn..fn+3 and fn+16..fn+19.
//
// Adapted from the few-row qmm core in the Yukon MLX.fast promoted submissions
// (Apache-2.0); the PQ2_0 block layout is different, so the decode is new.
template <short NCB, short NKS>
kernel void kernel_mul_mm_pq2_0_f32_fewrow(
        constant ggml_metal_kargs_mul_mm_fewrow & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int K = args.ne00;
    const int N = args.ne01;
    const int M = args.ne11;

    const short cb = sgitg / NKS;
    const short ks = sgitg % NKS;

    const int col0 = (tgpig.x*NCB + cb)*32;
    const int row0 = tgpig.y*16;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 16, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> mm;

    auto cA = mm.get_left_input_cooperative_tensor<half, half, float>();
    auto cB = mm.get_right_input_cooperative_tensor<half, half, float>();
    auto cC = mm.get_destination_cooperative_tensor<remove_addrspace_t<decltype(cA)>, remove_addrspace_t<decltype(cB)>, float>();

    FOR_UNROLL (short i = 0; i < 16; ++i) {
        cC[i] = 0.0f;
    }

    const short fn = 4*(tiisg & 1) + 8*((tiisg >> 3) & 1);
    const short fm = ((tiisg >> 1) & 3) | (((tiisg >> 4) & 1) << 2);
    const short fb = fn >> 2; // this lane's byte within a 16-value word

    // the quad (same fm) shares one 32-byte qs run per weight row and block: quad lane w
    // loads bytes [8w, 8w+8), i.e. the words of steps 2w and 2w+1
    ushort qlane[4];
    FOR_UNROLL (short w = 0; w < 4; ++w) {
        qlane[w] = ushort((tiisg & ~9) | (w & 1) | ((w >> 1) << 3));
    }

    device const block_pq2_0 * wrow[4];
    FOR_UNROLL (short j = 0; j < 4; ++j) {
        wrow[j] = (device const block_pq2_0 *) (src0 + args.nb01*min(col0 + fm + 8*j, N - 1));
    }

    device const float * x0 = (device const float *) (src1 + args.nb11*min(row0 + fm,     M - 1)) + fn;
    device const float * x1 = (device const float *) (src1 + args.nb11*min(row0 + fm + 8, M - 1)) + fn;

    const int nb = K/QK_PQ2_0;

    for (int ib = ks; ib < nb; ib += NKS) {
        uint  wq[4][2];
        half  d[4];
        FOR_UNROLL (short j = 0; j < 4; ++j) {
            device const ushort * q = (device const ushort *) (wrow[j][ib].qs) + 4*fb;
            wq[j][0] = uint(q[0]) | (uint(q[1]) << 16);
            wq[j][1] = uint(q[2]) | (uint(q[3]) << 16);
            d[j] = wrow[j][ib].d;
        }

        FOR_UNROLL (short st = 0; st < 8; ++st) {
            const int k = ib*QK_PQ2_0 + st*16;

            const float4 xa = *((device const float4 *) (x0 + k));
            const float4 xb = *((device const float4 *) (x1 + k));
            FOR_UNROLL (short q = 0; q < 4; ++q) {
                cA[q]     = (half) xa[q];
                cA[4 + q] = (half) xb[q];
            }

            FOR_UNROLL (short j = 0; j < 4; ++j) {
                const uint  word = simd_shuffle(wq[j][st & 1], qlane[st >> 1]);
                const uint  by   = (word >> (8*fb)) & 0xffu;
                FOR_UNROLL (short q = 0; q < 4; ++q) {
                    cB[4*j + q] = (half(float((by >> (2*q)) & 3u)) - (half) 1.0h)*d[j];
                }
            }

            mm.run(cA, cB, cC);
        }
    }

    if (NKS > 1) {
        threadgroup float * red = (threadgroup float *) shmem + cb*(NKS - 1)*16*32;
        if (ks > 0) {
            FOR_UNROLL (short i = 0; i < 16; ++i) {
                red[((ks - 1)*16 + i)*32 + tiisg] = cC[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks > 0) {
            return;
        }
        FOR_UNROLL (short s = 1; s < NKS; ++s) {
            FOR_UNROLL (short i = 0; i < 16; ++i) {
                cC[i] += red[((s - 1)*16 + i)*32 + tiisg];
            }
        }
    }

    device float * y = (device float *) dst;
    FOR_UNROLL (short i = 0; i < 16; ++i) {
        const int m = row0 + fm + 8*((i >> 2) & 1);
        const int n = col0 + fn + (i & 3) + 16*(i >> 3);
        if (m < M && n < N) {
            y[m*args.ne0 + n] = cC[i];
        }
    }
}

typedef decltype(kernel_mul_mm_pq2_0_f32_fewrow<1, 4>) mul_mm_pq2_0_fewrow_t;

template [[host_name("kernel_mul_mm_pq2_0_f32_fewrow_cb1_ks4")]] kernel mul_mm_pq2_0_fewrow_t kernel_mul_mm_pq2_0_f32_fewrow<1, 4>;
template [[host_name("kernel_mul_mm_pq2_0_f32_fewrow_cb2_ks2")]] kernel mul_mm_pq2_0_fewrow_t kernel_mul_mm_pq2_0_f32_fewrow<2, 2>;
template [[host_name("kernel_mul_mm_pq2_0_f32_fewrow_cb4_ks1")]] kernel mul_mm_pq2_0_fewrow_t kernel_mul_mm_pq2_0_f32_fewrow<4, 1>;
template [[host_name("kernel_mul_mm_pq2_0_f32_fewrow_cb1_ks2")]] kernel mul_mm_pq2_0_fewrow_t kernel_mul_mm_pq2_0_f32_fewrow<1, 2>;
template [[host_name("kernel_mul_mm_pq2_0_f32_fewrow_cb2_ks1")]] kernel mul_mm_pq2_0_fewrow_t kernel_mul_mm_pq2_0_f32_fewrow<2, 1>;

// each block_q contains 16*nl weights
template<
    typename SA, typename SB,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread typename value_type_4x4<SA>::type &),
    typename T0, typename T1>
kernel void kernel_mul_mm_tensor(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    using SA_4x4 = typename value_type_4x4<SA>::type;
    using SA_8x8 = typename value_type_8x8<SA>::type;
    using SB_2x4 = typename value_type_2x4<SB>::type;
    using SB_8x8 = typename value_type_8x8<SB>::type;
    using T0_4x4 = typename value_type_4x4<T0>::type;
    using T1_2x4 = typename value_type_2x4<T1>::type;

    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch dimension handling
    const int im = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // Batch offsets for srcA and srcB
    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // Tile dimensions
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    // tA wraps threadgroup memory
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure matmul operation
    // note: K is dynamic_extent (clamped to the valid range in PHASE 2), since a static
    //       N_MM_NK_TOTAL K tile would read src1 out of bounds when K % N_MM_NK_TOTAL != 0
    // ref: https://github.com/ggml-org/llama.cpp/pull/27064
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over K dimension
    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // === PHASE 1: Dequantization of A into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            // Bounds check: skip device read if row is out of matrix bounds
            if (ra + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
                    // Mirrors the legacy kernel's existing guard.
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Tensor matmul ===
        // Clamp the K extent of both operand tensors to the remaining valid K range so
        // the dynamic-K op never reads past the K extent of src1 (or the staged A tile).
        const int kExt = min(N_MM_NK_TOTAL, K - loop_k);

        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, N_MM_NK_TOTAL}));
        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(kExt, N - rb), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result tile to output matrix (with batch offset)
    // a slice is not used because its byte offset is int32 and wraps at 2 GiB
    // ref: https://github.com/ggml-org/llama.cpp/pull/28748
    const int tileM = min(NRA, M - ra);
    const int tileN = min(NRB, N - rb);

    device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;

    auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, tileN), array<int, 2>({1, M}));
    cT.store(tD);
}

template<typename S0, typename S1, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread typename value_type_4x4<S0>::type &), typename T0, typename T1>
kernel void kernel_mul_mm_id_tensor(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    using S0_4x4 = typename value_type_4x4<S0>::type;
    using S0_8x8 = typename value_type_8x8<S0>::type;
    using S1_2x4 = typename value_type_2x4<S1>::type;
    using S1_8x8 = typename value_type_8x8<S1>::type;
    using T0_4x4 = typename value_type_4x4<T0>::type;
    using T1_2x4 = typename value_type_2x4<T1>::type;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + NR0*NK*sizeof(S0));

    threadgroup float * sc = (threadgroup float *)(shmem);

    const int tile = tgpig.x; // entry tile, described by the prepass
    const int r0   = tgpig.y*NR0;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    // the prepass emits one (expert, first entry, valid rows) descriptor per tile that has
    // work - the grid is sized by a static upper bound, so the tail early exits
    if (tile >= (int) tpe_u32[0]) {
        return;
    }

    const int im  = tpe_u32[1 + 3*tile];     // expert
    const int idb = tpe_u32[1 + 3*tile + 1]; // first entry of this tile in hids

    // number of entries in this tile, always <= NR1
    const short nr1 = (short) ((tpe_u32[1 + 3*tile + 2] < NR1) ? tpe_u32[1 + 3*tile + 2] : NR1);

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63

    const short il0 = (tiitg % NL0);

    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    const short iy = 8*(tiitg % NL1);

    // skip the upper half of the token tile when the expert did not fill it
    constexpr short NR1H = NR1/2;

    auto tA  = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));

    // sb is [NR1][NK] row-major
    auto tB0 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb,             dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + NR1H*NK,   dextents<int32_t, 2>(NK, NR1H));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;


    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    short il = il0;

    const int id = ids_i32[idb + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const int   i12 = (id / args.ne20);

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    const bool has_hi = nr1 > NR1H;

    auto cT0 = mm.get_destination_cooperative_tensor<decltype(tB0), decltype(tA), float>();
    auto cT1 = mm.get_destination_cooperative_tensor<decltype(tB1), decltype(tA), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                //const short lx = (tiitg/NL1)%8;
                //const short ly = i;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            //const short lx = i;
            const short ly = (tiitg/NL1)%8;
            //const short lx = (tiitg/NL1)%8;
            //const short ly = i;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA  = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);

        mm.run(sB0, sA, cT0);

        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);

            mm.run(sB1, sA, cT1);
        }
    }

    // block is smaller than 64x32, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc,             dextents<int32_t, 2>(NR0, NR1H));
    cT0.store(tC0);

    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + NR1H*NR0, dextents<int32_t, 2>(NR0, NR1H));
        cT1.store(tC1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[idb + j];

        const short ide = id % args.ne20;
        const int   idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + (uint64_t) ide*args.ne0 + (uint64_t) idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_tensor<half, half, float4x4, 1, dequantize_f32, float, float>) mul_mm_tensor_t;

template [[host_name("kernel_mul_mm_tensor_f32_f32")]] kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, float4x4, 1, dequantize_f32, float, float>;
template [[host_name("kernel_mul_mm_tensor_f16_f32")]] kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, half4x4,  1, dequantize_f16, half,  float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_tensor_f32_f32_prec_bf16")]] kernel mul_mm_tensor_t kernel_mul_mm_tensor<bfloat, bfloat, float4x4, 1, dequantize_f32, float, float>;
#endif


#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_tensor_bf16_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<bfloat, bfloat, bfloat4x4, 1, dequantize_bf16, bfloat, float>;
template [[host_name("kernel_mul_mm_tensor_bf16_bf16")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<bfloat, bfloat, bfloat4x4, 1, dequantize_bf16, bfloat, bfloat>;
#endif
template [[host_name("kernel_mul_mm_tensor_q1_0_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q1_0,    8,     dequantize_q1_0,    float, float>;
template [[host_name("kernel_mul_mm_tensor_pq2_0_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_pq2_0,    8,     dequantize_pq2_0,    float, float>;
template [[host_name("kernel_mul_mm_tensor_ptq1_0_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_ptq1_0,    8,     dequantize_ptq1_0,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q2_0_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q2_0,    4,     dequantize_q2_0,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q4_0_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q4_0,    2,     dequantize_q4_0,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q4_1_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q4_1,    2,     dequantize_q4_1,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q5_0_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q5_0,    2,     dequantize_q5_0,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q5_1_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q5_1,    2,     dequantize_q5_1,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q8_0_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q8_0,    2,     dequantize_q8_0,    float, float>;
template [[host_name("kernel_mul_mm_tensor_mxfp4_f32")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_mxfp4,   2,     dequantize_mxfp4,   float, float>;
template [[host_name("kernel_mul_mm_tensor_nvfp4_f32")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_nvfp4,   4,     dequantize_nvfp4,   float, float>;
template [[host_name("kernel_mul_mm_tensor_q2_K_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q3_K_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q4_K_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q5_K_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_tensor_q6_K_f32")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_tensor_iq2_xxs_f32")]] kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_tensor_iq2_xs_f32")]]  kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_tensor_iq3_xxs_f32")]] kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_tensor_iq3_s_f32")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_tensor_iq2_s_f32")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_tensor_iq1_s_f32")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_tensor_iq1_m_f32")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_tensor_iq4_nl_f32")]]  kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_tensor_iq4_xs_f32")]]  kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_tensor_tq2_0_f32")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;

template [[host_name("kernel_mul_mm_tensor_f32_f16")]]     kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, float4x4,      1,     dequantize_f32,     float, half>;
template [[host_name("kernel_mul_mm_tensor_f16_f16")]]     kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, half4x4,       1,     dequantize_f16,     half,  half>;
template [[host_name("kernel_mul_mm_tensor_q1_0_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q1_0,    8,     dequantize_q1_0,    float, half>;
template [[host_name("kernel_mul_mm_tensor_pq2_0_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_pq2_0,    8,     dequantize_pq2_0,    float, half>;
template [[host_name("kernel_mul_mm_tensor_ptq1_0_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_ptq1_0,    8,     dequantize_ptq1_0,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q2_0_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q2_0,    4,     dequantize_q2_0,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q4_0_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q4_0,    2,     dequantize_q4_0,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q4_1_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q4_1,    2,     dequantize_q4_1,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q5_0_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q5_0,    2,     dequantize_q5_0,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q5_1_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q5_1,    2,     dequantize_q5_1,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q8_0_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q8_0,    2,     dequantize_q8_0,    float, half>;
template [[host_name("kernel_mul_mm_tensor_mxfp4_f16")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_mxfp4,   2,     dequantize_mxfp4,   float, half>;
template [[host_name("kernel_mul_mm_tensor_nvfp4_f16")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_nvfp4,   4,     dequantize_nvfp4,   float, half>;
template [[host_name("kernel_mul_mm_tensor_q2_K_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q2_K,    QK_NL, dequantize_q2_K,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q3_K_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q3_K,    QK_NL, dequantize_q3_K,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q4_K_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q4_K,    QK_NL, dequantize_q4_K,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q5_K_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q5_K,    QK_NL, dequantize_q5_K,    float, half>;
template [[host_name("kernel_mul_mm_tensor_q6_K_f16")]]    kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_q6_K,    QK_NL, dequantize_q6_K,    float, half>;
template [[host_name("kernel_mul_mm_tensor_iq2_xxs_f16")]] kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, half>;
template [[host_name("kernel_mul_mm_tensor_iq2_xs_f16")]]  kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, half>;
template [[host_name("kernel_mul_mm_tensor_iq3_xxs_f16")]] kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, half>;
template [[host_name("kernel_mul_mm_tensor_iq3_s_f16")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, half>;
template [[host_name("kernel_mul_mm_tensor_iq2_s_f16")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, half>;
template [[host_name("kernel_mul_mm_tensor_iq1_s_f16")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, half>;
template [[host_name("kernel_mul_mm_tensor_iq1_m_f16")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, half>;
template [[host_name("kernel_mul_mm_tensor_iq4_nl_f16")]]  kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq4_nl,  2,     dequantize_iq4_nl,  float, half>;
template [[host_name("kernel_mul_mm_tensor_iq4_xs_f16")]]  kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, half>;
template [[host_name("kernel_mul_mm_tensor_tq2_0_f16")]]   kernel mul_mm_tensor_t kernel_mul_mm_tensor<half, half, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, half>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id_tensor<half, half, float4x4, 1, dequantize_f32, float, float>) mul_mm_id_tensor_t;

template [[host_name("kernel_mul_mm_id_tensor_f32_f32")]]     kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, float4x4, 1, dequantize_f32, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_f16_f32")]]     kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, half4x4,  1, dequantize_f16, half,  float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_tensor_bf16_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<bfloat, bfloat, bfloat4x4, 1, dequantize_bf16, bfloat, float>;
#endif
template [[host_name("kernel_mul_mm_id_tensor_q1_0_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q1_0, 8, dequantize_q1_0, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_pq2_0_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_pq2_0, 8, dequantize_pq2_0, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_ptq1_0_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_ptq1_0, 8, dequantize_ptq1_0, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q2_0_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q2_0, 4, dequantize_q2_0, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q4_0_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q4_0, 2, dequantize_q4_0, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q4_1_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q4_1, 2, dequantize_q4_1, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q5_0_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q5_0, 2, dequantize_q5_0, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q5_1_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q5_1, 2, dequantize_q5_1, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q8_0_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q8_0, 2, dequantize_q8_0, float, float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_tensor_mxfp4_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<bfloat, bfloat, block_mxfp4, 2, dequantize_mxfp4, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_nvfp4_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<bfloat, bfloat, block_nvfp4, 4, dequantize_nvfp4, float, float>;
#else
template [[host_name("kernel_mul_mm_id_tensor_mxfp4_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_mxfp4, 2, dequantize_mxfp4, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_nvfp4_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_nvfp4, 4, dequantize_nvfp4, float, float>;
#endif
template [[host_name("kernel_mul_mm_id_tensor_q2_K_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q3_K_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q4_K_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q5_K_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q6_K_f32")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_xxs_f32")]] kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_xs_f32")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq3_xxs_f32")]] kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq3_s_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_s_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq1_s_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq1_m_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq4_nl_f32")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq4_xs_f32")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_tensor_tq2_0_f32")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_tensor_f32_f32_prec_bf16")]]     kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<bfloat, bfloat, float4x4, 1, dequantize_f32, float, float>;
#endif

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_tensor_f16_f32_prec_bf16")]]     kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, half4x4,       1,     dequantize_f16,     half,  float>;
template [[host_name("kernel_mul_mm_id_tensor_q1_0_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q1_0,    8,     dequantize_q1_0,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_pq2_0_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_pq2_0,    8,     dequantize_pq2_0,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_ptq1_0_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_ptq1_0,    8,     dequantize_ptq1_0,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q2_0_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q2_0,    4,     dequantize_q2_0,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q4_0_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q4_0,    2,     dequantize_q4_0,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q4_1_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q4_1,    2,     dequantize_q4_1,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q5_0_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q5_0,    2,     dequantize_q5_0,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q5_1_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q5_1,    2,     dequantize_q5_1,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q8_0_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q8_0,    2,     dequantize_q8_0,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q2_K_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q2_K,    QK_NL, dequantize_q2_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q3_K_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q3_K,    QK_NL, dequantize_q3_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q4_K_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q4_K,    QK_NL, dequantize_q4_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q5_K_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q5_K,    QK_NL, dequantize_q5_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_q6_K_f32_prec_bf16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_q6_K,    QK_NL, dequantize_q6_K,    float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_xxs_f32_prec_bf16")]] kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_xs_f32_prec_bf16")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq3_xxs_f32_prec_bf16")]] kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq3_s_f32_prec_bf16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_s_f32_prec_bf16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq1_s_f32_prec_bf16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq1_m_f32_prec_bf16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq4_nl_f32_prec_bf16")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq4_nl,  2,     dequantize_iq4_nl,  float, float>;
template [[host_name("kernel_mul_mm_id_tensor_iq4_xs_f32_prec_bf16")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float>;
template [[host_name("kernel_mul_mm_id_tensor_tq2_0_f32_prec_bf16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, bfloat, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, float>;
#endif

template [[host_name("kernel_mul_mm_id_tensor_f32_f16")]]     kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, float4x4,   1, dequantize_f32,  float, half>;
template [[host_name("kernel_mul_mm_id_tensor_f16_f16")]]     kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, half4x4,    1, dequantize_f16,  half,  half>;
template [[host_name("kernel_mul_mm_id_tensor_q1_0_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q1_0, 8, dequantize_q1_0, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_pq2_0_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_pq2_0, 8, dequantize_pq2_0, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_ptq1_0_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_ptq1_0, 8, dequantize_ptq1_0, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q2_0_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q2_0, 4, dequantize_q2_0, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q4_0_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q4_0, 2, dequantize_q4_0, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q4_1_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q4_1, 2, dequantize_q4_1, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q5_0_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q5_0, 2, dequantize_q5_0, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q5_1_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q5_1, 2, dequantize_q5_1, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q8_0_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q8_0, 2, dequantize_q8_0, float, half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_tensor_mxfp4_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<bfloat, bfloat, block_mxfp4, 2, dequantize_mxfp4, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_nvfp4_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<bfloat, bfloat, block_nvfp4, 4, dequantize_nvfp4, float, half>;
#else
template [[host_name("kernel_mul_mm_id_tensor_mxfp4_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_mxfp4, 2, dequantize_mxfp4, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_nvfp4_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_nvfp4, 4, dequantize_nvfp4, float, half>;
#endif
template [[host_name("kernel_mul_mm_id_tensor_q2_K_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q2_K,    QK_NL, dequantize_q2_K,    float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q3_K_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q3_K,    QK_NL, dequantize_q3_K,    float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q4_K_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q4_K,    QK_NL, dequantize_q4_K,    float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q5_K_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q5_K,    QK_NL, dequantize_q5_K,    float, half>;
template [[host_name("kernel_mul_mm_id_tensor_q6_K_f16")]]    kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_q6_K,    QK_NL, dequantize_q6_K,    float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_xxs_f16")]] kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_xs_f16")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq3_xxs_f16")]] kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq3_s_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq3_s,   QK_NL, dequantize_iq3_s,   float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq2_s_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq2_s,   QK_NL, dequantize_iq2_s,   float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq1_s_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq1_s,   QK_NL, dequantize_iq1_s,   float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq1_m_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq1_m,   QK_NL, dequantize_iq1_m,   float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq4_nl_f16")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq4_nl,  2,     dequantize_iq4_nl,  float, half>;
template [[host_name("kernel_mul_mm_id_tensor_iq4_xs_f16")]]  kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, half>;
template [[host_name("kernel_mul_mm_id_tensor_tq2_0_f16")]]   kernel mul_mm_id_tensor_t kernel_mul_mm_id_tensor<half, half, block_tq2_0,   QK_NL, dequantize_tq2_0,   float, half>;

#endif // GGML_METAL_HAS_TENSOR

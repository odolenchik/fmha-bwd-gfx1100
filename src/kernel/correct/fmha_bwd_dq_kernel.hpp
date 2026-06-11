#pragma once
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include "fmha_bwd_config.h"

using half_t = _Float16;
using bhalf_t = rocwmma::bfloat16_t;

// ============================================================
// Naive CPU-reference kernel (correct, slow) — baseline
// dQ = dS @ K  where: dS[MxN], K[N x K_dim] -> dQ[M x K_dim]
// Each thread computes one element of the output matrix.
// ============================================================
__global__ void fmha_bwd_dq_kernel_naive(const half_t* dS, const half_t* K, half_t* dQ,
                                    int M, int N, int K_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / K_dim;
    int k = idx % K_dim;

    if (m < M && k < K_dim) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum += static_cast<float>(dS[m * N + n]) *
                   static_cast<float>(K[n * K_dim + k]);
        }
        dQ[m * K_dim + k] = static_cast<half_t>(sum);
    }
}

// ============================================================
// WMMA-optimized kernel: dQ = dS @ K
// M x N  (dS)  x  N x K_dim (K)  ->  M x K_dim (dQ)
//
// Tile layout: BM blocks along M, BN blocks along N dimension of dS/K
// BK is the inner dimension chunk size for WMMA.
// Warp mapping: warp_m = warp_id % num_warp_rows, warp_k = warp_id / num_warp_rows
// ============================================================

__global__ void fmha_bwd_dq_kernel_wmma(
    const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
    int M_global, int N_global, int K_dim_global)
{
    __shared__ bhalf_t s_dS[BM][BN + LDS_PAD];
    __shared__ bhalf_t s_K[BK][BN + LDS_PAD];

    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bhalf_t, rocwmma::col_major>;
    using FragC = rocwmma::fragment<accumulator, 16, 16, 16, float>;
    using FragOut = rocwmma::fragment<accumulator, 16, 16, 16, bhalf_t>;

    int block_m = blockIdx.y;
    int block_k = blockIdx.x;

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;  // 8 warps per block (512/64)
    int warp_m_idx = warp_id % 4;
    int warp_k_idx = warp_id / 4;

    int m_start = block_m * BM;
    int k_start = block_k * BN;
    int w_m_start = m_start + warp_m_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // ---- Load dS tile into shared memory (full tile broadcast) ----
    for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
        int m = i / BN, n = i % BN;
        int gm = m_start + m;
        if (gm < M_global && n < N_global) {
            s_dS[m][n] = dS[gm * N_global + n];
        } else {
            s_dS[m][n] = half_t(0.0f);
        }
    }
    __syncthreads();

    // ---- WMMA multiply: accumulate dQ_tile = dS_tile @ K_slice ----
    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int n_start = 0; n_start < N_global; n_start += BK) {
        // Load K slice into shared memory
        for (int i = tid; i < BK * BN; i += BLOCK_SIZE) {
            int k_idx = i / BN, n = i % BN;
            int gn = n_start + n, gk = k_start + k_idx;
            s_K[k_idx][n] = (gn < N_global && gk < K_dim_global)
                            ? K[gn * K_dim_global + gk] : bhalf_t(0.0f);
        }
        __syncthreads();

        // WMMA: process 16x16 sub-tiles, two iterations for full BK-width
        for (int sub = 0; sub < 2; ++sub) {
            FragA a0;
            FragB b0;

            bhalf_t *a_ptr0 = reinterpret_cast<bhalf_t*>(&a0);
            bhalf_t *b_ptr0 = reinterpret_cast<bhalf_t*>(&b0);

            int n_off = sub * 16;

            // Fill A fragment from dS (row-major: m=rows, n=cols)
            for (int i = 0; i < 16; ++i) {
                int m_local = warp_m_idx * 16 + i;
                for (int j = 0; j < 16; ++j) {
                    int n = n_start + n_off + j;
                    bhalf_t val = (m_local < BM && n < BN) ? s_dS[m_local][n] : bhalf_t(0.0f);
                    a_ptr0[i * 16 + j] = val;
                }
            }

            // Fill B fragment from K slice (col-major indexing: b_ptr[j*16+i])
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = warp_k_idx * 16 + j;
                    bhalf_t val = (k_idx_local < BK && (n_off + i) < BN)
                                  ? s_K[k_idx_local][n_off + i] : bhalf_t(0.0f);
                    // Col-major: transpose during load — KEY for WMMA matrix_b layout
                    b_ptr0[j * 16 + i] = val;
                }
            }

            rocwmma::mma_sync(acc, a0, b0, acc);
        }
    }

    // ---- Write back results ----
    if (w_m_start < M_global && w_k_start < K_dim_global) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) {
            out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        }
        for (int i = 0; i < 16; ++i) {
            int m_local = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (m_local < M_global && k < K_dim_global) {
                    dQ[m_local * K_dim_global + k] = out.x[i * 16 + j];
                }
            }
        }
    }
}

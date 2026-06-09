#pragma once
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using bhalf_t = rocwmma::bfloat16_t;

// ============================================================
// WMMA-optimized kernel: dK = dS^T @ Q
// (dS[M x N])^T  x  Q[M x K_dim]  ->  N x K_dim (dK)
// dK[n][k] = sum_{m=0}^{M-1} dS[m][n] * Q[m][k]
//
// Tile layout: BN blocks along N, BK tiles along K_dim.
// WMMA shape: A is [N x M] (dS transposed), B is [M x K_dim], C = A @ B
// ============================================================

constexpr int BM = 64, BN = 64, BK = 32, LDS_PAD = 8, BLOCK_SIZE = 512;

__global__ void fmha_bwd_dk_kernel_wmma(
    const bhalf_t* dS,      // [M x N] row-major
    const bhalf_t* Q,       // [M x K_dim] row-major  
    bhalf_t* dK,            // [N x K_dim] row-major
    int M_global, int N_global, int K_dim_global)
{
    __shared__ bhalf_t s_dS[BK][BM + LDS_PAD];   // transposed: stores dS[m][n] at s_dS[n%BK][m]
    __shared__ bhalf_t s_Q[BM][BK + LDS_PAD];    // Q: [BM x BK]

    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    using FragOut = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, bhalf_t>;

    int block_n = blockIdx.x;      // which N tile (BN=64)
    int block_k = blockIdx.y;      // which K_dim tile (BK=32)
    
    int tid = threadIdx.x;
    int warp_id = tid / 64;        // 8 warps per block
    int warp_n_idx = warp_id % 4;   // which N sub-tile (0..3, each 16-wide)
    int warp_k_idx = warp_id / 4;   // which K_dim sub-tile (0..1, each 16-wide)
    
    int n_start = block_n * BN;      // starting column in dS, row in dK
    int k_start = block_k * BK;      // starting column in Q and dK
    int w_n_start = n_start + warp_n_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // ---- Load dS transposed into shared memory: s_dS[n%BK][m] = dS[m][n] ----
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int n_local = i / BM, m_local = i % BM;  // [BN x BM] layout
        s_dS[n_local % BK][m_local] = (n_start + n_local < N_global && m_local < M_global) 
                                       ? dS[m_local * N_global + n_start + n_local] : half_t(0.0f);
    }
    __syncthreads();

    // ---- WMMA accumulate: dK_tile = dS^T @ Q ----
    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int m_start = 0; m_start < M_global; m_start += BM) {
        int m_chunk = min(BM, M_global - m_start);

        // Load Q[m][k] into shared memory
        for (int i = tid; i < m_chunk * BK; i += blockDim.x) {
            int m_local = i / BK, k_idx = i % BK;
            s_Q[m_local][k_idx] = (m_start + m_local < M_global && k_start + k_idx < K_dim_global)
                                  ? Q[(m_start + m_local) * K_dim_global + k_start + k_idx] 
                                  : bhalf_t(0.0f);
        }
        __syncthreads();

        // WMMA: process 16x16 sub-tiles over M dimension (BM=64 -> 4 iterations of 16)
        for (int m_sub = 0; m_sub < 4; ++m_sub) {
            FragA a_frag;
            FragB b_frag;

            bhalf_t *a_ptr = reinterpret_cast<bhalf_t*>(&a_frag);
            bhalf_t *b_ptr = reinterpret_cast<bhalf_t*>(&b_frag);

            int m_off = m_sub * 16;

            // Fill A fragment from dS^T (row-major: n=rows, m=cols)
            for (int i = 0; i < 16; ++i) {
                int n_local = warp_n_idx * 16 + i;
                for (int j = 0; j < 16 && (m_off + j) < m_chunk; ++j) {
                    bhalf_t val = (n_local < BN && (m_off + j) < BM) 
                                  ? s_dS[n_local % BK][m_off + j] : bhalf_t(0.0f);
                    a_ptr[i * 16 + j] = val;
                }
            }

            // Fill B fragment from Q slice (col-major indexing for WMMA matrix_b)
            for (int i = 0; i < 16 && (m_off + i) < m_chunk; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = warp_k_idx * 16 + j;
                    bhalf_t val = ((m_off + i) < BM && k_idx_local < BK) 
                                  ? s_Q[m_off + i][k_idx_local] : bhalf_t(0.0f);
                    // Col-major transpose for WMMA matrix_b layout
                    b_ptr[j * 16 + i] = val;
                }
            }

            rocwmma::mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    // ---- Write back results: dK[n][k] from accumulator ----
    if (w_n_start < N_global && w_k_start < K_dim_global) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) {
            out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        }
        for (int i = 0; i < 16; ++i) {
            int n_local = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k_idx = w_k_start + j;
                if (n_local < N_global && k_idx < K_dim_global) {
                    dK[n_local * K_dim_global + k_idx] = out.x[i * 16 + j];
                }
            }
        }
    }
}

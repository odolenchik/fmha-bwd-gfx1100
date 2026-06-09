#pragma once
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

constexpr int BM = 64, BN = 64, BK = 32, WARP_SIZE = 64, BLOCK_SIZE = 512, LDS_PAD = 8;

// dP = dO @ V^T
// dO: [M x K_dim], V: [N x K_dim], output: [M x N]
__global__ void fmha_bwd_dp_kernel(const bhalf_t* dO, const bhalf_t* V, bhalf_t* dP, int M, int N, int K_dim) {
    __shared__ bhalf_t s_dO[BM][BK + LDS_PAD], s_V[BK][BN + LDS_PAD];

    using FragA = fragment<matrix_a, 16, 16, 16, bhalf_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, bhalf_t>;

    int block_m = blockIdx.y, block_n = blockIdx.x;
    int m_start = block_m * BM, n_start = block_n * BN;
    int tid = threadIdx.x;

    // BLOCK_SIZE=512 → 8 warps per block → 2x4 warp grid (warp_m x warp_n)
    int warp_id = tid / WARP_SIZE;
    int warp_m = warp_id / 4;        // row dimension: covers BM/16 = 4 tiles, we have 2 → covers half the rows
    int warp_n = warp_id % 4;        // col dimension: covers BN/16 = 4 tiles
    int w_m_start = m_start + warp_m * 16, w_n_start = n_start + warp_n * 16;

    FragC acc;
    fill_fragment(acc, 0.0f);

    // Loop over K dimension in BK-sized tiles
    for (int k_start = 0; k_start < K_dim; k_start += BK) {
        // Load dO tile [BM x BK] → shared memory row-major: s_dO[m][k]
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int m = i / BK, k = i % BK;
            int gm = m_start + m, gk = k_start + k;
            s_dO[m][k] = (gm < M && gk < K_dim) ? dO[gm * K_dim + gk] : bhalf_t(0.0f);
        }

        // Load V tile [BK x BN] → shared memory transposed: s_V[k][n]
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int k = i / BN, n = i % BN;
            int gk = k_start + k, gn = n_start + n;
            s_V[k][n] = (gk < K_dim && gn < N) ? V[gn * K_dim + gk] : bhalf_t(0.0f);
        }

        __syncthreads();

        // Each warp computes a 16x16 tile of the output matrix product
        if (w_m_start < M && w_n_start < N) {
            FragA a0, a1;
            FragB b0, b1;
            bhalf_t *a_ptr0 = reinterpret_cast<bhalf_t*>(&a0), *b_ptr0 = reinterpret_cast<bhalf_t*>(&b0);
            bhalf_t *a_ptr1 = reinterpret_cast<bhalf_t*>(&a1), *b_ptr1 = reinterpret_cast<bhalf_t*>(&b1);

            // BK=32, split into 2 sub-tiles of 16 for the K dimension
            for (int sub = 0; sub < 2; ++sub) {
                int k_off = sub * 16;

               // Load A: row_major — same tile for both sub-iterations (A doesn't change across K)
                for (int i = 0; i < 16; ++i) {
                    int m = warp_m * 16 + i;
                    for (int j = 0; j < 16; ++j) {
                        a_ptr0[i * 16 + j] = (m < BM && j < BK) ? s_dO[m][j] : bhalf_t(0.0f);
                        a_ptr1[i * 16 + j] = (m < BM && j < BK) ? s_dO[m][j] : bhalf_t(0.0f);
                    }
                }

                // Load B: col_major — different K-sub-tile per iteration
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        int n = warp_n * 16 + j;
                        bhalf_t val = ((k_off + i) < BK && n < BN) ? s_V[k_off + i][n] : bhalf_t(0.0f);
                        b_ptr0[j * 16 + i] = val;
                        b_ptr1[j * 16 + i] = val;
                    }
                }

                mma_sync(acc, a0, b0, acc);
                mma_sync(acc, a1, b1, acc);
            }
        }

        __syncthreads();
    }

    // Write result: convert float accumulator to bhalf and store
    if (w_m_start < M && w_n_start < N) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) {
            out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        }
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int n = w_n_start + j;
                if (m < M && n < N) {
                    dP[m * N + n] = static_cast<bhalf_t>(out.x[i * 16 + j]);
                }
            }
        }
    }
}

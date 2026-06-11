#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cmath>

using half_t = _Float16;
using bhalf_t = rocwmma::bfloat16_t;

constexpr int BM = 64, BN = 64, BK = 32, LDS_PAD = 8, BLOCK_SIZE = 512;

__global__ void fmha_bwd_dk_kernel_wmma(
    const bhalf_t* dS,      // [M x N] row-major
    const bhalf_t* Q,       // [M x K_dim] row-major
    bhalf_t* dK,            // [N x K_dim] row-major
    int M_global, int N_global, int K_dim_global)
{
    __shared__ bhalf_t s_dS[BN][BM + LDS_PAD];   // transposed: s_dS[n_local][m_local] = dS[m_local][n_start+n_local]
    __shared__ bhalf_t s_Q[BK+LDS_PAD][BM];      // Q slice: s_Q[k_idx_local][m_idx]

    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bhalf_t, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    using FragOut = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, bhalf_t>;

    int block_n = blockIdx.x;
    int block_k = blockIdx.y;

    int tid = threadIdx.x;
    int warp_id   = tid / 64;        // 0..7
    int warp_n_idx = warp_id % 4;    // N sub-tile (16-wide each, 4 across)
    int warp_k_idx = warp_id / 4;    // K sub-tile (16-wide each, 2 down)

    int n_start = block_n * BN;
    int k_start = block_k * BK;
    int w_n_start = n_start + warp_n_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // Load dS transposed: s_dS[n][m] = dS[m][n]
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int n_local = i / BM, m_local = i % BM;
        s_dS[n_local][m_local] = (n_start + n_local < N_global && m_local < M_global)
                                 ? dS[m_local * N_global + n_start + n_local] : half_t(0.0f);
    }
    __syncthreads();

    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int m_start = 0; m_start < M_global; m_start += BM) {
        int m_chunk = (BM < M_global - m_start) ? BM : (M_global - m_start);

        // Load Q slice: s_Q[k_idx_local][m_idx] = Q[m_idx][k_start+k_idx]
        for (int i = tid; i < BK * m_chunk; i += blockDim.x) {
            int k_local = i / m_chunk, m_idx = i % m_chunk;
            s_Q[k_local][m_idx] = (k_start + k_local < K_dim_global && m_start + m_idx < M_global)
                                  ? Q[(m_start + m_idx) * K_dim_global + k_start + k_local]
                                  : bhalf_t(0.0f);
        }
        __syncthreads();

        for (int m_sub = 0; m_sub < 4; ++m_sub) {
            FragA a_frag;
            FragB b_frag;

            bhalf_t *a_ptr = reinterpret_cast<bhalf_t*>(&a_frag);
            bhalf_t *b_ptr = reinterpret_cast<bhalf_t*>(&b_frag);

            int m_off = m_sub * 16;

            // Fill A from dS^T (row_major: n=rows, m=cols)
            for (int i = 0; i < 16; ++i) {
                int n_local = warp_n_idx * 16 + i;
                for (int j = 0; j < 16 && (m_off + j) < m_chunk; ++j) {
                    bhalf_t val = (n_local < BN && (m_off + j) < BM)
                                  ? s_dS[n_local][m_off + j] : bhalf_t(0.0f);
                    a_ptr[i * 16 + j] = val;
                }
            }

            // Fill B from Q slice (col_major transpose for WMMA matrix_b)
            for (int i = 0; i < 16 && (m_off + i) < m_chunk; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = warp_k_idx * 16 + j;
                    bhalf_t val = ((m_off + i) < BM && k_idx_local < BK)
                                  ? s_Q[k_idx_local][m_off + i] : bhalf_t(0.0f);
                    b_ptr[j * 16 + i] = val;
                }
            }

            rocwmma::mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    // Write back: dK[n][k] from accumulator
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

void cpu_ref_dk(const float* dS, const float* Q, float* dK, int M, int N, int K_dim) {
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                sum += dS[m * N + n] * Q[m * K_dim + k];
            }
            dK[n * K_dim + k] = sum;
        }
    }
}

int main() {
    const int M = 64, N = 64, K_dim = 32;

    bhalf_t h_dS[M*N], h_Q[M*K_dim];
    float f_dS[M*N], f_Q[M*K_dim], f_dk_ref[N*K_dim];

    for (int i = 0; i < M*N; ++i) {
        h_dS[i] = static_cast<bhalf_t>(static_cast<float>(i) * 0.1f);
        f_dS[i] = static_cast<float>(i) * 0.1f;
    }
    for (int i = 0; i < M*K_dim; ++i) {
        float val = static_cast<float>((i % 7)) * 0.2f - 1.0f;
        h_Q[i] = static_cast<bhalf_t>(val);
        f_Q[i] = val;
    }

    cpu_ref_dk(f_dS, f_Q, f_dk_ref, M, N, K_dim);

    bhalf_t *d_dS, *d_Q, *d_dK;
    hipMalloc(&d_dS, M*N*sizeof(bhalf_t));
    hipMalloc(&d_Q, M*K_dim*sizeof(bhalf_t));
    hipMalloc(&d_dK, N*K_dim*sizeof(bhalf_t));

    hipMemcpy(d_dS, h_dS, M*N*sizeof(bhalf_t), hipMemcpyHostToDevice);
    hipMemcpy(d_Q, h_Q, M*K_dim*sizeof(bhalf_t), hipMemcpyHostToDevice);

    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK);
    printf("Grid: (%d, %d)\n", grid.x, grid.y);

    hipLaunchKernelGGL(fmha_bwd_dk_kernel_wmma, grid, dim3(BLOCK_SIZE), 0, 0, d_dS, d_Q, d_dK, M, N, K_dim);
    hipDeviceSynchronize();

    bhalf_t h_dK_gpu[N*K_dim];
    hipMemcpy(h_dK_gpu, d_dK, N*K_dim*sizeof(bhalf_t), hipMemcpyDeviceToHost);

    float max_err = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < N * K_dim; ++i) {
        float gpu_val = static_cast<float>(h_dK_gpu[i]);
        float ref_val = f_dk_ref[i];
        float err = fabsf(gpu_val - ref_val);
        max_err = fmaxf(max_err, err);
        if (err > 2.0f) {
            mismatches++;
            int n = i / K_dim, k = i % K_dim;
            printf("MISMATCH dK[%d][%d]: gpu=%.4f ref=%.4f err=%.4f\n",
                   n, k, gpu_val, ref_val, err);
        }
    }

    printf("\nTotal mismatches (>2.0): %d / %d\n", mismatches, N * K_dim);
    printf("Max error: %.6f\n", max_err);
    return (mismatches == 0) ? 0 : 1;
}

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>

using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

constexpr int BM = 64, BN = 64, BK = 32, LDS_PAD = 8, BLOCK_SIZE = 512;

// dV kernel: inline copy to avoid external header dependency
__global__ void fmha_bwd_dv_kernel_wmma(
    const bhalf_t* P,       // [M x N] row-major
    const bhalf_t* dO,      // [M x K_dim] row-major
    bhalf_t* dV,            // [N x K_dim] row-major
    int M_global, int N_global, int K_dim_global)
{
    __shared__ bhalf_t s_P[BN][BM + LDS_PAD];
    __shared__ bhalf_t s_dO[BK+LDS_PAD][BM];

    using FragA = fragment<matrix_a, 16, 16, 16, bhalf_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, bhalf_t>;

    int block_n = blockIdx.x;
    int block_k = blockIdx.y;

    int tid = threadIdx.x;
    int warp_id   = tid / 64;
    int warp_n_idx = warp_id % 4;
    int warp_k_idx = warp_id / 4;

    int n_start = block_n * BN;
    int k_start = block_k * BK;
    int w_n_start = n_start + warp_n_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // Load P transposed: s_P[n][m] = P[m][n]
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int n_local = i / BM, m_local = i % BM;
        s_P[n_local][m_local] = (n_start + n_local < N_global && m_local < M_global)
                                ? P[m_local * N_global + n_start + n_local] : _Float16(0.0f);
    }
    __syncthreads();

    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int m_start = 0; m_start < M_global; m_start += BM) {
        int m_chunk = (BM < M_global - m_start) ? BM : (M_global - m_start);

        for (int i = tid; i < BK * m_chunk; i += blockDim.x) {
            int k_local = i / m_chunk, m_idx = i % m_chunk;
            s_dO[k_local][m_idx] = (k_start + k_local < K_dim_global && m_start + m_idx < M_global)
                                   ? dO[(m_start + m_idx) * K_dim_global + k_start + k_local]
                                   : bhalf_t(0.0f);
        }
        __syncthreads();

        for (int m_sub = 0; m_sub < 4; ++m_sub) {
            FragA a_frag;
            FragB b_frag;

            bhalf_t *a_ptr = reinterpret_cast<bhalf_t*>(&a_frag);
            bhalf_t *b_ptr = reinterpret_cast<bhalf_t*>(&b_frag);

            int m_off = m_sub * 16;

            for (int i = 0; i < 16; ++i) {
                int n_local = warp_n_idx * 16 + i;
                for (int j = 0; j < 16 && (m_off + j) < m_chunk; ++j) {
                    bhalf_t val = (n_local < BN && (m_off + j) < BM)
                                  ? s_P[n_local][m_off + j] : bhalf_t(0.0f);
                    a_ptr[i * 16 + j] = val;
                }
            }

            for (int i = 0; i < 16 && (m_off + i) < m_chunk; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = warp_k_idx * 16 + j;
                    bhalf_t val = ((m_off + i) < BM && k_idx_local < BK)
                                  ? s_dO[k_idx_local][m_off + i] : bhalf_t(0.0f);
                    b_ptr[j * 16 + i] = val;
                }
            }

            rocwmma::mma_sync(acc, a_frag, b_frag, acc);
        }
    }

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
                    dV[n_local * K_dim_global + k_idx] = out.x[i * 16 + j];
                }
            }
        }
    }
}

int main() {
    const int M = 64, N = 64, K_dim = 32;

    bhalf_t h_P[M*N], h_dO[M*K_dim];
    for (int i = 0; i < M*N; ++i) h_P[i] = static_cast<bhalf_t>(1.0f);
    for (int i = 0; i < M*K_dim; ++i) h_dO[i] = static_cast<bhalf_t>(-0.5f);

    bhalf_t *d_P, *d_dO, *d_result;
    hipMalloc(&d_P, M*N*sizeof(bhalf_t));
    hipMalloc(&d_dO, M*K_dim*sizeof(bhalf_t));
    hipMalloc(&d_result, N*K_dim*sizeof(bhalf_t));

    hipMemcpy(d_P, h_P, M*N*sizeof(bhalf_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO, M*K_dim*sizeof(bhalf_t), hipMemcpyHostToDevice);

    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK);
    printf("Grid: %d x %d, Block: %d\n", grid.x, grid.y, BLOCK_SIZE);

    hipLaunchKernelGGL(fmha_bwd_dv_kernel_wmma, grid, dim3(BLOCK_SIZE), 0, 0, d_P, d_dO, d_result, M, N, K_dim);
    hipDeviceSynchronize();

    bhalf_t h_result[N*K_dim];
    hipMemcpy(h_result, d_result, N*K_dim*sizeof(bhalf_t), hipMemcpyDeviceToHost);

    printf("dV kernel result[0..3]: %.4f %.4f %.4f %.4f\n",
           static_cast<float>(h_result[0]), static_cast<float>(h_result[1]),
           static_cast<float>(h_result[2]), static_cast<float>(h_result[3]));
    printf("Expected: -32.0 (M=64 * 1.0 * -0.5)\n");

    hipFree(d_P); hipFree(d_dO); hipFree(d_result);
    return 0;
}

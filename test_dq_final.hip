#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>

using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

constexpr int BM = 64, BN = 64, BK = 32, LDS_PAD = 8, BLOCK_SIZE = 512;

__global__ void dq_kernel(
    const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
    int M, int N, int K_dim)
{
    __shared__ bhalf_t s_dS[BM][BN + LDS_PAD];
    __shared__ bhalf_t s_K[BK][BN + LDS_PAD];

    using FragA = fragment<matrix_a, 16, 16, 16, bhalf_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, bhalf_t>;

    int block_m = blockIdx.y, block_k = blockIdx.x;
    int m_start = block_m * BM, k_start = block_k * BN;
    int tid = threadIdx.x;

    int warp_id = tid / 64;
    int warp_m = warp_id % 4;
    int warp_k = warp_id / 4;
    int w_m_start = m_start + warp_m * 16, w_k_start = k_start + warp_k * 16;

    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        int gm = m_start + m;
        if (gm < M && n < N) s_dS[m][n] = dS[gm * N + n];
    }
    __syncthreads();

    if (w_m_start < M && w_k_start < K_dim) {
        FragC acc;
        fill_fragment(acc, 0.0f);

        for (int n_start = 0; n_start < N; n_start += BK) {
            for (int i = tid; i < BK * BN; i += blockDim.x) {
                int k = i / BN, n = i % BN;
                int gn = n_start + n, gk = k_start + k;
                s_K[k][n] = (gn < N && gk < K_dim) ? K[gn * K_dim + gk] : bhalf_t(0.0f);
            }
            __syncthreads();

            for (int sub = 0; sub < 2; ++sub) {
                FragA a0, a1; FragB b0, b1;
                bhalf_t *a_ptr0 = reinterpret_cast<bhalf_t*>(&a0), *b_ptr0 = reinterpret_cast<bhalf_t*>(&b0);
                bhalf_t *a_ptr1 = reinterpret_cast<bhalf_t*>(&a1), *b_ptr1 = reinterpret_cast<bhalf_t*>(&b1);

                int n_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    int m = warp_m * 16 + i;
                    for (int j = 0; j < 16; ++j) {
                        int n = n_start + n_off + j;
                        bhalf_t val = (m < BM && n < BN) ? s_dS[m][n] : bhalf_t(0.0f);
                        a_ptr0[i*16 + j] = val;
                        a_ptr1[i*16 + j] = val;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        int k = warp_k * 16 + j;
                        bhalf_t val = (k < BK && (n_off + i) < BN) ? s_K[k][n_off + i] : bhalf_t(0.0f);
                        b_ptr0[j*16 + i] = val;
                        b_ptr1[j*16 + i] = val;
                    }
                }
                mma_sync(acc, a0, b0, acc);
                mma_sync(acc, a1, b1, acc);
            }
        }

        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (m < M && k < K_dim) dQ[m * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

int main() {
    const int M = 64, N = 64, K_dim = 32;
    bhalf_t dS[M * N], K[N * K_dim], dQ[M * K_dim];
    for (int i = 0; i < M * N; i++) dS[i] = (bhalf_t)(-0.5f);
    for (int i = 0; i < N * K_dim; i++) K[i] = (bhalf_t)1.0f;

    bhalf_t *d_dS, *d_K, *d_dQ;
    hipMalloc(&d_dS, M * N * sizeof(bhalf_t));
    hipMalloc(&d_K, N * K_dim * sizeof(bhalf_t));
    hipMalloc(&d_dQ, M * K_dim * sizeof(bhalf_t));
    hipMemcpy(d_dS, dS, M * N * sizeof(bhalf_t), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K, N * K_dim * sizeof(bhalf_t), hipMemcpyHostToDevice);

    dim3 grid((K_dim + BN - 1) / BN, (M + BM - 1) / BM);
    hipLaunchKernelGGL(dq_kernel, grid, dim3(BLOCK_SIZE), 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();

    hipMemcpy(dQ, d_dQ, M * K_dim * sizeof(bhalf_t), hipMemcpyDeviceToHost);
    printf("dQ[0..3]: %.1f %.1f %.1f %.1f\n", (float)dQ[0], (float)dQ[1], (float)dQ[2], (float)dQ[3]);
    printf("Expected: -32.0\n");

    hipFree(d_dS); hipFree(d_K); hipFree(d_dQ);
    return 0;
}

#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

constexpr int BM = 64, BN = 64, BK = 32, WARP_SIZE = 64, N_WAVES = 8, BLOCK_SIZE = 512, LDS_PAD = 8;

// dP = dO @ V^T
__global__ void dp_kernel(const bhalf_t* dO, const bhalf_t* V, bhalf_t* dP, int M, int N, int K_dim, int total_heads) {
    __shared__ bhalf_t s_dO[BM][BK+LDS_PAD], s_VT[BK][BN+LDS_PAD];
    using FragA = fragment<matrix_a, 16, 16, 16, bhalf_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, bhalf_t>;
    int block_m = blockIdx.y, block_n = blockIdx.x, head_idx = blockIdx.z;
    int m_start = block_m * BM, n_start = block_n * BN;
    int tid = threadIdx.x, warp_id = tid / WARP_SIZE, warp_m = warp_id / 4, warp_n = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_n_start = n_start + warp_n * 16;
    dO += head_idx * M * K_dim; V += head_idx * N * K_dim; dP += head_idx * M * N;
    if (w_m_start < M && w_n_start < N) {
        FragC acc; fill_fragment(acc, 0.0f);
        for (int k_block = 0; k_block < K_dim; k_block += BK) {
            for (int i = tid; i < BM * BK; i += blockDim.x) {
                int m = i / BK, k = i % BK;
                int gm = m_start + m, gk = k_block + k;
                s_dO[m][k] = (gm < M && gk < K_dim) ? dO[gm * K_dim + gk] : bhalf_t(0.0f);
            }
            for (int i = tid; i < BN * BK; i += blockDim.x) {
                int n = i / BK, k = i % BK;
                int gn = n_start + n, gk = k_block + k;
                s_VT[k][n] = (gn < N && gk < K_dim) ? V[gn * K_dim + gk] : bhalf_t(0.0f);
            }
            __syncthreads();
            for (int sub = 0; sub < 1; ++sub) {
                FragA a0, a1; FragB b0, b1;
                bhalf_t *a_ptr0 = reinterpret_cast<bhalf_t*>(&a0), *b_ptr0 = reinterpret_cast<bhalf_t*>(&b0);
                bhalf_t *a_ptr1 = reinterpret_cast<bhalf_t*>(&a1), *b_ptr1 = reinterpret_cast<bhalf_t*>(&b1);
                int k_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    int m = warp_m * 16 + i;
                    for (int j = 0; j < 16; ++j) {
                        a_ptr0[i*16 + j] = (m < BM && (k_off + j) < BK) ? s_dO[m][k_off + j] : bhalf_t(0.0f);
                        a_ptr1[i*16 + j] = (m < BM && (k_off + j) < BK) ? s_dO[m][k_off + j] : bhalf_t(0.0f);
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        int n = warp_n * 16 + j;
                        bhalf_t val = ((k_off + i) < BK && n < BN) ? s_VT[k_off + i][n] : bhalf_t(0.0f);
                        b_ptr0[j*16 + i] = val;
                        b_ptr1[j*16 + i] = val;
                    }
                }
                mma_sync(acc, a0, b0, acc);
                mma_sync(acc, a1, b1, acc);
            }
            __syncthreads();
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int n = w_n_start + j;
                if (m < M && n < N) dP[m * N + n] = out.x[i*16 + j];
            }
        }
    }
}

// Softmax backward
__global__ void softmax_bwd_kernel(const bhalf_t* P, const bhalf_t* dP, bhalf_t* dS, int M, int N, int total_heads) {
    __shared__ bhalf_t s_P[BM][BN+LDS_PAD], s_dP[BM][BN+LDS_PAD];
    __shared__ float s_rowsum[BM];
    int block_m = blockIdx.x, m_start = block_m * BM, head_idx = blockIdx.y, tid = threadIdx.x;
    P += head_idx * M * N; dP += head_idx * M * N; dS += head_idx * M * N;
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN, gm = m_start + m;
        if (gm < M && n < N) {
            s_P[m][n] = P[gm * N + n];
            s_dP[m][n] = dP[gm * N + n];
        }
    }
    __syncthreads();
    if (tid < BM) {
        int gm = m_start + tid;
        if (gm < M) {
            float sum = 0.0f;
            for (int n = 0; n < BN; ++n) sum += static_cast<float>(s_P[tid][n]) * static_cast<float>(s_dP[tid][n]);
            s_rowsum[tid] = sum;
        }
    }
    __syncthreads();
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN, gm = m_start + m;
        if (gm < M && n < N) {
            float p = static_cast<float>(s_P[m][n]), dp = static_cast<float>(s_dP[m][n]), rs = s_rowsum[m];
            dS[gm * N + n] = static_cast<bhalf_t>(p * (dp - rs));
        }
    }
}

// dQ = dS @ K
__global__ void dq_kernel(const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ, int M, int N, int K_dim, int total_heads) {
    __shared__ bhalf_t s_dS[BM][BN+LDS_PAD], s_K[BN][BK+LDS_PAD];
    using FragA = fragment<matrix_a, 16, 16, 16, bhalf_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, bhalf_t>;
    int block_m = blockIdx.y, block_k = blockIdx.x, head_idx = blockIdx.z;
    int m_start = block_m * BM, k_start = block_k * BN;
    int tid = threadIdx.x, warp_id = tid / WARP_SIZE, warp_m = warp_id / 4, warp_k = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_k_start = k_start + warp_k * 16;
    dS += head_idx * M * N; K += head_idx * N * K_dim; dQ += head_idx * M * K_dim;
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        if (m_start + m < M && n < N) s_dS[m][n] = dS[(m_start + m) * N + n];
    }
    __syncthreads();
    if (w_m_start < M && w_k_start < K_dim) {
        FragC acc; fill_fragment(acc, 0.0f);
        for (int n_start = 0; n_start < N; n_start += BK) {
            for (int i = tid; i < BN * BK; i += blockDim.x) {
                int n = i / BK, k = i % BK;
                int gn = n_start + n, gk = k_start + k;
                s_K[n][k] = (gn < N && gk < K_dim) ? K[gn * K_dim + gk] : bhalf_t(0.0f);
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
                        a_ptr0[i*16 + j] = (m < BM && (n_off + j) < BN) ? s_dS[m][n_off + j] : bhalf_t(0.0f);
                        a_ptr1[i*16 + j] = (m < BM && (n_off + j) < BN) ? s_dS[m][n_off + j] : bhalf_t(0.0f);
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        bhalf_t val = ((n_off + i) < BN && warp_k * 16 + j < BK) ? s_K[n_off + i][warp_k * 16 + j] : bhalf_t(0.0f);
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
                if (m < M && k < K_dim) dQ[m * K_dim + k] = out.x[i*16 + j];
            }
        }
    }
}

// dK = dS^T @ Q
__global__ void dk_kernel(const bhalf_t* dS, const bhalf_t* Q, bhalf_t* dK, int M, int N, int K_dim, int total_heads) {
    __shared__ bhalf_t s_dST[BN][BM+LDS_PAD], s_Q[BM][BK+LDS_PAD];
    using FragAT = fragment<matrix_a, 16, 16, 16, bhalf_t, col_major>;
    using FragB  = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC  = fragment<accumulator, 16, 16, 16, float>;
    using FragOut= fragment<accumulator, 16, 16, 16, bhalf_t>;
    int block_n = blockIdx.y, block_k = blockIdx.x, head_idx = blockIdx.z;
    int n_start = block_n * BM, k_start = block_k * BN;
    int tid = threadIdx.x, warp_id = tid / WARP_SIZE, warp_n = warp_id / 4, warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;
    dS += head_idx * M * N; Q += head_idx * M * K_dim; dK += head_idx * N * K_dim;
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        if (m < M && w_n_start + n < N) s_dST[n][m] = dS[m * N + (w_n_start + n)];
    }
    for (int i = tid; i < BM * BK; i += blockDim.x) {
        int m = i / BK, k = i % BK;
        if (m < M && w_k_start + k < K_dim) s_Q[m][k] = Q[m * K_dim + (w_k_start + k)];
    }
    __syncthreads();
    if (w_n_start < N && w_k_start < K_dim) {
        FragC acc; fill_fragment(acc, 0.0f);
        for (int m_start = 0; m_start < M; m_start += BK) {
            __syncthreads();
            for (int sub = 0; sub < 2; ++sub) {
                FragAT a0, a1; FragB b0, b1;
                bhalf_t *a_ptr0 = reinterpret_cast<bhalf_t*>(&a0), *b_ptr0 = reinterpret_cast<bhalf_t*>(&b0);
                bhalf_t *a_ptr1 = reinterpret_cast<bhalf_t*>(&a1), *b_ptr1 = reinterpret_cast<bhalf_t*>(&b1);
                int m_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        bhalf_t val_a = (m_start + m_off + j < BM && warp_n * 16 + i < BN) ? s_dST[warp_n * 16 + i][m_start + m_off + j] : bhalf_t(0.0f);
                        a_ptr0[i*16 + j] = val_a;
                        a_ptr1[i*16 + j] = val_a;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        bhalf_t val_b = (m_start + m_off + i < BM && warp_k * 16 + j < BK) ? s_Q[m_start + m_off + i][warp_k * 16 + j] : bhalf_t(0.0f);
                        b_ptr0[j*16 + i] = val_b;
                        b_ptr1[j*16 + i] = val_b;
                    }
                }
                mma_sync(acc, a0, b0, acc);
                mma_sync(acc, a1, b1, acc);
            }
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (n < N && k < K_dim) dK[n * K_dim + k] = out.x[i*16 + j];
            }
        }
    }
}

// dV = P^T @ dO
__global__ void dv_kernel(const bhalf_t* P, const bhalf_t* dO, bhalf_t* dV, int M, int N, int K_dim, int total_heads) {
    __shared__ bhalf_t s_PT[BN][BM+LDS_PAD], s_dO[BM][BK+LDS_PAD];
    using FragAT = fragment<matrix_a, 16, 16, 16, bhalf_t, col_major>;
    using FragB  = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC  = fragment<accumulator, 16, 16, 16, float>;
    using FragOut= fragment<accumulator, 16, 16, 16, bhalf_t>;
    int block_n = blockIdx.y, block_k = blockIdx.x, head_idx = blockIdx.z;
    int n_start = block_n * BM, k_start = block_k * BN;
    int tid = threadIdx.x, warp_id = tid / WARP_SIZE, warp_n = warp_id / 4, warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;
    P += head_idx * M * N; dO += head_idx * M * K_dim; dV += head_idx * N * K_dim;
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        if (m < M && w_n_start + n < N) s_PT[n][m] = P[m * N + (w_n_start + n)];
    }
    for (int i = tid; i < BM * BK; i += blockDim.x) {
        int m = i / BK, k = i % BK;
        if (m < M && w_k_start + k < K_dim) s_dO[m][k] = dO[m * K_dim + (w_k_start + k)];
    }
    __syncthreads();
    if (w_n_start < N && w_k_start < K_dim) {
        FragC acc; fill_fragment(acc, 0.0f);
        for (int m_start = 0; m_start < M; m_start += BK) {
            __syncthreads();
            for (int sub = 0; sub < 2; ++sub) {
                FragAT a0, a1; FragB b0, b1;
                bhalf_t *a_ptr0 = reinterpret_cast<bhalf_t*>(&a0), *b_ptr0 = reinterpret_cast<bhalf_t*>(&b0);
                bhalf_t *a_ptr1 = reinterpret_cast<bhalf_t*>(&a1), *b_ptr1 = reinterpret_cast<bhalf_t*>(&b1);
                int m_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        bhalf_t val_a = (m_start + m_off + j < BM && warp_n * 16 + i < BN) ? s_PT[warp_n * 16 + i][m_start + m_off + j] : bhalf_t(0.0f);
                        a_ptr0[i*16 + j] = val_a;
                        a_ptr1[i*16 + j] = val_a;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        bhalf_t val_b = (m_start + m_off + i < BM && warp_k * 16 + j < BK) ? s_dO[m_start + m_off + i][warp_k * 16 + j] : bhalf_t(0.0f);
                        b_ptr0[j*16 + i] = val_b;
                        b_ptr1[j*16 + i] = val_b;
                    }
                }
                mma_sync(acc, a0, b0, acc);
                mma_sync(acc, a1, b1, acc);
            }
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (n < N && k < K_dim) dV[n * K_dim + k] = out.x[i*16 + j];
            }
        }
    }
}

extern "C" {
    void fmha_bwd_full_py(bhalf_t* dQ, bhalf_t* dK, bhalf_t* dV,
        const bhalf_t* Q, const bhalf_t* K, const bhalf_t* V,
        const bhalf_t* P, const bhalf_t* dO, int M, int N, int K_dim, int total_heads)
    {
        bhalf_t *dP, *dS;
        size_t size_P = total_heads * M * N * sizeof(bhalf_t);
        size_t size_S = total_heads * M * N * sizeof(bhalf_t);
        hipMalloc(&dP, size_P); hipMemset(dP, 0, size_P);
        hipMalloc(&dS, size_S); hipMemset(dS, 0, size_S);
        
        hipLaunchKernelGGL(dp_kernel, dim3((N+BN-1)/BN, (M+BM-1)/BM, total_heads), dim3(BLOCK_SIZE), 0, 0,
            dO, V, dP, M, N, K_dim, total_heads);
        hipLaunchKernelGGL(softmax_bwd_kernel, dim3((M+BM-1)/BM, total_heads, 1), dim3(BLOCK_SIZE), 0, 0,
            P, dP, dS, M, N, total_heads);
        hipLaunchKernelGGL(dq_kernel, dim3((K_dim+BN-1)/BN, (M+BM-1)/BM, total_heads), dim3(BLOCK_SIZE), 0, 0,
            dS, K, dQ, M, N, K_dim, total_heads);
        hipLaunchKernelGGL(dk_kernel, dim3((K_dim+BN-1)/BN, (N+BM-1)/BM, total_heads), dim3(BLOCK_SIZE), 0, 0,
            dS, Q, dK, M, N, K_dim, total_heads);
        hipLaunchKernelGGL(dv_kernel, dim3((K_dim+BN-1)/BN, (N+BM-1)/BM, total_heads), dim3(BLOCK_SIZE), 0, 0,
            P, dO, dV, M, N, K_dim, total_heads);
        hipDeviceSynchronize();
        hipFree(dP); hipFree(dS);
    }
}

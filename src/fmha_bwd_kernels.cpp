#include "fmha_bwd_kernels.hpp"
#include <iostream>

using namespace rocwmma;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

// ----------------------------------------------------------------------------
// dP = dO @ V^T
// ----------------------------------------------------------------------------
__global__ void dp_kernel(
    const half_t* __restrict__ dO,
    const half_t* __restrict__ V,
    half_t* __restrict__ dP,
    int M, int N, int K_dim)
{
    __shared__ half_t s_dO[BM][BK];
    __shared__ half_t s_V[BN][BK];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_m = blockIdx.y, block_n = blockIdx.x;
    int m_start = block_m * BM, n_start = block_n * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_m = warp_id / 4, warp_n = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_n_start = n_start + warp_n * 16;
    
    for (int i = tid; i < BM * BK; i += blockDim.x) {
        int m = i / BK, k = i % BK;
        if (m_start + m < M && k < K_dim) s_dO[m][k] = dO[(m_start + m) * K_dim + k];
    }
    for (int i = tid; i < BN * BK; i += blockDim.x) {
        int n = i / BK, k = i % BK;
        if (n_start + n < N && k < K_dim) s_V[n][k] = V[(n_start + n) * K_dim + k];
    }
    __syncthreads();
    
    if (w_m_start < M && w_n_start < N) {
        FragC acc; fill_fragment(acc, 0.0f);
        for (int k_start = 0; k_start < K_dim; k_start += BK) {
            FragA a; FragB b;
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                for (int j = 0; j < 16; ++j) a.x[i*16+j] = (m < BM && k_start + j < BK) ? s_dO[m][k_start + j] : 0;
            }
            for (int i = 0; i < 16; ++i) {
                int n = warp_n * 16 + i;
                for (int j = 0; j < 16; ++j) b.x[i*16+j] = (n < BN && k_start + j < BK) ? s_V[n][k_start + j] : 0;
            }
            mma_sync(acc, a, b, acc);
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int n = w_n_start + j;
                if (m < M && n < N) dP[m * N + n] = out.x[i*16+j];
            }
        }
    }
}

void launch_dp_kernel(half_t* dP, const half_t* dO, const half_t* V,
                      int M, int N, int K_dim, hipStream_t stream) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);
    hipLaunchKernelGGL(dp_kernel, grid, block, 0, stream, dO, V, dP, M, N, K_dim);
}

// ----------------------------------------------------------------------------
// Softmax Backward
// ----------------------------------------------------------------------------
__global__ void softmax_bwd_kernel(
    const half_t* __restrict__ P,
    const half_t* __restrict__ dP,
    half_t* __restrict__ dS,
    int M, int N)
{
    __shared__ half_t s_P[BM][BN];
    __shared__ half_t s_dP[BM][BN];
    __shared__ float s_rowsum[BM];
    
    int block_m = blockIdx.x, m_start = block_m * BM;
    int tid = threadIdx.x;
    
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        int gm = m_start + m, gn = n;
        if (gm < M && gn < N) {
            s_P[m][n] = P[gm * N + gn];
            s_dP[m][n] = dP[gm * N + gn];
        }
    }
    __syncthreads();
    
    if (tid < BM) {
        int gm = m_start + tid;
        if (gm < M) {
            float sum = 0.0f;
            for (int n = 0; n < BN; n++) sum += (float)s_P[tid][n] * (float)s_dP[tid][n];
            s_rowsum[tid] = sum;
        }
    }
    __syncthreads();
    
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        int gm = m_start + m, gn = n;
        if (gm < M && gn < N) {
            float p = s_P[m][n], dp = s_dP[m][n], rs = s_rowsum[m];
            dS[gm * N + gn] = (half_t)(p * (dp - rs));
        }
    }
}

void launch_softmax_bwd_kernel(half_t* dS, const half_t* P, const half_t* dP,
                               int M, int N, hipStream_t stream) {
    dim3 grid((M + BM - 1) / BM);
    dim3 block(256);
    hipLaunchKernelGGL(softmax_bwd_kernel, grid, block, 0, stream, P, dP, dS, M, N);
}

// ----------------------------------------------------------------------------
// dQ = dS @ K
// ----------------------------------------------------------------------------
__global__ void dq_kernel(
    const half_t* __restrict__ dS,
    const half_t* __restrict__ K,
    half_t* __restrict__ dQ,
    int M, int N, int K_dim)
{
    __shared__ half_t s_dS[BM][BN];
    __shared__ half_t s_K[BN][BK];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_m = blockIdx.y, block_k = blockIdx.x;
    int m_start = block_m * BM, k_start = block_k * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_m = warp_id / 4, warp_k = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_k_start = k_start + warp_k * 16;
    
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        if (m_start + m < M && n < N) s_dS[m][n] = dS[(m_start + m) * N + n];
    }
    for (int i = tid; i < BN * BK; i += blockDim.x) {
        int n = i / BK, k = i % BK;
        if (n < N && k_start + k < K_dim) s_K[n][k] = K[n * K_dim + (k_start + k)];
    }
    __syncthreads();
    
    if (w_m_start < M && w_k_start < K_dim) {
        FragC acc; fill_fragment(acc, 0.0f);
        for (int n_start = 0; n_start < N; n_start += BK) {
            FragA a; FragB b;
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                for (int j = 0; j < 16; ++j) a.x[i*16+j] = (m < BM && n_start + j < BN) ? s_dS[m][n_start + j] : 0;
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) b.x[i*16+j] = (n_start + i < BN && warp_k * 16 + j < BK) ? s_K[n_start + i][warp_k * 16 + j] : 0;
            }
            mma_sync(acc, a, b, acc);
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = (half_t)acc.x[i];
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (m < M && k < K_dim) dQ[m * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

void launch_dq_kernel(half_t* dQ, const half_t* dS, const half_t* K,
                      int M, int N, int K_dim, hipStream_t stream) {
    dim3 grid((K_dim + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);
    hipLaunchKernelGGL(dq_kernel, grid, block, 0, stream, dS, K, dQ, M, N, K_dim);
}

// ----------------------------------------------------------------------------
// dK = dS^T @ Q
// ----------------------------------------------------------------------------
__global__ void dk_kernel(
    const half_t* __restrict__ dS,
    const half_t* __restrict__ Q,
    half_t* __restrict__ dK,
    int M, int N, int K_dim)
{
    __shared__ half_t s_dS[BN][BM];
    __shared__ half_t s_Q[BM][BK];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, col_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_n = blockIdx.y, block_k = blockIdx.x;
    int n_start = block_n * BN, k_start = block_k * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_n = warp_id / 4, warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;
    
    for (int i = tid; i < BN * BM; i += blockDim.x) {
        int n = i / BM, m = i % BM;
        if (n_start + n < N && m < M) s_dS[n][m] = dS[m * N + (n_start + n)];
    }
    for (int i = tid; i < BM * BK; i += blockDim.x) {
        int m = i / BK, k = i % BK;
        if (m < M && k_start + k < K_dim) s_Q[m][k] = Q[m * K_dim + (k_start + k)];
    }
    __syncthreads();
    
    if (w_n_start < N && w_k_start < K_dim) {
        FragC acc; fill_fragment(acc, 0.0f);
        for (int m_start = 0; m_start < M; m_start += BK) {
            FragA a; FragB b;
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) a.x[i*16+j] = (m_start + j < BM && warp_n * 16 + i < BN) ? s_dS[warp_n * 16 + i][m_start + j] : 0;
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) b.x[i*16+j] = (m_start + i < BM && warp_k * 16 + j < BK) ? s_Q[m_start + i][warp_k * 16 + j] : 0;
            }
            mma_sync(acc, a, b, acc);
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = (half_t)acc.x[i];
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (n < N && k < K_dim) dK[n * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

void launch_dk_kernel(half_t* dK, const half_t* dS, const half_t* Q,
                      int M, int N, int K_dim, hipStream_t stream) {
    dim3 grid((K_dim + BN - 1) / BN, (N + BM - 1) / BM);
    dim3 block(128);
    hipLaunchKernelGGL(dk_kernel, grid, block, 0, stream, dS, Q, dK, M, N, K_dim);
}

// ----------------------------------------------------------------------------
// dV = P^T @ dO
// ----------------------------------------------------------------------------
__global__ void dv_kernel(
    const half_t* __restrict__ P,
    const half_t* __restrict__ dO,
    half_t* __restrict__ dV,
    int M, int N, int K_dim)
{
    __shared__ half_t s_P[BN][BM];
    __shared__ half_t s_dO[BM][BK];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, col_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_n = blockIdx.y, block_k = blockIdx.x;
    int n_start = block_n * BN, k_start = block_k * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_n = warp_id / 4, warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;
    
    for (int i = tid; i < BN * BM; i += blockDim.x) {
        int n = i / BM, m = i % BM;
        if (n_start + n < N && m < M) s_P[n][m] = P[m * N + (n_start + n)];
    }
    for (int i = tid; i < BM * BK; i += blockDim.x) {
        int m = i / BK, k = i % BK;
        if (m < M && k_start + k < K_dim) s_dO[m][k] = dO[m * K_dim + (k_start + k)];
    }
    __syncthreads();
    
    if (w_n_start < N && w_k_start < K_dim) {
        FragC acc; fill_fragment(acc, 0.0f);
        for (int m_start = 0; m_start < M; m_start += BK) {
            FragA a; FragB b;
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) a.x[i*16+j] = (m_start + j < BM && warp_n * 16 + i < BN) ? s_P[warp_n * 16 + i][m_start + j] : 0;
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) b.x[i*16+j] = (m_start + i < BM && warp_k * 16 + j < BK) ? s_dO[m_start + i][warp_k * 16 + j] : 0;
            }
            mma_sync(acc, a, b, acc);
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = (half_t)acc.x[i];
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (n < N && k < K_dim) dV[n * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

void launch_dv_kernel(half_t* dV, const half_t* P, const half_t* dO,
                      int M, int N, int K_dim, hipStream_t stream) {
    dim3 grid((K_dim + BN - 1) / BN, (N + BM - 1) / BM);
    dim3 block(128);
    hipLaunchKernelGGL(dv_kernel, grid, block, 0, stream, P, dO, dV, M, N, K_dim);
}

// ----------------------------------------------------------------------------
// Полный backward pass
// ----------------------------------------------------------------------------
void fmha_bwd_full(
    half_t* dQ, half_t* dK, half_t* dV,
    const half_t* Q, const half_t* K, const half_t* V,
    const half_t* P, const half_t* dO,
    int M, int N, int K_dim,
    hipStream_t stream)
{
    // Выделяем временный буфер для dP и dS
    half_t *dP, *dS;
    size_t size_P = M * N * sizeof(half_t);
    hipMalloc(&dP, size_P);
    hipMalloc(&dS, size_P);
    
    // 1. dP = dO @ V^T
    launch_dp_kernel(dP, dO, V, M, N, K_dim, stream);
    
    // 2. dS = softmax_backward(P, dP)
    launch_softmax_bwd_kernel(dS, P, dP, M, N, stream);
    
    // 3. dQ = dS @ K
    launch_dq_kernel(dQ, dS, K, M, N, K_dim, stream);
    
    // 4. dK = dS^T @ Q
    launch_dk_kernel(dK, dS, Q, M, N, K_dim, stream);
    
    // 5. dV = P^T @ dO
    launch_dv_kernel(dV, P, dO, M, N, K_dim, stream);
    
    hipFree(dP);
    hipFree(dS);
}

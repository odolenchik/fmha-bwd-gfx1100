#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

// ----------------------------------------------------------------------------
// dP = dO @ V^T (оптимизированное WMMA + LDS)
// ----------------------------------------------------------------------------
constexpr int BM_dP = 64, BN_dP = 64, BK_dP = 32;

__global__ void dp_kernel(
    const half_t* __restrict__ dO,
    const half_t* __restrict__ V,
    half_t* __restrict__ dP,
    int M, int N, int K_dim)
{
    __shared__ half_t s_dO[BM_dP][BK_dP];
    __shared__ half_t s_VT[BK_dP][BN_dP];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_m = blockIdx.y, block_n = blockIdx.x;
    int m_start = block_m * BM_dP, n_start = block_n * BN_dP;
    int tid = threadIdx.x;
    int warp_id = tid / 64, warp_m = warp_id / 4, warp_n = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_n_start = n_start + warp_n * 16;
    
    if (w_m_start < M && w_n_start < N) {
        FragC acc[2];
        fill_fragment(acc[0], 0.0f);
        fill_fragment(acc[1], 0.0f);
        
        for (int k_block = 0; k_block < K_dim; k_block += BK_dP) {
            for (int i = tid; i < BM_dP * BK_dP; i += blockDim.x) {
                int m = i / BK_dP, k = i % BK_dP;
                int gm = m_start + m, gk = k_block + k;
                s_dO[m][k] = (gm < M && gk < K_dim) ? dO[gm * K_dim + gk] : half_t(0);
            }
            for (int i = tid; i < BN_dP * BK_dP; i += blockDim.x) {
                int n = i / BK_dP, k = i % BK_dP;
                int gn = n_start + n, gk = k_block + k;
                s_VT[k][n] = (gn < N && gk < K_dim) ? V[gn * K_dim + gk] : half_t(0);
            }
            __syncthreads();
            
            for (int sub = 0; sub < 2; ++sub) {
                FragA a0, a1; FragB b0, b1;
                half_t* a_ptr0 = reinterpret_cast<half_t*>(&a0);
                half_t* b_ptr0 = reinterpret_cast<half_t*>(&b0);
                half_t* a_ptr1 = reinterpret_cast<half_t*>(&a1);
                half_t* b_ptr1 = reinterpret_cast<half_t*>(&b1);
                
                int k_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    int m = warp_m * 16 + i;
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m < BM_dP && (k_off + j) < BK_dP) ? s_dO[m][k_off + j] : half_t(0);
                        a_ptr0[i*16 + j] = val;
                        a_ptr1[i*16 + j] = val;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        int n = warp_n * 16 + j;
                        half_t val = ((k_off + i) < BK_dP && n < BN_dP) ? s_VT[k_off + i][n] : half_t(0);
                        b_ptr0[i*16 + j] = val;
                        b_ptr1[i*16 + j] = val;
                    }
                }
                mma_sync(acc[0], a0, b0, acc[0]);
                mma_sync(acc[1], a1, b1, acc[1]);
            }
            __syncthreads();
        }
        
        for (int i = 0; i < acc[0].num_elements; ++i)
            acc[0].x[i] += acc[1].x[i];
        FragOut out;
        for (int i = 0; i < acc[0].num_elements; ++i) out.x[i] = static_cast<half_t>(acc[0].x[i]);
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int n = w_n_start + j;
                if (m < M && n < N) dP[m * N + n] = out.x[i*16+j];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Softmax backward (fused LDS)
// ----------------------------------------------------------------------------
constexpr int BM_sm = 64, BN_sm = 64;

__global__ void softmax_bwd_kernel(
    const half_t* __restrict__ P,
    const half_t* __restrict__ dP,
    half_t* __restrict__ dS,
    int M, int N)
{
    __shared__ half_t s_P[BM_sm][BN_sm];
    __shared__ half_t s_dP[BM_sm][BN_sm];
    __shared__ float s_rowsum[BM_sm];
    
    int block_m = blockIdx.x, m_start = block_m * BM_sm;
    int tid = threadIdx.x;
    
    for (int i = tid; i < BM_sm * BN_sm; i += blockDim.x) {
        int m = i / BN_sm, n = i % BN_sm;
        int gm = m_start + m, gn = n;
        if (gm < M && gn < N) {
            s_P[m][n] = P[gm * N + gn];
            s_dP[m][n] = dP[gm * N + gn];
        }
    }
    __syncthreads();
    
    if (tid < BM_sm) {
        int gm = m_start + tid;
        if (gm < M) {
            float sum = 0.0f;
            for (int n = 0; n < BN_sm; ++n)
                sum += (float)s_P[tid][n] * (float)s_dP[tid][n];
            s_rowsum[tid] = sum;
        }
    }
    __syncthreads();
    
    for (int i = tid; i < BM_sm * BN_sm; i += blockDim.x) {
        int m = i / BN_sm, n = i % BN_sm;
        int gm = m_start + m, gn = n;
        if (gm < M && gn < N) {
            float p = s_P[m][n], dp = s_dP[m][n], rs = s_rowsum[m];
            dS[gm * N + gn] = (half_t)(p * (dp - rs));
        }
    }
}

// ----------------------------------------------------------------------------
// dQ = dS @ K (оптимизированное)
// ----------------------------------------------------------------------------
constexpr int BM_dQ = 64, BN_dQ = 64, BK_dQ = 32;

__global__ void dq_kernel(
    const half_t* __restrict__ dS,
    const half_t* __restrict__ K,
    half_t* __restrict__ dQ,
    int M, int N, int K_dim)
{
    __shared__ half_t s_dS[BM_dQ][BN_dQ];
    __shared__ half_t s_K[BN_dQ][BK_dQ];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_m = blockIdx.y, block_k = blockIdx.x;
    int m_start = block_m * BM_dQ, k_start = block_k * BN_dQ;
    int tid = threadIdx.x;
    int warp_id = tid / 64, warp_m = warp_id / 4, warp_k = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_k_start = k_start + warp_k * 16;
    
    for (int i = tid; i < BM_dQ * BN_dQ; i += blockDim.x) {
        int m = i / BN_dQ, n = i % BN_dQ;
        int gm = m_start + m;
        if (gm < M && n < N) s_dS[m][n] = dS[gm * N + n];
    }
    __syncthreads();
    
    if (w_m_start < M && w_k_start < K_dim) {
        FragC acc[2];
        fill_fragment(acc[0], 0.0f);
        fill_fragment(acc[1], 0.0f);
        
        for (int n_start = 0; n_start < N; n_start += BK_dQ) {
            for (int i = tid; i < BN_dQ * BK_dQ; i += blockDim.x) {
                int n = i / BK_dQ, k = i % BK_dQ;
                int gn = n_start + n, gk = w_k_start + k;
                s_K[n][k] = (gn < N && gk < K_dim) ? K[gn * K_dim + gk] : half_t(0);
            }
            __syncthreads();
            
            for (int sub = 0; sub < 2; ++sub) {
                FragA a0, a1; FragB b0, b1;
                half_t* a_ptr0 = reinterpret_cast<half_t*>(&a0);
                half_t* b_ptr0 = reinterpret_cast<half_t*>(&b0);
                half_t* a_ptr1 = reinterpret_cast<half_t*>(&a1);
                half_t* b_ptr1 = reinterpret_cast<half_t*>(&b1);
                
                int n_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    int m = warp_m * 16 + i;
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m < BM_dQ && (n_off + j) < BN_dQ) ? s_dS[m][n_off + j] : half_t(0);
                        a_ptr0[i*16 + j] = val;
                        a_ptr1[i*16 + j] = val;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = ((n_off + i) < BN_dQ && warp_k * 16 + j < BK_dQ) ? s_K[n_off + i][warp_k * 16 + j] : half_t(0);
                        b_ptr0[i*16 + j] = val;
                        b_ptr1[i*16 + j] = val;
                    }
                }
                mma_sync(acc[0], a0, b0, acc[0]);
                mma_sync(acc[1], a1, b1, acc[1]);
            }
            __syncthreads();
        }
        
        for (int i = 0; i < acc[0].num_elements; ++i)
            acc[0].x[i] += acc[1].x[i];
        FragOut out;
        for (int i = 0; i < acc[0].num_elements; ++i) out.x[i] = static_cast<half_t>(acc[0].x[i]);
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (m < M && k < K_dim) dQ[m * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// dK = dS^T @ Q (оптимизированное)
// ----------------------------------------------------------------------------
__global__ void dk_kernel(
    const half_t* __restrict__ dS,
    const half_t* __restrict__ Q,
    half_t* __restrict__ dK,
    int M, int N, int K_dim)
{
    __shared__ half_t s_dST[BN_dQ][BM_dQ];
    __shared__ half_t s_Q[BM_dQ][BK_dQ];
    
    using FragAT = fragment<matrix_a, 16, 16, 16, half_t, col_major>;
    using FragB  = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC  = fragment<accumulator, 16, 16, 16, float>;
    using FragOut= fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_n = blockIdx.y, block_k = blockIdx.x;
    int n_start = block_n * BN_dQ, k_start = block_k * BN_dQ;
    int tid = threadIdx.x;
    int warp_id = tid / 64, warp_n = warp_id / 4, warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;
    
    for (int i = tid; i < BM_dQ * BN_dQ; i += blockDim.x) {
        int m = i / BN_dQ, n = i % BN_dQ;
        if (m < M && w_n_start + n < N) s_dST[n][m] = dS[m * N + (w_n_start + n)];
    }
    for (int i = tid; i < BM_dQ * BK_dQ; i += blockDim.x) {
        int m = i / BK_dQ, k = i % BK_dQ;
        if (m < M && w_k_start + k < K_dim) s_Q[m][k] = Q[m * K_dim + (w_k_start + k)];
    }
    __syncthreads();
    
    if (w_n_start < N && w_k_start < K_dim) {
        FragC acc[2];
        fill_fragment(acc[0], 0.0f);
        fill_fragment(acc[1], 0.0f);
        
        for (int m_start = 0; m_start < M; m_start += BK_dQ) {
            __syncthreads();
            for (int sub = 0; sub < 2; ++sub) {
                FragAT a0, a1; FragB b0, b1;
                half_t* a_ptr0 = reinterpret_cast<half_t*>(&a0);
                half_t* b_ptr0 = reinterpret_cast<half_t*>(&b0);
                half_t* a_ptr1 = reinterpret_cast<half_t*>(&a1);
                half_t* b_ptr1 = reinterpret_cast<half_t*>(&b1);
                
                int m_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m_start + m_off + j < BM_dQ && warp_n * 16 + i < BN_dQ) ? s_dST[warp_n * 16 + i][m_start + m_off + j] : half_t(0);
                        a_ptr0[i*16 + j] = val;
                        a_ptr1[i*16 + j] = val;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m_start + m_off + i < BM_dQ && warp_k * 16 + j < BK_dQ) ? s_Q[m_start + m_off + i][warp_k * 16 + j] : half_t(0);
                        b_ptr0[i*16 + j] = val;
                        b_ptr1[i*16 + j] = val;
                    }
                }
                mma_sync(acc[0], a0, b0, acc[0]);
                mma_sync(acc[1], a1, b1, acc[1]);
            }
        }
        
        for (int i = 0; i < acc[0].num_elements; ++i)
            acc[0].x[i] += acc[1].x[i];
        FragOut out;
        for (int i = 0; i < acc[0].num_elements; ++i) out.x[i] = static_cast<half_t>(acc[0].x[i]);
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (n < N && k < K_dim) dK[n * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// dV = P^T @ dO (оптимизированное)
// ----------------------------------------------------------------------------
__global__ void dv_kernel(
    const half_t* __restrict__ P,
    const half_t* __restrict__ dO,
    half_t* __restrict__ dV,
    int M, int N, int K_dim)
{
    __shared__ half_t s_PT[BN_dQ][BM_dQ];
    __shared__ half_t s_dO[BM_dQ][BK_dQ];
    
    using FragAT = fragment<matrix_a, 16, 16, 16, half_t, col_major>;
    using FragB  = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC  = fragment<accumulator, 16, 16, 16, float>;
    using FragOut= fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_n = blockIdx.y, block_k = blockIdx.x;
    int n_start = block_n * BN_dQ, k_start = block_k * BN_dQ;
    int tid = threadIdx.x;
    int warp_id = tid / 64, warp_n = warp_id / 4, warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;
    
    for (int i = tid; i < BM_dQ * BN_dQ; i += blockDim.x) {
        int m = i / BN_dQ, n = i % BN_dQ;
        if (m < M && w_n_start + n < N) s_PT[n][m] = P[m * N + (w_n_start + n)];
    }
    for (int i = tid; i < BM_dQ * BK_dQ; i += blockDim.x) {
        int m = i / BK_dQ, k = i % BK_dQ;
        if (m < M && w_k_start + k < K_dim) s_dO[m][k] = dO[m * K_dim + (w_k_start + k)];
    }
    __syncthreads();
    
    if (w_n_start < N && w_k_start < K_dim) {
        FragC acc[2];
        fill_fragment(acc[0], 0.0f);
        fill_fragment(acc[1], 0.0f);
        
        for (int m_start = 0; m_start < M; m_start += BK_dQ) {
            __syncthreads();
            for (int sub = 0; sub < 2; ++sub) {
                FragAT a0, a1; FragB b0, b1;
                half_t* a_ptr0 = reinterpret_cast<half_t*>(&a0);
                half_t* b_ptr0 = reinterpret_cast<half_t*>(&b0);
                half_t* a_ptr1 = reinterpret_cast<half_t*>(&a1);
                half_t* b_ptr1 = reinterpret_cast<half_t*>(&b1);
                
                int m_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m_start + m_off + j < BM_dQ && warp_n * 16 + i < BN_dQ) ? s_PT[warp_n * 16 + i][m_start + m_off + j] : half_t(0);
                        a_ptr0[i*16 + j] = val;
                        a_ptr1[i*16 + j] = val;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m_start + m_off + i < BM_dQ && warp_k * 16 + j < BK_dQ) ? s_dO[m_start + m_off + i][warp_k * 16 + j] : half_t(0);
                        b_ptr0[i*16 + j] = val;
                        b_ptr1[i*16 + j] = val;
                    }
                }
                mma_sync(acc[0], a0, b0, acc[0]);
                mma_sync(acc[1], a1, b1, acc[1]);
            }
        }
        
        for (int i = 0; i < acc[0].num_elements; ++i)
            acc[0].x[i] += acc[1].x[i];
        FragOut out;
        for (int i = 0; i < acc[0].num_elements; ++i) out.x[i] = static_cast<half_t>(acc[0].x[i]);
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (n < N && k < K_dim) dV[n * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main() {
    constexpr int M = 1024, N = 1024, K_dim = 64;
    std::cout << "\n=== ALL 5 SEPARATE OPTIMIZED KERNELS ===\n";

    size_t size_Q = M*K_dim, size_K = N*K_dim, size_V = N*K_dim, size_P = M*N, size_dO = M*K_dim;
    size_t size_dQ = M*K_dim, size_dK = N*K_dim, size_dV = N*K_dim;

    std::vector<half_t> h_Q(size_Q), h_K(size_K), h_V(size_V), h_P(size_P), h_dO(size_dO);
    std::vector<half_t> h_dP(size_P), h_dS(size_P);
    std::vector<half_t> h_dQ_gpu(size_dQ), h_dK_gpu(size_dK), h_dV_gpu(size_dV);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.0f);
    for (auto& v : h_Q) v = (half_t)dist(gen);
    for (auto& v : h_K) v = (half_t)dist(gen);
    for (auto& v : h_V) v = (half_t)dist(gen);
    for (auto& v : h_dO) v = (half_t)dist(gen);
    for (auto& v : h_P) v = (half_t)dist_pos(gen);

    for (int m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) sum += (float)h_P[m * N + n];
        for (int n = 0; n < N; ++n) h_P[m * N + n] = (half_t)((float)h_P[m * N + n] / sum);
    }

    half_t *d_Q, *d_K, *d_V, *d_P, *d_dO, *d_dP, *d_dS, *d_dQ, *d_dK, *d_dV;
    hipMalloc(&d_Q, size_Q*sizeof(half_t)); hipMalloc(&d_K, size_K*sizeof(half_t));
    hipMalloc(&d_V, size_V*sizeof(half_t)); hipMalloc(&d_P, size_P*sizeof(half_t));
    hipMalloc(&d_dO, size_dO*sizeof(half_t));
    hipMalloc(&d_dP, size_P*sizeof(half_t)); hipMalloc(&d_dS, size_P*sizeof(half_t));
    hipMalloc(&d_dQ, size_dQ*sizeof(half_t)); hipMalloc(&d_dK, size_dK*sizeof(half_t));
    hipMalloc(&d_dV, size_dV*sizeof(half_t));

    hipMemcpy(d_Q, h_Q.data(), size_Q*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K.data(), size_K*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), size_V*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_P, h_P.data(), size_P*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO.data(), size_dO*sizeof(half_t), hipMemcpyHostToDevice);

    // Warmup
    dim3 grid_dP((N + BN_dP - 1) / BN_dP, (M + BM_dP - 1) / BM_dP);
    dim3 grid_softmax((M + BM_sm - 1) / BM_sm);
    dim3 grid_dQ((K_dim + BN_dQ - 1) / BN_dQ, (M + BM_dQ - 1) / BM_dQ);
    dim3 grid_dK((K_dim + BN_dQ - 1) / BN_dQ, (N + BM_dQ - 1) / BM_dQ);
    dim3 block_dP(512), block_softmax(256), block_dQ(512);

    hipLaunchKernelGGL(dp_kernel, grid_dP, block_dP, 0, 0, d_dO, d_V, d_dP, M, N, K_dim);
    hipLaunchKernelGGL(softmax_bwd_kernel, grid_softmax, block_softmax, 0, 0, d_P, d_dP, d_dS, M, N);
    hipLaunchKernelGGL(dq_kernel, grid_dQ, block_dQ, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipLaunchKernelGGL(dk_kernel, grid_dK, block_dQ, 0, 0, d_dS, d_Q, d_dK, M, N, K_dim);
    hipLaunchKernelGGL(dv_kernel, grid_dK, block_dQ, 0, 0, d_P, d_dO, d_dV, M, N, K_dim);
    hipDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    hipLaunchKernelGGL(dp_kernel, grid_dP, block_dP, 0, 0, d_dO, d_V, d_dP, M, N, K_dim);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    hipLaunchKernelGGL(softmax_bwd_kernel, grid_softmax, block_softmax, 0, 0, d_P, d_dP, d_dS, M, N);
    hipDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    hipLaunchKernelGGL(dq_kernel, grid_dQ, block_dQ, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    
    hipLaunchKernelGGL(dk_kernel, grid_dK, block_dQ, 0, 0, d_dS, d_Q, d_dK, M, N, K_dim);
    hipDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    
    hipLaunchKernelGGL(dv_kernel, grid_dK, block_dQ, 0, 0, d_P, d_dO, d_dV, M, N, K_dim);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto time_dP = std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count();
    auto time_sm = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto time_dQ = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    auto time_dK = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    auto time_dV = std::chrono::duration_cast<std::chrono::microseconds>(end - t4).count();
    auto time_total = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    hipMemcpy(h_dQ_gpu.data(), d_dQ, size_dQ*sizeof(half_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_dK_gpu.data(), d_dK, size_dK*sizeof(half_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_dV_gpu.data(), d_dV, size_dV*sizeof(half_t), hipMemcpyDeviceToHost);

    bool has_nan = false;
    for (size_t i = 0; i < size_dQ; ++i) if (!std::isfinite((float)h_dQ_gpu[i])) { has_nan = true; break; }
    for (size_t i = 0; i < size_dK; ++i) if (!std::isfinite((float)h_dK_gpu[i])) { has_nan = true; break; }
    for (size_t i = 0; i < size_dV; ++i) if (!std::isfinite((float)h_dV_gpu[i])) { has_nan = true; break; }

    std::cout << "dP:      " << time_dP << " us\n";
    std::cout << "Softmax: " << time_sm << " us\n";
    std::cout << "dQ:      " << time_dQ << " us\n";
    std::cout << "dK:      " << time_dK << " us\n";
    std::cout << "dV:      " << time_dV << " us\n";
    std::cout << "-------------------\n";
    std::cout << "Total:   " << time_total << " us\n";
    std::cout << "NaN check: " << (has_nan ? "FAIL" : "PASS") << "\n";

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_P); hipFree(d_dO);
    hipFree(d_dP); hipFree(d_dS); hipFree(d_dQ); hipFree(d_dK); hipFree(d_dV);
    return 0;
}

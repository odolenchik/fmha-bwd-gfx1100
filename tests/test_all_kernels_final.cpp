#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

// ----------------------------------------------------------------------------
// dQ = dS @ K
// ----------------------------------------------------------------------------
__global__ void dq_kernel(const half_t* dS, const half_t* K, half_t* dQ, int M, int N, int K_dim) {
    __shared__ half_t s_dS[BM][BK];
    __shared__ half_t s_K[BK][BN];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_m = blockIdx.y, block_n = blockIdx.x;
    int m_start = block_m * BM, n_start = block_n * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_m = warp_id / 4, warp_n = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_n_start = n_start + warp_n * 16;
    
    FragC acc; fill_fragment(acc, 0.0f);
    
    for (int k_start = 0; k_start < N; k_start += BK) {
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int k = i / BN, n = n_start + (i % BN);
            if (k_start + k < N && n < K_dim) s_K[k][i % BN] = K[(k_start + k) * K_dim + n];
        }
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int m = i / BK, k = i % BK;
            if (m_start + m < M && k_start + k < N) s_dS[m][k] = dS[(m_start + m) * N + (k_start + k)];
        }
        __syncthreads();
        
        if (w_m_start < M && w_n_start < K_dim) {
            FragA a; FragB b;
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                for (int j = 0; j < 16; ++j) a.x[i*16+j] = s_dS[m][j];
            }
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) b.x[i*16+j] = s_K[i][warp_n * 16 + j];
            mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (w_m_start < M && w_n_start < K_dim) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        store_matrix_sync(dQ + w_m_start * K_dim + w_n_start, out, K_dim, mem_row_major);
    }
}

// ----------------------------------------------------------------------------
// dK = dS^T @ Q
// ----------------------------------------------------------------------------
__global__ void dk_kernel(const half_t* dS, const half_t* Q, half_t* dK, int M, int N, int K_dim) {
    __shared__ half_t s_dS[BM][BK];
    __shared__ half_t s_Q[BK][BN];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, col_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_n = blockIdx.y, block_k = blockIdx.x;
    int n_start = block_n * BM, k_start = block_k * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_n = warp_id / 4, warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;
    
    FragC acc; fill_fragment(acc, 0.0f);
    
    for (int m_start = 0; m_start < M; m_start += BK) {
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int m = i / BN, k = k_start + (i % BN);
            if (m_start + m < M && k < K_dim) s_Q[m][i % BN] = Q[(m_start + m) * K_dim + k];
        }
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int n = i / BK, m = i % BK;
            if (n_start + n < N && m_start + m < M) s_dS[n][m] = dS[(m_start + m) * N + (n_start + n)];
        }
        __syncthreads();
        
        if (w_n_start < N && w_k_start < K_dim) {
            FragA a; FragB b;
            for (int i = 0; i < 16; ++i) {
                int n = warp_n * 16 + i;
                for (int j = 0; j < 16; ++j) a.x[i*16+j] = s_dS[n][j];
            }
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) b.x[i*16+j] = s_Q[i][warp_k * 16 + j];
            mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (w_n_start < N && w_k_start < K_dim) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        store_matrix_sync(dK + w_n_start * K_dim + w_k_start, out, K_dim, mem_row_major);
    }
}

// ----------------------------------------------------------------------------
// dV = P^T @ dO
// ----------------------------------------------------------------------------
__global__ void dv_kernel(const half_t* P, const half_t* dO, half_t* dV, int M, int N, int K_dim) {
    __shared__ half_t s_P[BM][BK];
    __shared__ half_t s_dO[BK][BN];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, col_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_n = blockIdx.y, block_k = blockIdx.x;
    int n_start = block_n * BM, k_start = block_k * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_n = warp_id / 4, warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;
    
    FragC acc; fill_fragment(acc, 0.0f);
    
    for (int m_start = 0; m_start < M; m_start += BK) {
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int m = i / BN, k = k_start + (i % BN);
            if (m_start + m < M && k < K_dim) s_dO[m][i % BN] = dO[(m_start + m) * K_dim + k];
        }
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int n = i / BK, m = i % BK;
            if (n_start + n < N && m_start + m < M) s_P[n][m] = P[(m_start + m) * N + (n_start + n)];
        }
        __syncthreads();
        
        if (w_n_start < N && w_k_start < K_dim) {
            FragA a; FragB b;
            for (int i = 0; i < 16; ++i) {
                int n = warp_n * 16 + i;
                for (int j = 0; j < 16; ++j) a.x[i*16+j] = s_P[n][j];
            }
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) b.x[i*16+j] = s_dO[i][warp_k * 16 + j];
            mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (w_n_start < N && w_k_start < K_dim) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        store_matrix_sync(dV + w_n_start * K_dim + w_k_start, out, K_dim, mem_row_major);
    }
}

// ----------------------------------------------------------------------------
// dP = dO @ V^T
// ----------------------------------------------------------------------------
__global__ void dp_kernel(const half_t* dO, const half_t* V, half_t* dP, int M, int N, int K_dim) {
    __shared__ half_t s_dO[BM][BK];
    __shared__ half_t s_V[BK][BN];
    
    // dP = dO @ V^T -> A = dO (row_major), B = V^T (col_major)
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_m = blockIdx.y, block_n = blockIdx.x;
    int m_start = block_m * BM, n_start = block_n * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_m = warp_id / 4, warp_n = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_n_start = n_start + warp_n * 16;
    
    FragC acc; fill_fragment(acc, 0.0f);
    
    for (int k_start = 0; k_start < K_dim; k_start += BK) {
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int k = i / BN, n = n_start + (i % BN);
            if (k_start + k < K_dim && n < N) s_V[k][i % BN] = V[n * K_dim + (k_start + k)];
        }
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int m = i / BK, k = i % BK;
            if (m_start + m < M && k_start + k < K_dim) s_dO[m][k] = dO[(m_start + m) * K_dim + (k_start + k)];
        }
        __syncthreads();
        
        if (w_m_start < M && w_n_start < N) {
            FragA a; FragB b;
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                for (int j = 0; j < 16; ++j) a.x[i*16+j] = s_dO[m][j];
            }
            for (int i = 0; i < 16; ++i)
                for (int j = 0; j < 16; ++j) b.x[i*16+j] = s_V[i][warp_n * 16 + j];
            mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }
    if (w_m_start < M && w_n_start < N) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        store_matrix_sync(dP + w_m_start * N + w_n_start, out, N, mem_row_major);
    }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main() {
    constexpr int M = 1024, N = 1024, K_dim = 64;
    std::cout << "\n=== ALL KERNELS (BM=64, BN=64, BK=16) ===\n";
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << "\n\n";
    
    size_t size_AB = M * N;
    size_t size_C = M * K_dim;
    size_t size_D = N * K_dim;
    
    half_t *d_dS, *d_K, *d_Q, *d_P, *d_dO, *d_V;
    half_t *d_dQ, *d_dK, *d_dV, *d_dP;
    
    hipMalloc(&d_dS, size_AB * sizeof(half_t));
    hipMalloc(&d_K,  size_D * sizeof(half_t));
    hipMalloc(&d_Q,  size_C * sizeof(half_t));
    hipMalloc(&d_P,  size_AB * sizeof(half_t));
    hipMalloc(&d_dO, size_C * sizeof(half_t));
    hipMalloc(&d_V,  size_D * sizeof(half_t));
    hipMalloc(&d_dQ, size_C * sizeof(half_t));
    hipMalloc(&d_dK, size_D * sizeof(half_t));
    hipMalloc(&d_dV, size_D * sizeof(half_t));
    hipMalloc(&d_dP, size_AB * sizeof(half_t));
    
    dim3 grid_q((K_dim + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 grid_k((K_dim + BN - 1) / BN, (N + BM - 1) / BM);
    dim3 grid_p((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);
    
    // Warmup
    hipLaunchKernelGGL(dq_kernel, grid_q, block, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipLaunchKernelGGL(dk_kernel, grid_k, block, 0, 0, d_dS, d_Q, d_dK, M, N, K_dim);
    hipLaunchKernelGGL(dv_kernel, grid_k, block, 0, 0, d_P, d_dO, d_dV, M, N, K_dim);
    hipLaunchKernelGGL(dp_kernel, grid_p, block, 0, 0, d_dO, d_V, d_dP, M, N, K_dim);
    hipDeviceSynchronize();
    
    // Benchmark dQ
    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(dq_kernel, grid_q, block, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "dQ: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    
    // Benchmark dK
    start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(dk_kernel, grid_k, block, 0, 0, d_dS, d_Q, d_dK, M, N, K_dim);
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "dK: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    
    // Benchmark dV
    start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(dv_kernel, grid_k, block, 0, 0, d_P, d_dO, d_dV, M, N, K_dim);
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "dV: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    
    // Benchmark dP
    start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(dp_kernel, grid_p, block, 0, 0, d_dO, d_V, d_dP, M, N, K_dim);
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "dP: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    
    std::cout << "\n=== TFLOPS ===\n";
    double flops = 2.0 * M * N * K_dim;
    std::cout << "dQ: " << flops / 35e-6 / 1e12 << " TFLOPS\n";
    std::cout << "dK: " << flops / 29e-6 / 1e12 << " TFLOPS\n";
    
    hipFree(d_dS); hipFree(d_K); hipFree(d_Q); hipFree(d_P); hipFree(d_dO); hipFree(d_V);
    hipFree(d_dQ); hipFree(d_dK); hipFree(d_dV); hipFree(d_dP);
    return 0;
}

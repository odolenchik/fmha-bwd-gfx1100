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

__global__ void dv_rocwmma_lds_kernel(
    const half_t* __restrict__ P,
    const half_t* __restrict__ dO,
    half_t* __restrict__ dV,
    int M, int N, int K_dim)
{
    __shared__ half_t s_P[BM][BK];
    __shared__ half_t s_dO[BK][BN];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, col_major>;  // P^T
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;  // dO
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_n = blockIdx.y;
    int block_k = blockIdx.x;
    int n_start = block_n * BM;
    int k_start = block_k * BN;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_n = warp_id / (BN / 16);
    int warp_k = warp_id % (BN / 16);
    int w_n_start = n_start + warp_n * 16;
    int w_k_start = k_start + warp_k * 16;
    
    FragC acc_frag;
    fill_fragment(acc_frag, 0.0f);
    
    for (int m_start = 0; m_start < M; m_start += BK) {
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int m = i / BN;
            int k = k_start + (i % BN);
            if (m_start + m < M && k < K_dim) {
                s_dO[m][i % BN] = dO[(m_start + m) * K_dim + k];
            } else {
                s_dO[m][i % BN] = 0;
            }
        }
        
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int n = i / BK;
            int m = i % BK;
            if (n_start + n < N && m_start + m < M) {
                s_P[n][m] = P[(m_start + m) * N + (n_start + n)];
            } else {
                s_P[n][m] = 0;
            }
        }
        __syncthreads();
        
        if (w_n_start < N && w_k_start < K_dim) {
            FragA a_frag;
            FragB b_frag;
            
            for (int i = 0; i < 16; ++i) {
                int n = warp_n * 16 + i;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    a_frag.x[i * 16 + j] = s_P[n][j];
                }
            }
            
            for (int i = 0; i < 16; ++i) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    b_frag.x[i * 16 + j] = s_dO[i][warp_k * 16 + j];
                }
            }
            
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }
    
    if (w_n_start < N && w_k_start < K_dim) {
        FragOut out_frag;
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            out_frag.x[i] = static_cast<half_t>(acc_frag.x[i]);
        }
        store_matrix_sync(dV + w_n_start * K_dim + w_k_start, out_frag, K_dim, mem_row_major);
    }
}

int main() {
    constexpr int M = 1024, N = 1024, K_dim = 64;
    std::cout << "\n=== rocWMMA + LDS dV Kernel ===\n";
    
    size_t size_P = M * N;
    size_t size_dO = M * K_dim;
    size_t size_dV = N * K_dim;
    
    half_t *d_P, *d_dO, *d_dV;
    hipMalloc(&d_P,  size_P  * sizeof(half_t));
    hipMalloc(&d_dO, size_dO * sizeof(half_t));
    hipMalloc(&d_dV, size_dV * sizeof(half_t));
    
    dim3 grid((K_dim + BN - 1) / BN, (N + BM - 1) / BM);
    dim3 block(128);
    
    hipLaunchKernelGGL(dv_rocwmma_lds_kernel, grid, block, 0, 0, d_P, d_dO, d_dV, M, N, K_dim);
    hipDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(dv_rocwmma_lds_kernel, grid, block, 0, 0, d_P, d_dO, d_dV, M, N, K_dim);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double tflops = (2.0 * M * N * K_dim) / (time_us * 1e-6) / 1e12;
    
    std::cout << "Time: " << time_us << " us\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    
    hipFree(d_P); hipFree(d_dO); hipFree(d_dV);
    return 0;
}

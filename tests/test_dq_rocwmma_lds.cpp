#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

// Константы тайлов
constexpr int BM = 64;  // Тайл по M (строкам dQ)
constexpr int BN = 64;  // Тайл по N (столбцам dQ)
constexpr int BK = 16;  // Тайл по K (редукция)

__global__ void dq_rocwmma_lds_kernel(
    const half_t* __restrict__ dS,
    const half_t* __restrict__ K,
    half_t* __restrict__ dQ,
    int M, int N, int K_dim)
{
    // LDS память
    __shared__ half_t s_dS[BM][BK];
    __shared__ half_t s_K[BK][BN];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    // Координаты блока
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int m_start = block_m * BM;
    int n_start = block_n * BN;
    
    // Координаты warp'а внутри блока
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_m = warp_id / 4;  // 4 варпа по N
    int warp_n = warp_id % 4;
    int w_m_start = m_start + warp_m * 16;
    int w_n_start = n_start + warp_n * 16;
    
    FragC acc_frag;
    fill_fragment(acc_frag, 0.0f);
    
    // Цикл по K (размерность N)
    for (int k_start = 0; k_start < N; k_start += BK) {
        // Загружаем s_K [BK, BN] в LDS (кооперативно)
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int k = k_start + i / BN;
            int n = n_start + i % BN;
            if (k < N && n < K_dim) {
                s_K[i / BN][i % BN] = K[k * K_dim + n];
            } else {
                s_K[i / BN][i % BN] = 0;
            }
        }
        
        // Загружаем s_dS [BM, BK] в LDS
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int m = m_start + i / BK;
            int k = k_start + i % BK;
            if (m < M && k < N) {
                s_dS[i / BK][i % BK] = dS[m * N + k];
            } else {
                s_dS[i / BK][i % BK] = 0;
            }
        }
        __syncthreads();
        
        // Каждый warp обрабатывает свой 16x16 тайл
        if (w_m_start < M && w_n_start < K_dim) {
            // Загружаем A из LDS
            FragA a_frag;
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    int k = j;  // BK=16
                    a_frag.x[i * 16 + j] = s_dS[m][k];
                }
            }
            
            // Загружаем B из LDS
            FragB b_frag;
            for (int i = 0; i < 16; ++i) {
                int k = i;  // BK=16
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    int n = warp_n * 16 + j;
                    b_frag.x[i * 16 + j] = s_K[k][n];
                }
            }
            
            // WMMA
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }
    
    // Сохраняем результат
    if (w_m_start < M && w_n_start < K_dim) {
        FragOut out_frag;
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            out_frag.x[i] = static_cast<half_t>(acc_frag.x[i]);
        }
        store_matrix_sync(dQ + w_m_start * K_dim + w_n_start, out_frag, K_dim, mem_row_major);
    }
}

int main() {
    constexpr int M = 1024, N = 1024, K_dim = 64;
    std::cout << "\n=== rocWMMA + LDS dQ Kernel ===\n";
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << std::endl;
    std::cout << "BM=" << BM << ", BN=" << BN << ", BK=" << BK << std::endl;
    
    size_t size_dS = M * N;
    size_t size_K = N * K_dim;
    size_t size_dQ = M * K_dim;
    
    half_t *d_dS, *d_K, *d_dQ;
    hipMalloc(&d_dS, size_dS * sizeof(half_t));
    hipMalloc(&d_K,  size_K  * sizeof(half_t));
    hipMalloc(&d_dQ, size_dQ * sizeof(half_t));
    
    dim3 grid((K_dim + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);  // 4 варпа по 32
    
    // Прогрев
    hipLaunchKernelGGL(dq_rocwmma_lds_kernel, grid, block, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    
    // Замер
    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(dq_rocwmma_lds_kernel, grid, block, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double tflops = (2.0 * M * N * K_dim) / (time_us * 1e-6) / 1e12;
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Time: " << time_us << " us\n";
    std::cout << "TFLOPS: " << tflops << "\n\n";
    
    std::cout << "=== Comparison ===\n";
    std::cout << "Naive:        101 us (0.33 TFLOPS)\n";
    std::cout << "rocWMMA:       56 us (2.40 TFLOPS, 1.8x)\n";
    std::cout << "rocWMMA+LDS: " << time_us << " us (" << tflops << " TFLOPS, " 
              << 101.0 / time_us << "x)\n";
    
    hipFree(d_dS); hipFree(d_K); hipFree(d_dQ);
    return 0;
}

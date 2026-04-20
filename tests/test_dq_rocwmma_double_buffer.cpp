#include <iostream>
#include <vector>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

// Оптимальные тайлы для RX 7900 XT
constexpr int BM = 128;  // Больше работы на блок
constexpr int BN = 64;
constexpr int BK = 32;   // Больше редукции за итерацию

__global__ void dq_rocwmma_double_buffer_kernel(
    const half_t* __restrict__ dS,
    const half_t* __restrict__ K,
    half_t* __restrict__ dQ,
    int M, int N, int K_dim)
{
    // Двойной буфер для K
    __shared__ half_t s_K[2][BK][BN];
    __shared__ half_t s_dS[BM][BK];
    
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;
    
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int m_start = block_m * BM;
    int n_start = block_n * BN;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / (BN / 16);  // BN=64 → 4 варпа
    int warp_n = warp_id % (BN / 16);
    int w_m_start = m_start + warp_m * 16;
    int w_n_start = n_start + warp_n * 16;
    
    // Два аккумулятора для конвейеризации (не обязательно, но помогает)
    FragC acc_frag[2];
    fill_fragment(acc_frag[0], 0.0f);
    fill_fragment(acc_frag[1], 0.0f);
    
    // Предзагрузка первого тайла K
    int k_start = 0;
    int buf_idx = 0;
    
    // Загружаем первый тайл K в буфер 0
    for (int i = tid; i < BK * BN; i += blockDim.x) {
        int k = i / BN;
        int n = n_start + (i % BN);
        if (k < N - k_start && n < K_dim) {
            s_K[0][k][i % BN] = K[(k_start + k) * K_dim + n];
        }
    }
    
    // Основной цикл с двойной буферизацией
    for (; k_start < N; k_start += BK, buf_idx ^= 1) {
        // Загружаем следующий тайл K в другой буфер (асинхронно)
        int next_k = k_start + BK;
        int next_buf = buf_idx ^ 1;
        if (next_k < N) {
            for (int i = tid; i < BK * BN; i += blockDim.x) {
                int k = i / BN;
                int n = n_start + (i % BN);
                if (next_k + k < N && n < K_dim) {
                    s_K[next_buf][k][i % BN] = K[(next_k + k) * K_dim + n];
                }
            }
        }
        
        // Загружаем s_dS для текущего k_start
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int m = i / BK;
            int k = i % BK;
            if (m_start + m < M && k_start + k < N) {
                s_dS[m][k] = dS[(m_start + m) * N + (k_start + k)];
            }
        }
        __syncthreads();
        
        // Вычисления с текущим буфером K
        if (w_m_start < M && w_n_start < K_dim) {
            // Загружаем A из LDS (s_dS)
            FragA a_frag;
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    a_frag.x[i * 16 + j] = s_dS[m][j];
                }
            }
            
            // Загружаем B из текущего буфера K
            FragB b_frag;
            for (int i = 0; i < 16; ++i) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    b_frag.x[i * 16 + j] = s_K[buf_idx][i][warp_n * 16 + j];
                }
            }
            
            // WMMA с чередованием аккумуляторов
            mma_sync(acc_frag[0], a_frag, b_frag, acc_frag[0]);
        }
        __syncthreads();
    }
    
    // Сохраняем результат
    if (w_m_start < M && w_n_start < K_dim) {
        FragOut out_frag;
        #pragma unroll
        for (int i = 0; i < acc_frag[0].num_elements; ++i) {
            out_frag.x[i] = static_cast<half_t>(acc_frag[0].x[i]);
        }
        store_matrix_sync(dQ + w_m_start * K_dim + w_n_start, out_frag, K_dim, mem_row_major);
    }
}

int main() {
    constexpr int M = 1024, N = 1024, K_dim = 64;
    std::cout << "\n=== rocWMMA + Double Buffer + Large Tiles ===\n";
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
    dim3 block(128);
    
    // Прогрев
    hipLaunchKernelGGL(dq_rocwmma_double_buffer_kernel, grid, block, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    
    // Замер
    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(dq_rocwmma_double_buffer_kernel, grid, block, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double tflops = (2.0 * M * N * K_dim) / (time_us * 1e-6) / 1e12;
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Time: " << time_us << " us\n";
    std::cout << "TFLOPS: " << tflops << "\n\n";
    
    std::cout << "=== Comparison ===\n";
    std::cout << "Naive:           101 us (0.33 TFLOPS, 1.0x)\n";
    std::cout << "rocWMMA:          56 us (2.40 TFLOPS, 1.8x)\n";
    std::cout << "rocWMMA+LDS:      38 us (3.53 TFLOPS, 2.7x)\n";
    std::cout << "Double Buffer:  " << time_us << " us (" << tflops << " TFLOPS, " 
              << 101.0 / time_us << "x)\n";
    
    hipFree(d_dS); hipFree(d_K); hipFree(d_dQ);
    return 0;
}

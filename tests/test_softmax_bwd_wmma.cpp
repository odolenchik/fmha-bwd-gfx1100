#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

constexpr int BM = 64;   // Тайл по M (строкам)
constexpr int BN = 64;   // Тайл по N (столбцам)
constexpr int BK = 16;   // Для редукции

// ----------------------------------------------------------------------------
// Kernel 1: Вычисление rowsum = reduce(dP * P)
// ----------------------------------------------------------------------------
__global__ void softmax_bwd_rowsum_kernel(
    const half_t* __restrict__ P,
    const half_t* __restrict__ dP,
    half_t* __restrict__ rowsum,
    int M, int N)
{
    __shared__ half_t s_data[BM][BN];
    __shared__ float s_reduce[BM];
    
    int block_m = blockIdx.x;
    int m_start = block_m * BM;
    int tid = threadIdx.x;
    
    // Загружаем тайл P и dP, вычисляем dP * P
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN;
        int n = i % BN;
        int global_m = m_start + m;
        int global_n = n;
        if (global_m < M && global_n < N) {
            half_t p_val = P[global_m * N + global_n];
            half_t dp_val = dP[global_m * N + global_n];
            s_data[m][n] = static_cast<half_t>(static_cast<float>(p_val) * static_cast<float>(dp_val));
        } else {
            s_data[m][n] = 0;
        }
    }
    __syncthreads();
    
    // Редукция по строкам внутри блока
    for (int m = tid; m < BM; m += blockDim.x) {
        int global_m = m_start + m;
        if (global_m < M) {
            float sum = 0.0f;
            for (int n = 0; n < BN; n++) {
                sum += static_cast<float>(s_data[m][n]);
            }
            s_reduce[m] = sum;
        }
    }
    __syncthreads();
    
    // Сохраняем результат
    if (tid < BM) {
        int global_m = m_start + tid;
        if (global_m < M) {
            rowsum[global_m] = static_cast<half_t>(s_reduce[tid]);
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel 2: dS = P * (dP - rowsum)
// ----------------------------------------------------------------------------
__global__ void softmax_bwd_ds_kernel(
    const half_t* __restrict__ P,
    const half_t* __restrict__ dP,
    const half_t* __restrict__ rowsum,
    half_t* __restrict__ dS,
    int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / N;
    int n = idx % N;
    
    if (m < M && n < N) {
        float p_val = static_cast<float>(P[m * N + n]);
        float dp_val = static_cast<float>(dP[m * N + n]);
        float rs = static_cast<float>(rowsum[m]);
        dS[m * N + n] = static_cast<half_t>(p_val * (dp_val - rs));
    }
}

// ----------------------------------------------------------------------------
// CPU reference
// ----------------------------------------------------------------------------
void cpu_softmax_bwd(const half_t* P, const half_t* dP, half_t* dS, int M, int N) {
    std::vector<float> rowsum(M, 0.0f);
    
    // rowsum = sum(dP * P)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            rowsum[m] += static_cast<float>(P[m * N + n]) * static_cast<float>(dP[m * N + n]);
        }
    }
    
    // dS = P * (dP - rowsum)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float p_val = static_cast<float>(P[m * N + n]);
            float dp_val = static_cast<float>(dP[m * N + n]);
            dS[m * N + n] = static_cast<half_t>(p_val * (dp_val - rowsum[m]));
        }
    }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main() {
    constexpr int M = 1024, N = 1024;
    std::cout << "\n=== Softmax Backward GPU ===\n";
    std::cout << "M=" << M << ", N=" << N << "\n\n";
    
    size_t size_MN = M * N;
    size_t size_M = M;
    
    // Host buffers
    std::vector<half_t> h_P(size_MN), h_dP(size_MN), h_dS_gpu(size_MN), h_dS_cpu(size_MN);
    std::vector<half_t> h_rowsum(size_M);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.0f);
    
    for (auto& v : h_P) v = static_cast<half_t>(dist_pos(gen));  // P > 0
    for (auto& v : h_dP) v = static_cast<half_t>(dist(gen));
    
    // Нормализуем P (softmax)
    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) sum += static_cast<float>(h_P[m * N + n]);
        for (int n = 0; n < N; n++) h_P[m * N + n] = static_cast<half_t>(static_cast<float>(h_P[m * N + n]) / sum);
    }
    
    // Device buffers
    half_t *d_P, *d_dP, *d_rowsum, *d_dS;
    hipMalloc(&d_P, size_MN * sizeof(half_t));
    hipMalloc(&d_dP, size_MN * sizeof(half_t));
    hipMalloc(&d_rowsum, size_M * sizeof(half_t));
    hipMalloc(&d_dS, size_MN * sizeof(half_t));
    
    hipMemcpy(d_P, h_P.data(), size_MN * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dP, h_dP.data(), size_MN * sizeof(half_t), hipMemcpyHostToDevice);
    
    // CPU reference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_softmax_bwd(h_P.data(), h_dP.data(), h_dS_cpu.data(), M, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    // GPU: rowsum
    dim3 grid_rs((M + BM - 1) / BM);
    dim3 block_rs(256);
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(softmax_bwd_rowsum_kernel, grid_rs, block_rs, 0, 0, d_P, d_dP, d_rowsum, M, N);
    hipDeviceSynchronize();
    
    // GPU: dS
    dim3 grid_ds((size_MN + 255) / 256);
    dim3 block_ds(256);
    hipLaunchKernelGGL(softmax_bwd_ds_kernel, grid_ds, block_ds, 0, 0, d_P, d_dP, d_rowsum, d_dS, M, N);
    hipDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    
    hipMemcpy(h_dS_gpu.data(), d_dS, size_MN * sizeof(half_t), hipMemcpyDeviceToHost);
    
    // Проверка точности
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_MN; i++) {
        float diff = std::abs(static_cast<float>(h_dS_gpu[i]) - static_cast<float>(h_dS_cpu[i]));
        if (diff > max_diff) max_diff = diff;
    }
    
    std::cout << "CPU time: " << cpu_time << " us\n";
    std::cout << "GPU time: " << gpu_time << " us\n";
    std::cout << "Speedup: " << static_cast<float>(cpu_time) / gpu_time << "x\n";
    std::cout << "Max diff: " << max_diff << "\n";
    
    if (max_diff < 0.01f) {
        std::cout << "✅ TEST PASSED\n";
    } else {
        std::cout << "❌ TEST FAILED\n";
    }
    
    hipFree(d_P); hipFree(d_dP); hipFree(d_rowsum); hipFree(d_dS);
    return (max_diff < 0.01f) ? 0 : 1;
}

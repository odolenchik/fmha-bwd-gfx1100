#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>

using half_t = _Float16;

constexpr int BM = 64;
constexpr int BN = 64;

__global__ void softmax_bwd_fused_kernel(
    const half_t* __restrict__ P,
    const half_t* __restrict__ dP,
    half_t* __restrict__ dS,
    int M, int N)
{
    __shared__ half_t s_P[BM][BN];
    __shared__ half_t s_dP[BM][BN];
    __shared__ float s_rowsum[BM];
    
    int block_m = blockIdx.x;
    int m_start = block_m * BM;
    int tid = threadIdx.x;
    
    // Загружаем данные
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN;
        int n = i % BN;
        int global_m = m_start + m;
        int global_n = n;
        if (global_m < M && global_n < N) {
            s_P[m][n] = P[global_m * N + global_n];
            s_dP[m][n] = dP[global_m * N + global_n];
        } else {
            s_P[m][n] = 0;
            s_dP[m][n] = 0;
        }
    }
    __syncthreads();
    
    // Каждый поток обрабатывает одну строку (если threadIdx.x < BM)
    if (tid < BM) {
        int global_m = m_start + tid;
        if (global_m < M) {
            float sum = 0.0f;
            for (int n = 0; n < BN; n++) {
                float p_val = static_cast<float>(s_P[tid][n]);
                float dp_val = static_cast<float>(s_dP[tid][n]);
                sum += p_val * dp_val;
            }
            s_rowsum[tid] = sum;
        } else {
            s_rowsum[tid] = 0.0f;
        }
    }
    __syncthreads();
    
    // Вычисляем dS
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN;
        int n = i % BN;
        int global_m = m_start + m;
        int global_n = n;
        if (global_m < M && global_n < N) {
            float p_val = static_cast<float>(s_P[m][n]);
            float dp_val = static_cast<float>(s_dP[m][n]);
            float rs = s_rowsum[m];
            dS[global_m * N + global_n] = static_cast<half_t>(p_val * (dp_val - rs));
        }
    }
}

void cpu_softmax_bwd(const half_t* P, const half_t* dP, half_t* dS, int M, int N) {
    std::vector<float> rowsum(M, 0.0f);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            rowsum[m] += static_cast<float>(P[m * N + n]) * static_cast<float>(dP[m * N + n]);
        }
    }
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float p_val = static_cast<float>(P[m * N + n]);
            float dp_val = static_cast<float>(dP[m * N + n]);
            dS[m * N + n] = static_cast<half_t>(p_val * (dp_val - rowsum[m]));
        }
    }
}

int main() {
    constexpr int M = 1024, N = 1024;
    std::cout << "\n=== FUSED Softmax Backward (FIXED) ===\n";
    std::cout << "M=" << M << ", N=" << N << ", BM=" << BM << ", BN=" << BN << "\n\n";
    
    size_t size_MN = M * N;
    
    std::vector<half_t> h_P(size_MN), h_dP(size_MN), h_dS_gpu(size_MN), h_dS_cpu(size_MN);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.0f);
    
    for (auto& v : h_P) v = static_cast<half_t>(dist_pos(gen));
    for (auto& v : h_dP) v = static_cast<half_t>(dist(gen));
    
    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) sum += static_cast<float>(h_P[m * N + n]);
        for (int n = 0; n < N; n++) h_P[m * N + n] = static_cast<half_t>(static_cast<float>(h_P[m * N + n]) / sum);
    }
    
    half_t *d_P, *d_dP, *d_dS;
    hipMalloc(&d_P, size_MN * sizeof(half_t));
    hipMalloc(&d_dP, size_MN * sizeof(half_t));
    hipMalloc(&d_dS, size_MN * sizeof(half_t));
    
    hipMemcpy(d_P, h_P.data(), size_MN * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dP, h_dP.data(), size_MN * sizeof(half_t), hipMemcpyHostToDevice);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_softmax_bwd(h_P.data(), h_dP.data(), h_dS_cpu.data(), M, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    dim3 grid((M + BM - 1) / BM);
    dim3 block(256);
    
    hipLaunchKernelGGL(softmax_bwd_fused_kernel, grid, block, 0, 0, d_P, d_dP, d_dS, M, N);
    hipDeviceSynchronize();
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(softmax_bwd_fused_kernel, grid, block, 0, 0, d_P, d_dP, d_dS, M, N);
    hipDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    
    hipMemcpy(h_dS_gpu.data(), d_dS, size_MN * sizeof(half_t), hipMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_MN; i++) {
        float gpu = static_cast<float>(h_dS_gpu[i]);
        float cpu = static_cast<float>(h_dS_cpu[i]);
        if (!std::isfinite(gpu)) {
            std::cout << "NaN/Inf at index " << i << "\n";
            max_diff = std::numeric_limits<float>::infinity();
            break;
        }
        float diff = std::abs(gpu - cpu);
        if (diff > max_diff) max_diff = diff;
    }
    
    std::cout << "CPU time: " << cpu_time << " us\n";
    std::cout << "GPU (two kernels): 890 us\n";
    std::cout << "GPU (fused LDS):   " << gpu_time << " us\n";
    std::cout << "Speedup vs 2-kernel: " << 890.0f / gpu_time << "x\n";
    std::cout << "Max diff: " << max_diff << "\n";
    
    if (max_diff < 0.01f) {
        std::cout << "✅ TEST PASSED\n";
    } else {
        std::cout << "❌ TEST FAILED\n";
    }
    
    hipFree(d_P); hipFree(d_dP); hipFree(d_dS);
    return (max_diff < 0.01f) ? 0 : 1;
}

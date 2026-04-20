#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>
#include "cpu_reference.hpp"

using half_t = _Float16;

// Правильное GPU ядро: dQ = dS @ K
__global__ void dq_kernel(const half_t* dS, const half_t* K, half_t* dQ,
                          int M, int N, int K_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / K_dim;
    int k = idx % K_dim;
    
    if (m < M && k < K_dim) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum += static_cast<float>(dS[m * N + n]) * 
                   static_cast<float>(K[n * K_dim + k]);
        }
        dQ[m * K_dim + k] = static_cast<half_t>(sum);
    }
}

int main() {
    constexpr int M = 16, N = 16, K_dim = 64;
    
    const size_t size_Q = M * K_dim;
    const size_t size_K = N * K_dim;
    const size_t size_V = N * K_dim;
    const size_t size_O = M * K_dim;
    const size_t size_P = M * N;
    
    std::cout << "\n=== GPU dQ Kernel (with dS) Test ===\n" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << std::endl;
    
    // Host buffers
    std::vector<half_t> h_Q(size_Q), h_K(size_K), h_V(size_V);
    std::vector<half_t> h_dO(size_O);
    std::vector<half_t> h_dQ_gpu(size_Q), h_dQ_cpu(size_Q);
    
    // Random init
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_Q) v = half_cast(dist(gen));
    for (auto& v : h_K) v = half_cast(dist(gen));
    for (auto& v : h_V) v = half_cast(dist(gen));
    for (auto& v : h_dO) v = half_cast(dist(gen));
    
    // CPU: Forward pass to get P
    std::vector<half_t> h_O(size_O);
    std::vector<float> h_P_float(size_P);
    fmha_forward_cpu(h_Q.data(), h_K.data(), h_V.data(), 
                     h_O.data(), h_P_float.data(), M, N, K_dim);
    
    // CPU: Backward pass
    std::vector<half_t> h_dK_cpu(size_K), h_dV_cpu(size_V);
    fmha_backward_cpu(h_Q.data(), h_K.data(), h_V.data(),
                      h_P_float.data(), h_dO.data(),
                      h_dQ_cpu.data(), h_dK_cpu.data(), h_dV_cpu.data(),
                      M, N, K_dim);
    
    // Получаем dS для GPU
    std::vector<float> h_dP_float(size_P);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K_dim; ++k) {
                sum += float_cast(h_dO[m * K_dim + k]) * 
                       float_cast(h_V[n * K_dim + k]);
            }
            h_dP_float[m * N + n] = sum;
        }
    }
    
    std::vector<float> h_dS_float(size_P);
    softmax_backward(h_P_float.data(), h_dP_float.data(), h_dS_float.data(), M, N);
    
    std::vector<half_t> h_dS_half(size_P);
    for (size_t i = 0; i < size_P; ++i) {
        h_dS_half[i] = half_cast(h_dS_float[i]);
    }
    
    // GPU buffers
    half_t *d_dS, *d_K, *d_dQ;
    hipMalloc(&d_dS, size_P * sizeof(half_t));
    hipMalloc(&d_K,  size_K * sizeof(half_t));
    hipMalloc(&d_dQ, size_Q * sizeof(half_t));
    
    hipMemcpy(d_dS, h_dS_half.data(), size_P * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_K,  h_K.data(),       size_K * sizeof(half_t), hipMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (M * K_dim + block_size - 1) / block_size;
    hipLaunchKernelGGL(dq_kernel, dim3(grid_size), dim3(block_size), 0, 0, 
                       d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    
    hipMemcpy(h_dQ_gpu.data(), d_dQ, size_Q * sizeof(half_t), hipMemcpyDeviceToHost);
    
    // Verify
    bool ok = true;
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_Q; ++i) {
        float gpu_val = float_cast(h_dQ_gpu[i]);
        float cpu_val = float_cast(h_dQ_cpu[i]);
        float diff = std::abs(gpu_val - cpu_val);
        if (diff > max_diff) max_diff = diff;
        if (diff > 5e-2f) {  // Больший порог из-за накопления ошибок
            std::cerr << "Mismatch at " << i << ": GPU=" << gpu_val 
                      << " CPU=" << cpu_val << " diff=" << diff << std::endl;
            ok = false;
        }
    }
    
    std::cout << "dQ max_diff: " << max_diff << std::endl;
    std::cout << "Test " << (ok ? "PASSED" : "FAILED") << std::endl;
    
    if (ok) {
        std::cout << "\nSample values:" << std::endl;
        std::cout << "dQ[0,0] = " << float_cast(h_dQ_gpu[0]) << std::endl;
        std::cout << "dQ[8,32] = " << float_cast(h_dQ_gpu[8 * K_dim + 32]) << std::endl;
    }
    
    hipFree(d_dS); hipFree(d_K); hipFree(d_dQ);
    return ok ? 0 : 1;
}

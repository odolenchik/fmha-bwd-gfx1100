#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>
#include "cpu_reference.hpp"

using half_t = _Float16;

// Правильное GPU ядро: dK = dS^T @ Q
__global__ void dk_kernel(const half_t* dS, const half_t* Q, half_t* dK,
                          int M, int N, int K_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx / K_dim;
    int k = idx % K_dim;
    
    if (n < N && k < K_dim) {
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            sum += static_cast<float>(dS[m * N + n]) * 
                   static_cast<float>(Q[m * K_dim + k]);
        }
        dK[n * K_dim + k] = static_cast<half_t>(sum);
    }
}

// Правильное GPU ядро: dV = P^T @ dO
__global__ void dv_kernel(const half_t* P, const half_t* dO, half_t* dV,
                          int M, int N, int K_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx / K_dim;
    int k = idx % K_dim;
    
    if (n < N && k < K_dim) {
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            sum += static_cast<float>(P[m * N + n]) * 
                   static_cast<float>(dO[m * K_dim + k]);
        }
        dV[n * K_dim + k] = static_cast<half_t>(sum);
    }
}

int main() {
    constexpr int M = 16, N = 16, K_dim = 64;
    
    const size_t size_Q = M * K_dim;
    const size_t size_K = N * K_dim;
    const size_t size_V = N * K_dim;
    const size_t size_O = M * K_dim;
    const size_t size_P = M * N;
    
    std::cout << "\n=== GPU dK/dV Kernels Test ===\n" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << std::endl;
    
    // Host buffers
    std::vector<half_t> h_Q(size_Q), h_K(size_K), h_V(size_V);
    std::vector<half_t> h_dO(size_O);
    std::vector<half_t> h_dK_gpu(size_K), h_dV_gpu(size_V);
    std::vector<half_t> h_dK_cpu(size_K), h_dV_cpu(size_V);
    
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
    
    // CPU: Backward pass (эталон)
    std::vector<half_t> h_dQ_cpu(size_Q);
    fmha_backward_cpu(h_Q.data(), h_K.data(), h_V.data(),
                      h_P_float.data(), h_dO.data(),
                      h_dQ_cpu.data(), h_dK_cpu.data(), h_dV_cpu.data(),
                      M, N, K_dim);
    
    // Получаем dS для dK
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
    std::vector<half_t> h_P_half(size_P);
    for (size_t i = 0; i < size_P; ++i) {
        h_dS_half[i] = half_cast(h_dS_float[i]);
        h_P_half[i] = half_cast(h_P_float[i]);
    }
    
    // GPU buffers
    half_t *d_dS, *d_Q, *d_dK, *d_P, *d_dO, *d_dV;
    hipMalloc(&d_dS, size_P * sizeof(half_t));
    hipMalloc(&d_Q,  size_Q * sizeof(half_t));
    hipMalloc(&d_dK, size_K * sizeof(half_t));
    hipMalloc(&d_P,  size_P * sizeof(half_t));
    hipMalloc(&d_dO, size_O * sizeof(half_t));
    hipMalloc(&d_dV, size_V * sizeof(half_t));
    
    hipMemcpy(d_dS, h_dS_half.data(), size_P * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_Q,  h_Q.data(),       size_Q * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_P,  h_P_half.data(),  size_P * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO.data(),      size_O * sizeof(half_t), hipMemcpyHostToDevice);
    
    // Launch dK kernel
    int block_size = 256;
    int grid_size = (N * K_dim + block_size - 1) / block_size;
    hipLaunchKernelGGL(dk_kernel, dim3(grid_size), dim3(block_size), 0, 0, 
                       d_dS, d_Q, d_dK, M, N, K_dim);
    
    // Launch dV kernel
    hipLaunchKernelGGL(dv_kernel, dim3(grid_size), dim3(block_size), 0, 0, 
                       d_P, d_dO, d_dV, M, N, K_dim);
    hipDeviceSynchronize();
    
    hipMemcpy(h_dK_gpu.data(), d_dK, size_K * sizeof(half_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_dV_gpu.data(), d_dV, size_V * sizeof(half_t), hipMemcpyDeviceToHost);
    
    // Verify dK
    bool dk_ok = true;
    float dk_max_diff = 0.0f;
    for (size_t i = 0; i < size_K; ++i) {
        float gpu_val = float_cast(h_dK_gpu[i]);
        float cpu_val = float_cast(h_dK_cpu[i]);
        float diff = std::abs(gpu_val - cpu_val);
        if (diff > dk_max_diff) dk_max_diff = diff;
        if (diff > 5e-2f) {
            std::cerr << "dK mismatch at " << i << ": GPU=" << gpu_val 
                      << " CPU=" << cpu_val << " diff=" << diff << std::endl;
            dk_ok = false;
        }
    }
    
    // Verify dV
    bool dv_ok = true;
    float dv_max_diff = 0.0f;
    for (size_t i = 0; i < size_V; ++i) {
        float gpu_val = float_cast(h_dV_gpu[i]);
        float cpu_val = float_cast(h_dV_cpu[i]);
        float diff = std::abs(gpu_val - cpu_val);
        if (diff > dv_max_diff) dv_max_diff = diff;
        if (diff > 5e-2f) {
            std::cerr << "dV mismatch at " << i << ": GPU=" << gpu_val 
                      << " CPU=" << cpu_val << " diff=" << diff << std::endl;
            dv_ok = false;
        }
    }
    
    std::cout << "dK max_diff: " << dk_max_diff << " -> " << (dk_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "dV max_diff: " << dv_max_diff << " -> " << (dv_ok ? "PASSED" : "FAILED") << std::endl;
    
    bool ok = dk_ok && dv_ok;
    std::cout << "\nOverall Test " << (ok ? "PASSED" : "FAILED") << std::endl;
    
    if (ok) {
        std::cout << "\nSample values:" << std::endl;
        std::cout << "dK[0,0] = " << float_cast(h_dK_gpu[0]) << std::endl;
        std::cout << "dV[0,0] = " << float_cast(h_dV_gpu[0]) << std::endl;
    }
    
    hipFree(d_dS); hipFree(d_Q); hipFree(d_dK);
    hipFree(d_P); hipFree(d_dO); hipFree(d_dV);
    return ok ? 0 : 1;
}

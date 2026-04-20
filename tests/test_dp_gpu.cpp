#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>
#include "cpu_reference.hpp"

using half_t = _Float16;

// GPU kernel: dP = dO @ V^T
__global__ void dp_kernel(const half_t* dO, const half_t* V, half_t* dP,
                          int M, int N, int K_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int m = idx / N;
    int n = idx % N;
    
    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K_dim; ++k) {
            sum += static_cast<float>(dO[m * K_dim + k]) * 
                   static_cast<float>(V[n * K_dim + k]);
        }
        dP[m * N + n] = static_cast<half_t>(sum);
    }
}

int main() {
    constexpr int M = 16, N = 16, K_dim = 64;
    
    const size_t size_dO = M * K_dim;
    const size_t size_V  = N * K_dim;
    const size_t size_dP = M * N;
    
    std::cout << "\n=== GPU dP Kernel Test ===\n" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << std::endl;
    
    // Host buffers
    std::vector<half_t> h_dO(size_dO), h_V(size_V);
    std::vector<half_t> h_dP_gpu(size_dP), h_dP_cpu(size_dP);
    
    // Random init
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_dO) v = static_cast<half_t>(dist(gen));
    for (auto& v : h_V)  v = static_cast<half_t>(dist(gen));
    
    // CPU reference
    std::vector<float> h_dP_cpu_float(size_dP);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K_dim; ++k) {
                sum += float_cast(h_dO[m * K_dim + k]) * 
                       float_cast(h_V[n * K_dim + k]);
            }
            h_dP_cpu_float[m * N + n] = sum;
            h_dP_cpu[m * N + n] = half_cast(sum);
        }
    }
    
    // GPU buffers
    half_t *d_dO, *d_V, *d_dP;
    hipMalloc(&d_dO, size_dO * sizeof(half_t));
    hipMalloc(&d_V,  size_V  * sizeof(half_t));
    hipMalloc(&d_dP, size_dP * sizeof(half_t));
    
    hipMemcpy(d_dO, h_dO.data(), size_dO * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_V,  h_V.data(),  size_V  * sizeof(half_t), hipMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (M * N + block_size - 1) / block_size;
    hipLaunchKernelGGL(dp_kernel, dim3(grid_size), dim3(block_size), 0, 0, 
                       d_dO, d_V, d_dP, M, N, K_dim);
    hipDeviceSynchronize();
    
    hipMemcpy(h_dP_gpu.data(), d_dP, size_dP * sizeof(half_t), hipMemcpyDeviceToHost);
    
    // Verify
    bool ok = true;
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_dP; ++i) {
        float gpu_val = float_cast(h_dP_gpu[i]);
        float cpu_val = float_cast(h_dP_cpu[i]);
        float diff = std::abs(gpu_val - cpu_val);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-2f) {
            std::cerr << "Mismatch at " << i << ": GPU=" << gpu_val 
                      << " CPU=" << cpu_val << " diff=" << diff << std::endl;
            ok = false;
        }
    }
    
    std::cout << "dP max_diff: " << max_diff << std::endl;
    std::cout << "Test " << (ok ? "PASSED" : "FAILED") << std::endl;
    
    if (ok) {
        std::cout << "\nSample values:" << std::endl;
        std::cout << "dP[0,0] = " << float_cast(h_dP_gpu[0]) << std::endl;
        std::cout << "dP[8,8] = " << float_cast(h_dP_gpu[8 * N + 8]) << std::endl;
        std::cout << "dP[15,15] = " << float_cast(h_dP_gpu[15 * N + 15]) << std::endl;
    }
    
    hipFree(d_dO); hipFree(d_V); hipFree(d_dP);
    return ok ? 0 : 1;
}

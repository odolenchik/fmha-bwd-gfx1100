#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>
#include "src/kernel/correct/cpu_reference.hpp"
#include "src/kernel/correct/fmha_bwd_dp_kernel.hpp"
#include "src/kernel/correct/fmha_bwd_softmax_kernel.hpp"
#include "src/kernel/correct/fmha_bwd_dq_kernel.hpp"
#include "src/kernel/correct/fmha_bwd_dk_kernel.hpp"
#include "src/kernel/correct/fmha_bwd_dv_kernel.hpp"

using half_t = _Float16;

bool verify(const half_t* gpu, const half_t* cpu, size_t size, 
            const char* name, float eps = 5e-2f) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float g = float_cast(gpu[i]);
        float c = float_cast(cpu[i]);
        float diff = std::abs(g - c);
        if (diff > max_diff) max_diff = diff;
        if (diff > eps) {
            std::cerr << name << " mismatch at " << i << ": GPU=" << g 
                      << " CPU=" << c << " diff=" << diff << std::endl;
            return false;
        }
    }
    std::cout << name << " max_diff=" << max_diff << std::endl;
    return true;
}

int main() {
    constexpr int B = 1, H = 1;
    constexpr int M = 16, N = 16, K_dim = 64;
    
    const size_t size_Q = B * H * M * K_dim;
    const size_t size_K = B * H * N * K_dim;
    const size_t size_V = B * H * N * K_dim;
    const size_t size_O = B * H * M * K_dim;
    const size_t size_P = B * H * M * N;
    
    std::cout << "\n=== Full FMHA Backward Correct Test ===\n" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << std::endl;
    
    // Host buffers
    std::vector<half_t> h_Q(size_Q), h_K(size_K), h_V(size_V);
    std::vector<half_t> h_dO(size_O);
    std::vector<half_t> h_dQ_gpu(size_Q), h_dK_gpu(size_K), h_dV_gpu(size_V);
    std::vector<half_t> h_dQ_cpu(size_Q), h_dK_cpu(size_K), h_dV_cpu(size_V);
    
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
    fmha_backward_cpu(h_Q.data(), h_K.data(), h_V.data(),
                      h_P_float.data(), h_dO.data(),
                      h_dQ_cpu.data(), h_dK_cpu.data(), h_dV_cpu.data(),
                      M, N, K_dim);
    
    // Конвертируем P в half
    std::vector<half_t> h_P_half(size_P);
    for (size_t i = 0; i < size_P; ++i) {
        h_P_half[i] = half_cast(h_P_float[i]);
    }
    
    // GPU buffers
    half_t *d_Q, *d_K, *d_V, *d_dO, *d_P, *d_dP, *d_dS, *d_dQ, *d_dK, *d_dV;
    hipMalloc(&d_Q,  size_Q * sizeof(half_t));
    hipMalloc(&d_K,  size_K * sizeof(half_t));
    hipMalloc(&d_V,  size_V * sizeof(half_t));
    hipMalloc(&d_dO, size_O * sizeof(half_t));
    hipMalloc(&d_P,  size_P * sizeof(half_t));
    hipMalloc(&d_dP, size_P * sizeof(half_t));
    hipMalloc(&d_dS, size_P * sizeof(half_t));
    hipMalloc(&d_dQ, size_Q * sizeof(half_t));
    hipMalloc(&d_dK, size_K * sizeof(half_t));
    hipMalloc(&d_dV, size_V * sizeof(half_t));
    
    hipMemcpy(d_Q,  h_Q.data(),  size_Q * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_K,  h_K.data(),  size_K * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_V,  h_V.data(),  size_V * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO.data(), size_O * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_P,  h_P_half.data(), size_P * sizeof(half_t), hipMemcpyHostToDevice);
    
    // Шаг 1: dP = dO @ V^T
    {
        int block_size = 256;
        int grid_size = (M * N + block_size - 1) / block_size;
        hipLaunchKernelGGL(fmha_bwd_dp_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                           d_dO, d_V, d_dP, M, N, K_dim);
    }
    hipDeviceSynchronize();
    
    // Шаг 2: dS = softmax_backward(P, dP)
    {
        int block_size = 32;
        size_t smem_size = block_size * sizeof(float);
        hipLaunchKernelGGL(fmha_bwd_softmax_kernel, dim3(M), dim3(block_size), smem_size, 0,
                           d_P, d_dP, d_dS, M, N);
    }
    hipDeviceSynchronize();
    
    // Шаг 3: dQ = dS @ K
    {
        int block_size = 256;
        int grid_size = (M * K_dim + block_size - 1) / block_size;
        hipLaunchKernelGGL(fmha_bwd_dq_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                           d_dS, d_K, d_dQ, M, N, K_dim);
    }
    
    // Шаг 4: dK = dS^T @ Q
    {
        int block_size = 256;
        int grid_size = (N * K_dim + block_size - 1) / block_size;
        hipLaunchKernelGGL(fmha_bwd_dk_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                           d_dS, d_Q, d_dK, M, N, K_dim);
    }
    
    // Шаг 5: dV = P^T @ dO
    {
        int block_size = 256;
        int grid_size = (N * K_dim + block_size - 1) / block_size;
        hipLaunchKernelGGL(fmha_bwd_dv_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                           d_P, d_dO, d_dV, M, N, K_dim);
    }
    hipDeviceSynchronize();
    
    // Копируем результаты
    hipMemcpy(h_dQ_gpu.data(), d_dQ, size_Q * sizeof(half_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_dK_gpu.data(), d_dK, size_K * sizeof(half_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_dV_gpu.data(), d_dV, size_V * sizeof(half_t), hipMemcpyDeviceToHost);
    
    // Верификация
    std::cout << "\n=== Verification ===" << std::endl;
    bool q_ok = verify(h_dQ_gpu.data(), h_dQ_cpu.data(), size_Q, "dQ", 5e-2f);
    bool k_ok = verify(h_dK_gpu.data(), h_dK_cpu.data(), size_K, "dK", 5e-2f);
    bool v_ok = verify(h_dV_gpu.data(), h_dV_cpu.data(), size_V, "dV", 5e-2f);
    
    bool ok = q_ok && k_ok && v_ok;
    std::cout << "\n=== Final Result: " << (ok ? "PASSED" : "FAILED") << " ===" << std::endl;
    
    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_dO); hipFree(d_P);
    hipFree(d_dP); hipFree(d_dS); hipFree(d_dQ); hipFree(d_dK); hipFree(d_dV);
    
    return ok ? 0 : 1;
}

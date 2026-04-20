#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

__global__ void dq_rocwmma_kernel(
    const half_t* __restrict__ dS,
    const half_t* __restrict__ K,
    half_t* __restrict__ dQ,
    int M, int N, int K_dim)
{
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;

    int warp_m = blockIdx.y;
    int warp_k = blockIdx.x;
    int m_start = warp_m * 16;
    int k_start = warp_k * 16;

    FragA a_frag;
    FragB b_frag;
    FragC acc_frag;

    fill_fragment(acc_frag, 0.0f);

    for (int n_start = 0; n_start < N; n_start += 16) {
        load_matrix_sync(a_frag, dS + m_start * N + n_start, N);
        load_matrix_sync(b_frag, K + n_start * K_dim + k_start, K_dim);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Layout для accumulator передаётся как 4-й аргумент
    store_matrix_sync(dQ + m_start * K_dim + k_start, acc_frag, K_dim, row_major{});
}

void cpu_dq(const half_t* dS, const half_t* K, half_t* dQ, int M, int N, int K_dim) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                sum += static_cast<float>(dS[m * N + n]) * static_cast<float>(K[n * K_dim + k]);
            }
            dQ[m * K_dim + k] = static_cast<half_t>(sum);
        }
    }
}

int main() {
    constexpr int M = 64, N = 64, K_dim = 64;
    std::cout << "\n=== rocWMMA dQ Kernel (FIXED v3) ===\n";
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << std::endl;
    
    size_t size_dS = M * N;
    size_t size_K = N * K_dim;
    size_t size_dQ = M * K_dim;
    
    std::vector<half_t> h_dS(size_dS), h_K(size_K), h_dQ_gpu(size_dQ), h_dQ_cpu(size_dQ);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_dS) v = static_cast<half_t>(dist(gen));
    for (auto& v : h_K)  v = static_cast<half_t>(dist(gen));
    
    half_t *d_dS, *d_K, *d_dQ;
    hipMalloc(&d_dS, size_dS * sizeof(half_t));
    hipMalloc(&d_K,  size_K  * sizeof(half_t));
    hipMalloc(&d_dQ, size_dQ * sizeof(half_t));
    
    hipMemcpy(d_dS, h_dS.data(), size_dS * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_K,  h_K.data(),  size_K  * sizeof(half_t), hipMemcpyHostToDevice);
    
    cpu_dq(h_dS.data(), h_K.data(), h_dQ_cpu.data(), M, N, K_dim);
    
    dim3 grid((K_dim + 15) / 16, (M + 15) / 16);
    dim3 block(32);
    
    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(dq_rocwmma_kernel, grid, block, 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    hipMemcpy(h_dQ_gpu.data(), d_dQ, size_dQ * sizeof(half_t), hipMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_dQ; ++i) {
        float diff = std::abs(static_cast<float>(h_dQ_gpu[i]) - static_cast<float>(h_dQ_cpu[i]));
        if (diff > max_diff) max_diff = diff;
    }
    
    double tflops = (2.0 * M * N * K_dim) / (time_us * 1e-6) / 1e12;
    
    std::cout << "Time: " << time_us << " us\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    std::cout << "Max diff: " << max_diff << "\n";
    
    std::cout << "\nFirst 5 elements:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  GPU=" << static_cast<float>(h_dQ_gpu[i])
                  << " CPU=" << static_cast<float>(h_dQ_cpu[i]) << "\n";
    }
    
    hipFree(d_dS); hipFree(d_K); hipFree(d_dQ);
    return (max_diff < 0.01f) ? 0 : 1;
}

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>
#include "cpu_reference.hpp"

using half_t = _Float16;

// GPU kernel: dS = P * (dP - rowsum(dP * P))
__global__ void softmax_backward_kernel(const half_t* P, const half_t* dP, half_t* dS,
                                        int M, int N) {
    extern __shared__ float s_data[];
    float* s_partial = s_data;
    
    int m = blockIdx.x;
    int tid = threadIdx.x;
    
    if (m >= M) return;
    
    // Каждый поток вычисляет свой кусок rowsum
    float local_sum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float p_val = static_cast<float>(P[m * N + n]);
        float dp_val = static_cast<float>(dP[m * N + n]);
        local_sum += dp_val * p_val;
    }
    
    // Сохраняем в shared memory
    s_partial[tid] = local_sum;
    __syncthreads();
    
    // Редукция в shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }
    
    float rowsum = s_partial[0];
    
    // dS = P * (dP - rowsum)
    for (int n = tid; n < N; n += blockDim.x) {
        float p_val = static_cast<float>(P[m * N + n]);
        float dp_val = static_cast<float>(dP[m * N + n]);
        float ds_val = p_val * (dp_val - rowsum);
        dS[m * N + n] = static_cast<half_t>(ds_val);
    }
}

int main() {
    constexpr int M = 16, N = 16, K_dim = 64;
    
    const size_t size_Q = M * K_dim;
    const size_t size_K = N * K_dim;
    const size_t size_V = N * K_dim;
    const size_t size_O = M * K_dim;
    const size_t size_P = M * N;
    
    std::cout << "\n=== GPU Softmax Backward Test ===\n" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << std::endl;
    
    // Host buffers
    std::vector<half_t> h_Q(size_Q), h_K(size_K), h_V(size_V);
    std::vector<half_t> h_dO(size_O);
    std::vector<half_t> h_P_half(size_P), h_dP_half(size_P);
    std::vector<half_t> h_dS_gpu(size_P), h_dS_cpu(size_P);
    
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
    
    // CPU: dP = dO @ V^T
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
    
    // Конвертируем в half
    for (size_t i = 0; i < size_P; ++i) {
        h_P_half[i] = half_cast(h_P_float[i]);
        h_dP_half[i] = half_cast(h_dP_float[i]);
    }
    
    // CPU: Softmax backward
    std::vector<float> h_dS_float(size_P);
    softmax_backward(h_P_float.data(), h_dP_float.data(), h_dS_float.data(), M, N);
    for (size_t i = 0; i < size_P; ++i) {
        h_dS_cpu[i] = half_cast(h_dS_float[i]);
    }
    
    // GPU buffers
    half_t *d_P, *d_dP, *d_dS;
    hipMalloc(&d_P,  size_P * sizeof(half_t));
    hipMalloc(&d_dP, size_P * sizeof(half_t));
    hipMalloc(&d_dS, size_P * sizeof(half_t));
    
    hipMemcpy(d_P,  h_P_half.data(),  size_P * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dP, h_dP_half.data(), size_P * sizeof(half_t), hipMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 32;  // Используем 32 потока (один warp) на блок
    size_t smem_size = block_size * sizeof(float);
    hipLaunchKernelGGL(softmax_backward_kernel, dim3(M), dim3(block_size), smem_size, 0, 
                       d_P, d_dP, d_dS, M, N);
    hipDeviceSynchronize();
    
    hipMemcpy(h_dS_gpu.data(), d_dS, size_P * sizeof(half_t), hipMemcpyDeviceToHost);
    
    // Verify
    bool ok = true;
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_P; ++i) {
        float gpu_val = float_cast(h_dS_gpu[i]);
        float cpu_val = float_cast(h_dS_cpu[i]);
        float diff = std::abs(gpu_val - cpu_val);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-2f) {
            std::cerr << "Mismatch at " << i << ": GPU=" << gpu_val 
                      << " CPU=" << cpu_val << " diff=" << diff << std::endl;
            ok = false;
        }
    }
    
    std::cout << "dS max_diff: " << max_diff << std::endl;
    
    // Проверяем свойство softmax backward: сумма по строке должна быть ~0
    bool rowsum_ok = true;
    for (int m = 0; m < M; ++m) {
        float rowsum = 0.0f;
        for (int n = 0; n < N; ++n) {
            rowsum += float_cast(h_dS_gpu[m * N + n]);
        }
        if (std::abs(rowsum) > 1e-4f) {
            std::cerr << "Row " << m << " sum = " << rowsum << std::endl;
            rowsum_ok = false;
        }
    }
    
    std::cout << "dS rowsum check: " << (rowsum_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Test " << (ok ? "PASSED" : "FAILED") << std::endl;
    
    if (ok) {
        std::cout << "\nSample values:" << std::endl;
        std::cout << "dS[0,0] = " << float_cast(h_dS_gpu[0]) << std::endl;
        std::cout << "dS[8,8] = " << float_cast(h_dS_gpu[8 * N + 8]) << std::endl;
    }
    
    hipFree(d_P); hipFree(d_dP); hipFree(d_dS);
    return ok ? 0 : 1;
}

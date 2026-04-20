#pragma once
#include <hip/hip_runtime.h>

using half_t = _Float16;

// Правильное GPU ядро: dV = P^T @ dO
__global__ void fmha_bwd_dv_kernel(const half_t* P, const half_t* dO, half_t* dV,
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

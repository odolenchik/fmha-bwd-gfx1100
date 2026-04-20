#pragma once
#include <hip/hip_runtime.h>

using half_t = _Float16;

// Правильное GPU ядро: dK = dS^T @ Q
__global__ void fmha_bwd_dk_kernel(const half_t* dS, const half_t* Q, half_t* dK,
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

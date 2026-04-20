#pragma once
#include <hip/hip_runtime.h>

using half_t = _Float16;

// Правильное GPU ядро: dQ = dS @ K
__global__ void fmha_bwd_dq_kernel(const half_t* dS, const half_t* K, half_t* dQ,
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

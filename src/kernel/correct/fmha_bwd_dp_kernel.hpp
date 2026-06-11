#pragma once
#include <hip/hip_runtime.h>
#include "fmha_bwd_config.h"

using half_t = _Float16;

// GPU kernel: dP = dO @ V^T
__global__ void fmha_bwd_dp_kernel(const half_t* dO, const half_t* V, half_t* dP,
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

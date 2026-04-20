#pragma once
#include <hip/hip_runtime.h>

using half_t = _Float16;

// GPU kernel: dS = P * (dP - rowsum(dP * P))
__global__ void fmha_bwd_softmax_kernel(const half_t* P, const half_t* dP, half_t* dS,
                                        int M, int N) {
    extern __shared__ float s_data[];
    float* s_partial = s_data;
    
    int m = blockIdx.x;
    int tid = threadIdx.x;
    
    if (m >= M) return;
    
    float local_sum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float p_val = static_cast<float>(P[m * N + n]);
        float dp_val = static_cast<float>(dP[m * N + n]);
        local_sum += dp_val * p_val;
    }
    
    s_partial[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }
    
    float rowsum = s_partial[0];
    
    for (int n = tid; n < N; n += blockDim.x) {
        float p_val = static_cast<float>(P[m * N + n]);
        float dp_val = static_cast<float>(dP[m * N + n]);
        float ds_val = p_val * (dp_val - rowsum);
        dS[m * N + n] = static_cast<half_t>(ds_val);
    }
}

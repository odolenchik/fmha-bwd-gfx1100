#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cmath>

using half_t = _Float16;
using bhalf_t = rocwmma::bfloat16_t;

__global__ void fmha_bwd_dk_kernel_naive(const bhalf_t* dS, const bhalf_t* Q, bhalf_t* dK,
    int M_global, int N_global, int K_dim_global) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_global * K_dim_global;
    if (idx >= total) return;

    int n = idx / K_dim_global;
    int k = idx % K_dim_global;

    float sum = 0.0f;
    for (int m = 0; m < M_global; ++m) {
        sum += static_cast<float>(dS[m * N_global + n]) *
               static_cast<float>(Q[m * K_dim_global + k]);
    }
    dK[n * K_dim_global + k] = static_cast<bhalf_t>(sum);
}

int main() {
    const int M = 64, N = 64, K_dim = 32;

    bhalf_t h_dS[M*N], h_Q[M*K_dim];
    float f_dk_ref[N*K_dim];

    for (int i = 0; i < M*N; ++i) {
        h_dS[i] = static_cast<bhalf_t>(static_cast<float>(i) * 0.1f);
    }
    for (int i = 0; i < M*K_dim; ++i) {
        float val = static_cast<float>((i % 7)) * 0.2f - 1.0f;
        h_Q[i] = static_cast<bhalf_t>(val);
    }

    // CPU ref: dK[n][k] = sum_m dS[m][n] * Q[m][k]
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                sum += static_cast<float>(h_dS[m * N + n]) *
                       static_cast<float>(h_Q[m * K_dim + k]);
            }
            f_dk_ref[n * K_dim + k] = sum;
        }
    }

    bhalf_t *d_dS, *d_Q, *d_dK;
    hipMalloc(&d_dS, M*N*sizeof(bhalf_t));
    hipMalloc(&d_Q, M*K_dim*sizeof(bhalf_t));
    hipMalloc(&d_dK, N*K_dim*sizeof(bhalf_t));

    hipMemcpy(d_dS, h_dS, M*N*sizeof(bhalf_t), hipMemcpyHostToDevice);
    hipMemcpy(d_Q, h_Q, M*K_dim*sizeof(bhalf_t), hipMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N * K_dim + threads - 1) / threads;
    hipLaunchKernelGGL(fmha_bwd_dk_kernel_naive, dim3(blocks), dim3(threads),
                       0, 0, d_dS, d_Q, d_dK, M, N, K_dim);
    hipDeviceSynchronize();

    bhalf_t h_result[N*K_dim];
    hipMemcpy(h_result, d_dK, N*K_dim*sizeof(bhalf_t), hipMemcpyDeviceToHost);

    float max_err = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < N * K_dim && mismatches < 16; ++i) {
        float gpu_val = static_cast<float>(h_result[i]);
        float ref_val = f_dk_ref[i];
        float err = fabsf(gpu_val - ref_val);
        max_err = fmaxf(max_err, err);
        if (err > 2.0f) {
            int n = i / K_dim, k = i % K_dim;
            printf("MISMATCH dK[%d][%d]: gpu=%.4f cpu=%.4f err=%.4f\n",
                   n, k, gpu_val, ref_val, err);
            mismatches++;
        }
    }

    int total_mismatches = 0;
    for (int i = 0; i < N * K_dim; ++i) {
        if (fabsf(static_cast<float>(h_result[i]) - f_dk_ref[i]) > 2.0f) total_mismatches++;
    }

    printf("\nNaive dK kernel: %d / %d mismatches, max_err=%.4f\n", total_mismatches, N*K_dim, max_err);
    return (total_mismatches == 0) ? 0 : 1;
}

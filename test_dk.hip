#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cmath>
#include "src/kernel/correct/fmha_bwd_dk_kernel.hpp"

using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

// CPU reference: dK = dS^T @ Q (naive)
void cpu_ref_dk(const float* dS, const float* Q, float* dK, int M, int N, int K_dim) {
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                sum += dS[m * N + n] * Q[m * K_dim + k];
            }
            dK[n * K_dim + k] = sum;
        }
    }
}

int main() {
    const int M = 64, N = 64, K_dim = 32;
    
    // Host data: dS[M x N], Q[M x K_dim], dK[N x K_dim]
    bhalf_t h_dS[M*N], h_Q[M*K_dim], h_dK_gpu[N*K_dim], h_dk_ref[N*K_dim];
    float f_dS[M*N], f_Q[M*K_dim], f_dk_ref[N*K_dim];
    
    // Initialize: dS = 1.0, Q = -0.5 -> expected dK[n][k] = M * 1.0 * (-0.5) = -32.0
    for (int i = 0; i < M*N; ++i) {
        h_dS[i] = static_cast<bhalf_t>(1.0f);
        f_dS[i] = 1.0f;
    }
    for (int i = 0; i < M*K_dim; ++i) {
        h_Q[i] = static_cast<bhalf_t>(-0.5f);
        f_Q[i] = -0.5f;
    }
    
    // CPU reference
    cpu_ref_dk(f_dS, f_Q, f_dk_ref, M, N, K_dim);
    
    // GPU
    bhalf_t *d_dS, *d_Q, *d_dK;
    hipMalloc(&d_dS, M*N*sizeof(bhalf_t));
    hipMalloc(&d_Q, M*K_dim*sizeof(bhalf_t));
    hipMalloc(&d_dK, N*K_dim*sizeof(bhalf_t));
    
    hipMemcpy(d_dS, h_dS, M*N*sizeof(bhalf_t), hipMemcpyHostToDevice);
    hipMemcpy(d_Q, h_Q, M*K_dim*sizeof(bhalf_t), hipMemcpyHostToDevice);
    
    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK);
    hipLaunchKernelGGL(fmha_bwd_dk_kernel_wmma, grid, dim3(BLOCK_SIZE), 0, 0, d_dS, d_Q, d_dK, M, N, K_dim);
    hipDeviceSynchronize();
    
    hipMemcpy(h_dK_gpu, d_dK, N*K_dim*sizeof(bhalf_t), hipMemcpyDeviceToHost);
    
    // Compare
    float max_err = 0.0f;
    bool all_close = true;
    for (int i = 0; i < N * K_dim; ++i) {
        float gpu_val = static_cast<float>(h_dK_gpu[i]);
        float ref_val = f_dk_ref[i];
        float err = fabsf(gpu_val - ref_val);
        max_err = fmaxf(max_err, err);
        if (err > 1.0f) {
            all_close = false;
            int n = i / K_dim, k = i % K_dim;
            printf("MISMATCH dK[%d][%d]: gpu=%.4f ref=%.4f err=%.4f\n", 
                   n, k, gpu_val, ref_val, err);
        }
    }
    
    printf("\n=== dK WMMA Kernel Test ===\n");
    printf("dS: all 1.0, Q: all -0.5, M=%d -> expected dK=%.1f\n", M, f_dk_ref[0]);
    printf("GPU dK[0..3]: %.4f %.4f %.4f %.4f\n", 
           static_cast<float>(h_dK_gpu[0]), static_cast<float>(h_dK_gpu[1]),
           static_cast<float>(h_dK_gpu[2]), static_cast<float>(h_dK_gpu[3]));
    printf("CPU ref dK[0..3]: %.4f %.4f %.4f %.4f\n", 
           f_dk_ref[0], f_dk_ref[1], f_dk_ref[2], f_dk_ref[3]);
    printf("Max error: %.6f\n", max_err);
    printf("Result: %s\n", all_close ? "✅ PASS" : "❌ FAIL");
    
    hipFree(d_dS); hipFree(d_Q); hipFree(d_dK);
    
    return all_close ? 0 : 1;
}

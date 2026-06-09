#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cmath>
#include "src/kernel/correct/fmha_bwd_dq_kernel.hpp"

using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

// CPU reference: dQ = dS @ K (naive)
void cpu_ref_dq(const float* dS, const float* K, float* dQ, int M, int N, int K_dim) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                sum += dS[m * N + n] * K[n * K_dim + k];
            }
            dQ[m * K_dim + k] = sum;
        }
    }
}

int main() {
    const int M = 64, N = 64, K_dim = 32;
    
    // Allocate host memory
    bhalf_t h_dS[M*N], h_K[N*K_dim], h_dQ_gpu[M*K_dim], h_dQ_ref[M*K_dim];
    float f_dS[M*N], f_K[N*K_dim], f_dQ_ref[M*K_dim];
    
    // Initialize: dS = -0.5, K = 1.0 -> expected dQ[m][k] = N * (-0.5) * 1.0 = -32.0
    for (int i = 0; i < M*N; ++i) {
        h_dS[i] = static_cast<bhalf_t>(-0.5f);
        f_dS[i] = -0.5f;
    }
    for (int i = 0; i < N*K_dim; ++i) {
        h_K[i] = static_cast<bhalf_t>(1.0f);
        f_K[i] = 1.0f;
    }
    
    // CPU reference
    cpu_ref_dq(f_dS, f_K, f_dQ_ref, M, N, K_dim);
    
    // GPU
    bhalf_t *d_dS, *d_K, *d_dQ;
    hipMalloc(&d_dS, M*N*sizeof(bhalf_t));
    hipMalloc(&d_K, N*K_dim*sizeof(bhalf_t));
    hipMalloc(&d_dQ, M*K_dim*sizeof(bhalf_t));
    
    hipMemcpy(d_dS, h_dS, M*N*sizeof(bhalf_t), hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K, N*K_dim*sizeof(bhalf_t), hipMemcpyHostToDevice);
    
    dim3 grid((K_dim + BN - 1) / BN, (M + BM - 1) / BM);
    hipLaunchKernelGGL(fmha_bwd_dq_kernel_wmma, grid, dim3(BLOCK_SIZE), 0, 0, d_dS, d_K, d_dQ, M, N, K_dim);
    hipDeviceSynchronize();
    
    hipMemcpy(h_dQ_gpu, d_dQ, M*K_dim*sizeof(bhalf_t), hipMemcpyDeviceToHost);
    
    // Compare
    float max_err = 0.0f;
    bool all_close = true;
    for (int i = 0; i < M * K_dim; ++i) {
        float gpu_val = static_cast<float>(h_dQ_gpu[i]);
        float ref_val = f_dQ_ref[i];
        float err = fabsf(gpu_val - ref_val);
        max_err = fmaxf(max_err, err);
        if (err > 1.0f) { // bfloat16 tolerance ~1 ULP for values around 32
            all_close = false;
            int m = i / K_dim, k = i % K_dim;
            printf("MISMATCH dQ[%d][%d]: gpu=%.4f ref=%.4f err=%.4f\n", 
                   m, k, gpu_val, ref_val, err);
        }
    }
    
    printf("\n=== dQ WMMA Kernel Test ===\n");
    printf("dS: all -0.5, K: all 1.0, N=%d -> expected dQ=%.1f\n", N, static_cast<float>(h_dQ_gpu[0]));
    printf("GPU dQ[0..3]: %.4f %.4f %.4f %.4f\n", 
           static_cast<float>(h_dQ_gpu[0]), static_cast<float>(h_dQ_gpu[1]),
           static_cast<float>(h_dQ_gpu[2]), static_cast<float>(h_dQ_gpu[3]));
    printf("CPU ref dQ[0..3]: %.4f %.4f %.4f %.4f\n", 
           f_dQ_ref[0], f_dQ_ref[1], f_dQ_ref[2], f_dQ_ref[3]);
    printf("Max error: %.6f\n", max_err);
    printf("Result: %s\n", all_close ? "✅ PASS" : "❌ FAIL");
    
    hipFree(d_dS); hipFree(d_K); hipFree(d_dQ);
    
    return all_close ? 0 : 1;
}

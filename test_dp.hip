#include <hip/hip_runtime.h>
#include "src/kernel/fmha_bwd_dp_kernel.hpp"
#include <cstdio>

using bhalf_t = rocwmma::bfloat16_t;

int main() {
    const int M = 64, N = 64, K_dim = 32;

    // Host data: dO[M x K_dim], V[N x K_dim] → dP[M x N]
    bhalf_t h_dO[M * K_dim], h_V[N * K_dim], h_dP[M * N];
    for (int i = 0; i < M * K_dim; i++) h_dO[i] = (bhalf_t)(1.0f);
    for (int i = 0; i < N * K_dim; i++) h_V[i] = (bhalf_t)(1.0f);

    // Expected: dP[m][n] = sum_k(1.0 * 1.0) = K_dim = 32
    for (int i = 0; i < M * N; i++) h_dP[i] = (bhalf_t)(static_cast<float>(K_dim));

    // Device data
    bhalf_t *d_dO, *d_V, *d_dP;
    hipMalloc(&d_dO, M * K_dim * sizeof(bhalf_t));
    hipMalloc(&d_V, N * K_dim * sizeof(bhalf_t));
    hipMalloc(&d_dP, M * N * sizeof(bhalf_t));

    hipMemcpy(d_dO, h_dO, M * K_dim * sizeof(bhalf_t), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V, N * K_dim * sizeof(bhalf_t), hipMemcpyHostToDevice);
    hipMemset(d_dP, 0, M * N * sizeof(bhalf_t));

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    hipLaunchKernelGGL(fmha_bwd_dp_kernel, grid, dim3(BLOCK_SIZE), 0, 0, d_dO, d_V, d_dP, M, N, K_dim);
    hipDeviceSynchronize();

    bhalf_t h_result[M * N];
    hipMemcpy(h_result, d_dP, M * N * sizeof(bhalf_t), hipMemcpyDeviceToHost);

    // Check correctness
    bool ok = true;
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        float expected = static_cast<float>(K_dim);
        float actual = static_cast<float>(h_result[i]);
        if (std::abs(actual - expected) > 0.5f) {
            if (errors < 5) {
                printf("ERROR at %d: got %.1f, expected %.1f\n", i, actual, expected);
            }
            errors++;
            ok = false;
        }
    }

    if (ok) {
        printf("PASS: dP kernel correct! All values = %.1f (expected %d)\n",
               static_cast<float>(h_result[0]), K_dim);
    } else {
        printf("FAIL: %d errors\n", errors);
    }

    hipFree(d_dO); hipFree(d_V); hipFree(d_dP);
    return ok ? 0 : 1;
}

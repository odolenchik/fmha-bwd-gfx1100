#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <iostream>

__global__ void simple_dp_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* V, __nv_bfloat16* dP,
                          int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m < M && n < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float do_val = __bfloat162float(dO[m * K + k]);
            float v_val  = __bfloat162float(V[n * K + k]);
            sum += do_val * v_val;
        }
        dP[m * N + n] = __float2bfloat16_rn(sum);
    }
}

int main() {
    const int M = 4, N = 4, K = 4;
    size_t dO_size = M * K * sizeof(__nv_bfloat16);
    size_t V_size  = N * K * sizeof(__nv_bfloat16);
    size_t dP_size = M * N * sizeof(__nv_bfloat16);
    
    __nv_bfloat16 *dO_d, *V_d, *dP_d;
    hipMalloc(&dO_d, dO_size);
    hipMalloc(&V_d, V_size);
    hipMalloc(&dP_d, dP_size);
    
    // Simple test data
    __nv_bfloat16 dO_h[16] = {__float2bfloat16_rn(1.0), __float2bfloat16_rn(2.0), __float2bfloat16_rn(3.0), __float2bfloat16_rn(4.0),
                              __float2bfloat16_rn(1.0), __float2bfloat16_rn(2.0), __float2bfloat16_rn(3.0), __float2bfloat16_rn(4.0),
                              __float2bfloat16_rn(1.0), __float2bfloat16_rn(2.0), __float2bfloat16_rn(3.0), __float2bfloat16_rn(4.0),
                              __float2bfloat16_rn(1.0), __float2bfloat16_rn(2.0), __float2bfloat16_rn(3.0), __float2bfloat16_rn(4.0)};
    __nv_bfloat16 V_h[16]  = {__float2bfloat16_rn(1.0), __float2bfloat16_rn(1.0), __float2bfloat16_rn(1.0), __float2bfloat16_rn(1.0),
                              __float2bfloat16_rn(2.0), __float2bfloat16_rn(2.0), __float2bfloat16_rn(2.0), __float2bfloat16_rn(2.0),
                              __float2bfloat16_rn(3.0), __float2bfloat16_rn(3.0), __float2bfloat16_rn(3.0), __float2bfloat16_rn(3.0),
                              __float2bfloat16_rn(4.0), __float2bfloat16_rn(4.0), __float2bfloat16_rn(4.0), __float2bfloat16_rn(4.0)};
                     
    hipMemcpy(dO_d, dO_h, dO_size, hipMemcpyHostToDevice);
    hipMemcpy(V_d,  V_h,  V_size,  hipMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    simple_dp_kernel<<<grid, block>>>(dO_d, V_d, dP_d, M, N, K);
    hipDeviceSynchronize();
    
    __nv_bfloat16 dP_h[16];
    hipMemcpy(dP_h, dP_d, dP_size, hipMemcpyDeviceToHost);
    
    // Expected: dP[m,n] = sum_k dO[m,k] * V[n,k]
    // For our test data: dO[m,k] = m+1, V[n,k] = n+1
    // So dP[m,n] = sum_k (m+1)*(n+1) = K * (m+1)*(n+1) = 4 * (m+1)*(n+1)
    bool pass = true;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float expected = 4.0f * (m+1) * (n+1);
            float actual = __bfloat162float(dP_h[m * N + n]);
            if (fabsf(expected - actual) > 0.01f) {
                printf("FAIL: dP[%d,%d] = %f, expected %f\n", m, n, actual, expected);
                pass = false;
            }
        }
    }
    
    if (pass) printf("PASS\n");
    
    hipFree(dO_d);
    hipFree(V_d);
    hipFree(dP_d);
    return 0;
}

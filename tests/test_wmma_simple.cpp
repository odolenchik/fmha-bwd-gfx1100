#include <iostream>
#include <hip/hip_runtime.h>
#include "ck/utility/dtype_vector.hpp"

using half_t = _Float16;
using namespace ck;

__global__ void test_wmma_kernel(half_t* output) {
    half16_t a, b;
    float8_t c;
    
    for (int i = 0; i < 16; i++) {
        reinterpret_cast<half_t*>(&a)[i] = static_cast<half_t>(i + 1);
        reinterpret_cast<half_t*>(&b)[i] = static_cast<half_t>(1.0f);
    }
    
    for (int i = 0; i < 8; i++) {
        reinterpret_cast<float*>(&c)[i] = 0.0f;
    }
    
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            output[i] = static_cast<half_t>(reinterpret_cast<float*>(&c)[i]);
        }
    }
}

int main() {
    half_t* d_out;
    half_t h_out[8] = {0};
    
    hipMalloc(&d_out, 8 * sizeof(half_t));
    hipLaunchKernelGGL(test_wmma_kernel, dim3(1), dim3(32), 0, 0, d_out);
    hipDeviceSynchronize();
    hipMemcpy(h_out, d_out, 8 * sizeof(half_t), hipMemcpyDeviceToHost);
    
    std::cout << "WMMA result (first 8 floats): ";
    for (int i = 0; i < 8; i++) {
        std::cout << static_cast<float>(h_out[i]) << " ";
    }
    std::cout << std::endl;
    
    float expected = 136.0f;  // sum of 1..16
    float actual = static_cast<float>(h_out[0]);
    std::cout << "Expected ~" << expected << ", got " << actual << std::endl;
    
    hipFree(d_out);
    return (std::abs(actual - expected) < 1.0f) ? 0 : 1;
}

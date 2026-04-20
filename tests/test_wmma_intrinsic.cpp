#include <iostream>
#include <hip/hip_runtime.h>

// Используем заголовки CK для векторных типов
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"

using half_t = _Float16;

__global__ void test_wmma_kernel(half_t* output) {
    // Векторные типы из CK
    using half16_t = typename ck_tile::vector_type<half_t, 16>::type;
    using float8_t = typename ck_tile::vector_type<float, 8>::type;
    
    // Определяем векторы (16 half)
    half16_t a;
    half16_t b;
    float8_t c;
    
    // Заполняем A значениями 1..16
    for (int i = 0; i < 16; i++) {
        reinterpret_cast<half_t*>(&a)[i] = static_cast<half_t>(i + 1);
        reinterpret_cast<half_t*>(&b)[i] = static_cast<half_t>(1.0f);
    }
    
    // Обнуляем C
    for (int i = 0; i < 8; i++) {
        reinterpret_cast<float*>(&c)[i] = 0.0f;
    }
    
    // Выполняем WMMA: C = A * B + C
#if defined(__gfx1100__) || defined(__gfx11__)
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
    
    // Сохраняем результат (только первый поток для проверки)
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
    
    // Проверяем ожидаемые значения
    // Для 16x16x16 WMMA, при A=1..16 и B=1, первая строка результата должна быть суммой
    float expected_first = 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16; // 136
    std::cout << "Expected first element ~136, got " << static_cast<float>(h_out[0]) << std::endl;
    
    hipFree(d_out);
    return 0;
}

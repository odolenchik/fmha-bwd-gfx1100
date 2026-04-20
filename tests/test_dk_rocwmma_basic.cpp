#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

__global__ void dk_rocwmma_basic_kernel(
    const half_t* __restrict__ dS,
    const half_t* __restrict__ Q,
    half_t* __restrict__ dK,
    int M, int N, int K_dim)
{
    // Для dK = dS^T @ Q, используем col_major для A
    using FragA = fragment<matrix_a, 16, 16, 16, half_t, col_major>;  // dS^T - col_major
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;  // Q - row_major
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, half_t>;

    int warp_n = blockIdx.y;
    int warp_k = blockIdx.x;
    int n_start = warp_n * 16;
    int k_start = warp_k * 16;

    FragC acc_frag;
    fill_fragment(acc_frag, 0.0f);

    for (int m_start = 0; m_start < M; m_start += 16) {
        FragA a_frag;
        FragB b_frag;

        // Загружаем dS^T используя col_major
        // dS^T[n, m] = dS[m, n], leading dimension = N
        load_matrix_sync(a_frag, dS + m_start * N + n_start, N);

        // Загружаем Q row-major
        load_matrix_sync(b_frag, Q + m_start * K_dim + k_start, K_dim);

        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (n_start < N && k_start < K_dim) {
        FragOut out_frag;
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            out_frag.x[i] = static_cast<half_t>(acc_frag.x[i]);
        }
        store_matrix_sync(dK + n_start * K_dim + k_start, out_frag, K_dim, mem_row_major);
    }
}

void cpu_dk(const half_t* dS, const half_t* Q, half_t* dK, int M, int N, int K_dim) {
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                sum += static_cast<float>(dS[m * N + n]) * static_cast<float>(Q[m * K_dim + k]);
            }
            dK[n * K_dim + k] = static_cast<half_t>(sum);
        }
    }
}

int main() {
    constexpr int M = 64, N = 64, K_dim = 64;
    std::cout << "\n=== rocWMMA dK (col_major for dS^T) ===\n";
    
    size_t size_dS = M * N;
    size_t size_Q = M * K_dim;
    size_t size_dK = N * K_dim;
    
    std::vector<half_t> h_dS(size_dS), h_Q(size_Q), h_dK_gpu(size_dK), h_dK_cpu(size_dK);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_dS) v = static_cast<half_t>(dist(gen));
    for (auto& v : h_Q) v = static_cast<half_t>(dist(gen));
    
    half_t *d_dS, *d_Q, *d_dK;
    hipMalloc(&d_dS, size_dS * sizeof(half_t));
    hipMalloc(&d_Q,  size_Q  * sizeof(half_t));
    hipMalloc(&d_dK, size_dK * sizeof(half_t));
    
    hipMemcpy(d_dS, h_dS.data(), size_dS * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_Q,  h_Q.data(),  size_Q  * sizeof(half_t), hipMemcpyHostToDevice);
    
    cpu_dk(h_dS.data(), h_Q.data(), h_dK_cpu.data(), M, N, K_dim);
    
    dim3 grid((K_dim + 15) / 16, (N + 15) / 16);
    dim3 block(32);
    
    hipLaunchKernelGGL(dk_rocwmma_basic_kernel, grid, block, 0, 0, d_dS, d_Q, d_dK, M, N, K_dim);
    hipDeviceSynchronize();
    
    hipMemcpy(h_dK_gpu.data(), d_dK, size_dK * sizeof(half_t), hipMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < size_dK; ++i) {
        float diff = std::abs(static_cast<float>(h_dK_gpu[i]) - static_cast<float>(h_dK_cpu[i]));
        if (diff > max_diff) max_diff = diff;
    }
    
    std::cout << "Max diff: " << max_diff << "\n";
    std::cout << "First 5 elements:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  GPU=" << static_cast<float>(h_dK_gpu[i])
                  << " CPU=" << static_cast<float>(h_dK_cpu[i]) << "\n";
    }
    
    hipFree(d_dS); hipFree(d_Q); hipFree(d_dK);
    return (max_diff < 0.01f) ? 0 : 1;
}

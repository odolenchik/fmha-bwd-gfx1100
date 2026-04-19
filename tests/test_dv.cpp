#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_bwd_dv_kernel.hpp"

using namespace ck_tile;
using half_t = _Float16;

// CPU reference for dV (simplified: P = Q @ K^T without softmax)
void cpu_reference_dv(const half_t* Q, const half_t* K, const half_t* dO, half_t* dV,
                      index_t M, index_t N, index_t K_dim)
{
    // Сначала вычислим P = Q @ K^T
    std::vector<float> P(M * N, 0.0f);
    for(index_t m = 0; m < M; ++m) {
        for(index_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for(index_t k = 0; k < K_dim; ++k) {
                sum += static_cast<float>(Q[m * K_dim + k]) *
                       static_cast<float>(K[n * K_dim + k]);
            }
            P[m * N + n] = sum;
        }
    }
    
    // dV = P^T @ dO
    for(index_t n = 0; n < N; ++n) {
        for(index_t k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for(index_t m = 0; m < M; ++m) {
                sum += P[m * N + n] * static_cast<float>(dO[m * K_dim + k]);
            }
            dV[n * K_dim + k] = static_cast<half_t>(sum);
        }
    }
}

bool verify(const half_t* gpu, const half_t* cpu, size_t size, float eps = 1e-1f)
{
    for (size_t i = 0; i < size; ++i) {
        float g = static_cast<float>(gpu[i]);
        float c = static_cast<float>(cpu[i]);
        if (std::abs(g - c) > eps) {
            std::cerr << "Mismatch at " << i << ": GPU=" << g << " CPU=" << c << std::endl;
            return false;
        }
    }
    return true;
}

struct FmhaBwdProblem
{
    using QDataType = half_t;
    using KDataType = half_t;
    using VDataType = half_t;
    using ODataType = half_t;
    using AccDataType = float;
    static constexpr bool kIsGroupMode = false;
    static constexpr bool kPadHeadDimQ = true;
    static constexpr bool kPadHeadDimV = true;
    static constexpr auto kHeadDim = 64;
};

int main()
{
    constexpr index_t B = 1, H = 1;
    constexpr index_t M = 16, N = 16, K_dim = 64;

    const size_t size_Q  = B * H * M * K_dim;
    const size_t size_K  = B * H * N * K_dim;
    const size_t size_dO = B * H * M * K_dim;
    const size_t size_dV = B * H * N * K_dim;

    std::vector<half_t> h_Q(size_Q), h_K(size_K), h_dO(size_dO), h_dV_cpu(size_dV), h_dV_gpu(size_dV);
    std::vector<half_t> h_P(M * N);  // P matrix

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_Q)  v = static_cast<half_t>(dist(gen));
    for (auto& v : h_K)  v = static_cast<half_t>(dist(gen));
    for (auto& v : h_dO) v = static_cast<half_t>(dist(gen));

    // Вычислим P = Q @ K^T на CPU и скопируем на GPU
    for(index_t m = 0; m < M; ++m) {
        for(index_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for(index_t k = 0; k < K_dim; ++k) {
                sum += static_cast<float>(h_Q[m * K_dim + k]) *
                       static_cast<float>(h_K[n * K_dim + k]);
            }
            h_P[m * N + n] = static_cast<half_t>(sum);
        }
    }

    half_t *d_P, *d_dO, *d_dV;
    hipMalloc(&d_P,  M * N * sizeof(half_t));
    hipMalloc(&d_dO, size_dO * sizeof(half_t));
    hipMalloc(&d_dV, size_dV * sizeof(half_t));

    hipMemcpy(d_P,  h_P.data(),  M * N * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO.data(), size_dO * sizeof(half_t), hipMemcpyHostToDevice);

    cpu_reference_dv(h_Q.data(), h_K.data(), h_dO.data(), h_dV_cpu.data(), M, N, K_dim);

    using Kernel = FmhaBwdDVKernel<FmhaBwdProblem>;
    typename Kernel::Kargs kargs;
    kargs.P_ptr = d_P;
    kargs.dO_ptr = d_dO;
    kargs.dV_ptr = d_dV;
    kargs.batch_count = B;
    kargs.num_heads = H;
    kargs.seq_len_q = M;
    kargs.seq_len_k = N;
    kargs.hidden_dim = K_dim;

    constexpr int block_size = 256;
    int grid_size = (N * K_dim + block_size - 1) / block_size;
    hipLaunchKernelGGL(Kernel::Run, dim3(grid_size), dim3(block_size), 0, 0, kargs);
    hipDeviceSynchronize();

    hipMemcpy(h_dV_gpu.data(), d_dV, size_dV * sizeof(half_t), hipMemcpyDeviceToHost);

    bool ok = verify(h_dV_gpu.data(), h_dV_cpu.data(), size_dV);
    std::cout << (ok ? "dV Test PASSED" : "dV Test FAILED") << std::endl;

    hipFree(d_P); hipFree(d_dO); hipFree(d_dV);
    return ok ? 0 : 1;
}

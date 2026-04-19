#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_bwd_dk_kernel.hpp"

using namespace ck_tile;
using half_t = _Float16;

// CPU reference for dK
void cpu_reference_dk(const half_t* Q, const half_t* dO, half_t* dK,
                      index_t M, index_t N, index_t K_dim)
{
    for (index_t n = 0; n < N; ++n) {
        for (index_t k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (index_t m = 0; m < M; ++m) {
                sum += static_cast<float>(Q[m * K_dim + k]) *
                       static_cast<float>(dO[m * K_dim + k]);
            }
            dK[n * K_dim + k] = static_cast<half_t>(sum);
        }
    }
}

bool verify(const half_t* gpu, const half_t* cpu, size_t size, float eps = 1e-2f)
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
    const size_t size_dO = B * H * M * K_dim;
    const size_t size_dK = B * H * N * K_dim;

    std::vector<half_t> h_Q(size_Q), h_dO(size_dO), h_dK_cpu(size_dK), h_dK_gpu(size_dK);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_Q)  v = static_cast<half_t>(dist(gen));
    for (auto& v : h_dO) v = static_cast<half_t>(dist(gen));

    half_t *d_Q, *d_dO, *d_dK;
    hipMalloc(&d_Q,  size_Q  * sizeof(half_t));
    hipMalloc(&d_dO, size_dO * sizeof(half_t));
    hipMalloc(&d_dK, size_dK * sizeof(half_t));

    hipMemcpy(d_Q,  h_Q.data(),  size_Q  * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO.data(), size_dO * sizeof(half_t), hipMemcpyHostToDevice);

    cpu_reference_dk(h_Q.data(), h_dO.data(), h_dK_cpu.data(), M, N, K_dim);

    using Kernel = FmhaBwdDKKernel<FmhaBwdProblem>;
    typename Kernel::Kargs kargs;
    kargs.Q_ptr = d_Q;
    kargs.dO_ptr = d_dO;
    kargs.dK_ptr = d_dK;
    kargs.batch_count = B;
    kargs.num_heads = H;
    kargs.seq_len_q = M;
    kargs.seq_len_k = N;
    kargs.hidden_dim = K_dim;

    constexpr int block_size = 256;
    int grid_size = (N * K_dim + block_size - 1) / block_size;
    hipLaunchKernelGGL(Kernel::Run, dim3(grid_size), dim3(block_size), 0, 0, kargs);
    hipDeviceSynchronize();

    hipMemcpy(h_dK_gpu.data(), d_dK, size_dK * sizeof(half_t), hipMemcpyDeviceToHost);

    bool ok = verify(h_dK_gpu.data(), h_dK_cpu.data(), size_dK);
    std::cout << (ok ? "dK Test PASSED" : "dK Test FAILED") << std::endl;

    hipFree(d_Q); hipFree(d_dO); hipFree(d_dK);
    return ok ? 0 : 1;
}

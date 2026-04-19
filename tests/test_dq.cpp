#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_bwd_dq_kernel.hpp"

using namespace ck_tile;
using half_t = _Float16;

// ----------------------------------------------------------------------------
// CPU reference (naive)
// ----------------------------------------------------------------------------
void cpu_reference_dq(const half_t* dO, const half_t* K, half_t* dQ,
                      index_t M, index_t N, index_t K_dim)
{
    for (index_t m = 0; m < M; ++m) {
        for (index_t k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (index_t n = 0; n < N; ++n) {
                sum += static_cast<float>(dO[m * K_dim + k]) *
                       static_cast<float>(K[n * K_dim + k]);
            }
            dQ[m * K_dim + k] = static_cast<half_t>(sum);
        }
    }
}

// ----------------------------------------------------------------------------
// Verification
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Problem definition (должен совпадать с тем, что в kernel)
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Main test
// ----------------------------------------------------------------------------
int main()
{
    constexpr index_t B = 1, H = 1;
    constexpr index_t M = 16, N = 16, K_dim = 64;  // минимальные размеры для WMMA 16x16x16

    const size_t size_dO = B * H * M * K_dim;
    const size_t size_K  = B * H * N * K_dim;
    const size_t size_dQ = B * H * M * K_dim;

    // Host buffers
    std::vector<half_t> h_dO(size_dO), h_K(size_K), h_dQ_cpu(size_dQ), h_dQ_gpu(size_dQ);

    // Random init
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_dO) v = static_cast<half_t>(dist(gen));
    for (auto& v : h_K)  v = static_cast<half_t>(dist(gen));

    // Device buffers
    half_t *d_dO, *d_K, *d_dQ;
    hipMalloc(&d_dO, size_dO * sizeof(half_t));
    hipMalloc(&d_K,  size_K  * sizeof(half_t));
    hipMalloc(&d_dQ, size_dQ * sizeof(half_t));

    hipMemcpy(d_dO, h_dO.data(), size_dO * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_K,  h_K.data(),  size_K  * sizeof(half_t), hipMemcpyHostToDevice);

    // CPU reference
    cpu_reference_dq(h_dO.data(), h_K.data(), h_dQ_cpu.data(), M, N, K_dim);

    // Setup kernel arguments
    using Kernel = FmhaBwdDQKernel<FmhaBwdProblem>;
    typename Kernel::Kargs kargs;
    kargs.dO_ptr = d_dO;
    kargs.K_ptr  = d_K;
    kargs.dQ_ptr = d_dQ;
    kargs.batch_count = B;
    kargs.num_heads = H;
    kargs.seq_len_q = M;
    kargs.seq_len_k = N;
    kargs.hidden_dim = K_dim;
    // strides (row-major, без батча/голов упрощённо)
    kargs.dO_stride_b = M * K_dim;
    kargs.dO_stride_h = M * K_dim;
    kargs.dO_stride_m = K_dim;
    kargs.dO_stride_k = 1;
    kargs.K_stride_b  = N * K_dim;
    kargs.K_stride_h  = N * K_dim;
    kargs.K_stride_n  = K_dim;
    kargs.K_stride_k  = 1;
    kargs.dQ_stride_b = M * K_dim;
    kargs.dQ_stride_h = M * K_dim;
    kargs.dQ_stride_m = K_dim;
    kargs.dQ_stride_k = 1;

    // Launch kernel (1 блок, т.к. M,N=16)
    constexpr int block_size = 256;
    hipLaunchKernelGGL(Kernel::Run, dim3((M * K_dim + block_size - 1) / block_size), dim3(block_size), 0, 0, kargs);
    hipDeviceSynchronize();

    // Copy back
    hipMemcpy(h_dQ_gpu.data(), d_dQ, size_dQ * sizeof(half_t), hipMemcpyDeviceToHost);

    // Verify
    bool ok = verify(h_dQ_gpu.data(), h_dQ_cpu.data(), size_dQ);
    std::cout << (ok ? "Test PASSED" : "Test FAILED") << std::endl;

    // Cleanup
    hipFree(d_dO); hipFree(d_K); hipFree(d_dQ);
    return ok ? 0 : 1;
}

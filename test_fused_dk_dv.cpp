#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

constexpr int BM = 64, BN = 64, BK = 32;
constexpr int WARP_SIZE = 64;
constexpr int N_WAVES = 8;
constexpr int BLOCK_SIZE = WARP_SIZE * N_WAVES;
constexpr int LDS_PAD = 8;

__global__ void fused_dk_dv_kernel(
    const half_t* __restrict__ Q,
    const half_t* __restrict__ dS,
    const half_t* __restrict__ P,
    const half_t* __restrict__ dO,
    half_t* __restrict__ dK,
    half_t* __restrict__ dV,
    int M, int N, int K_dim)
{
    __shared__ half_t s_Q[BM][BK + LDS_PAD];
    __shared__ half_t s_dO[BM][BK + LDS_PAD];
    __shared__ half_t s_dST[BN][BM + LDS_PAD];
    __shared__ half_t s_PT[BN][BM + LDS_PAD];

    using FragAT = fragment<matrix_a, 16, 16, 16, half_t, col_major>;
    using FragB  = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC  = fragment<accumulator, 16, 16, 16, float>;
    using FragOut= fragment<accumulator, 16, 16, 16, half_t>;

    int block_n = blockIdx.y, block_k = blockIdx.x;
    int n_start = block_n * BM, k_start = block_k * BN;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int warp_n = warp_id / 4;
    int warp_k = warp_id % 4;
    int w_n_start = n_start + warp_n * 16, w_k_start = k_start + warp_k * 16;

    // Загружаем Q и dO (нужны оба)
    for (int i = tid; i < BM * BK; i += blockDim.x) {
        int m = i / BK, k = i % BK;
        if (m < M && w_k_start + k < K_dim) {
            s_Q[m][k] = Q[m * K_dim + (w_k_start + k)];
            s_dO[m][k] = dO[m * K_dim + (w_k_start + k)];
        }
    }

    // Транспонируем dS и P
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        if (m < M && w_n_start + n < N) {
            s_dST[n][m] = dS[m * N + (w_n_start + n)];
            s_PT[n][m] = P[m * N + (w_n_start + n)];
        }
    }
    __syncthreads();

    // ---------- dK = dS^T @ Q ----------
    if (w_n_start < N && w_k_start < K_dim) {
        FragC acc_dK[2];
        fill_fragment(acc_dK[0], 0.0f);
        fill_fragment(acc_dK[1], 0.0f);

        for (int m_start = 0; m_start < M; m_start += BK) {
            __syncthreads();
            for (int sub = 0; sub < 2; ++sub) {
                FragAT a0, a1; FragB b0, b1;
                half_t* a_ptr0 = reinterpret_cast<half_t*>(&a0);
                half_t* b_ptr0 = reinterpret_cast<half_t*>(&b0);
                half_t* a_ptr1 = reinterpret_cast<half_t*>(&a1);
                half_t* b_ptr1 = reinterpret_cast<half_t*>(&b1);
                
                int m_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m_start + m_off + j < BM && warp_n * 16 + i < BN) ? s_dST[warp_n * 16 + i][m_start + m_off + j] : half_t(0);
                        a_ptr0[i*16 + j] = val;
                        a_ptr1[i*16 + j] = val;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m_start + m_off + i < BM && warp_k * 16 + j < BK) ? s_Q[m_start + m_off + i][warp_k * 16 + j] : half_t(0);
                        b_ptr0[i*16 + j] = val;
                        b_ptr1[i*16 + j] = val;
                    }
                }
                mma_sync(acc_dK[0], a0, b0, acc_dK[0]);
                mma_sync(acc_dK[1], a1, b1, acc_dK[1]);
            }
        }

        for (int i = 0; i < acc_dK[0].num_elements; ++i)
            acc_dK[0].x[i] += acc_dK[1].x[i];
        FragOut out;
        for (int i = 0; i < acc_dK[0].num_elements; ++i) out.x[i] = static_cast<half_t>(acc_dK[0].x[i]);
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (n < N && k < K_dim) dK[n * K_dim + k] = out.x[i*16+j];
            }
        }
    }

    // ---------- dV = P^T @ dO ----------
    if (w_n_start < N && w_k_start < K_dim) {
        FragC acc_dV[2];
        fill_fragment(acc_dV[0], 0.0f);
        fill_fragment(acc_dV[1], 0.0f);

        for (int m_start = 0; m_start < M; m_start += BK) {
            __syncthreads();
            for (int sub = 0; sub < 2; ++sub) {
                FragAT a0, a1; FragB b0, b1;
                half_t* a_ptr0 = reinterpret_cast<half_t*>(&a0);
                half_t* b_ptr0 = reinterpret_cast<half_t*>(&b0);
                half_t* a_ptr1 = reinterpret_cast<half_t*>(&a1);
                half_t* b_ptr1 = reinterpret_cast<half_t*>(&b1);
                
                int m_off = sub * 16;
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m_start + m_off + j < BM && warp_n * 16 + i < BN) ? s_PT[warp_n * 16 + i][m_start + m_off + j] : half_t(0);
                        a_ptr0[i*16 + j] = val;
                        a_ptr1[i*16 + j] = val;
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        half_t val = (m_start + m_off + i < BM && warp_k * 16 + j < BK) ? s_dO[m_start + m_off + i][warp_k * 16 + j] : half_t(0);
                        b_ptr0[i*16 + j] = val;
                        b_ptr1[i*16 + j] = val;
                    }
                }
                mma_sync(acc_dV[0], a0, b0, acc_dV[0]);
                mma_sync(acc_dV[1], a1, b1, acc_dV[1]);
            }
        }

        for (int i = 0; i < acc_dV[0].num_elements; ++i)
            acc_dV[0].x[i] += acc_dV[1].x[i];
        FragOut out;
        for (int i = 0; i < acc_dV[0].num_elements; ++i) out.x[i] = static_cast<half_t>(acc_dV[0].x[i]);
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (n < N && k < K_dim) dV[n * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

int main() {
    constexpr int M = 1024, N = 1024, K_dim = 64;
    std::cout << "\n=== FUSED dK + dV KERNEL ===\n";

    size_t size_Q = M*K_dim, size_dS = M*N, size_P = M*N, size_dO = M*K_dim;
    size_t size_dK = N*K_dim, size_dV = N*K_dim;

    std::vector<half_t> h_Q(size_Q), h_dS(size_dS), h_P(size_P), h_dO(size_dO);
    std::vector<half_t> h_dK_gpu(size_dK), h_dV_gpu(size_dV);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_Q) v = (half_t)dist(gen);
    for (auto& v : h_dS) v = (half_t)dist(gen);
    for (auto& v : h_P) v = (half_t)std::abs(dist(gen));
    for (auto& v : h_dO) v = (half_t)dist(gen);

    half_t *d_Q, *d_dS, *d_P, *d_dO, *d_dK, *d_dV;
    hipMalloc(&d_Q, size_Q*sizeof(half_t)); hipMalloc(&d_dS, size_dS*sizeof(half_t));
    hipMalloc(&d_P, size_P*sizeof(half_t)); hipMalloc(&d_dO, size_dO*sizeof(half_t));
    hipMalloc(&d_dK, size_dK*sizeof(half_t)); hipMalloc(&d_dV, size_dV*sizeof(half_t));

    hipMemcpy(d_Q, h_Q.data(), size_Q*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dS, h_dS.data(), size_dS*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_P, h_P.data(), size_P*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO.data(), size_dO*sizeof(half_t), hipMemcpyHostToDevice);

    dim3 grid((K_dim + BN - 1) / BN, (N + BM - 1) / BM);
    dim3 block(BLOCK_SIZE);

    hipLaunchKernelGGL(fused_dk_dv_kernel, grid, block, 0, 0,
                       d_Q, d_dS, d_P, d_dO, d_dK, d_dV, M, N, K_dim);
    hipDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(fused_dk_dv_kernel, grid, block, 0, 0,
                       d_Q, d_dS, d_P, d_dO, d_dK, d_dV, M, N, K_dim);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    hipMemcpy(h_dK_gpu.data(), d_dK, size_dK*sizeof(half_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_dV_gpu.data(), d_dV, size_dV*sizeof(half_t), hipMemcpyDeviceToHost);

    bool has_nan = false;
    for (size_t i = 0; i < size_dK; ++i) if (!std::isfinite((float)h_dK_gpu[i])) { has_nan = true; break; }
    for (size_t i = 0; i < size_dV; ++i) if (!std::isfinite((float)h_dV_gpu[i])) { has_nan = true; break; }

    std::cout << "Time: " << time_us << " us\n";
    std::cout << "NaN check: " << (has_nan ? "FAIL" : "PASS") << "\n";

    hipFree(d_Q); hipFree(d_dS); hipFree(d_P); hipFree(d_dO); hipFree(d_dK); hipFree(d_dV);
    return 0;
}

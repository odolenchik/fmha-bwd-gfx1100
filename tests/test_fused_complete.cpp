#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

constexpr int BM = 64, BN = 64, BK = 16;
constexpr int WARP_SIZE = 32;

__global__ void flash_attn_bwd_fused_complete(
    const half_t* __restrict__ Q,
    const half_t* __restrict__ K,
    const half_t* __restrict__ V,
    const half_t* __restrict__ P,
    const half_t* __restrict__ dO,
    half_t* __restrict__ dQ,
    half_t* __restrict__ dK,
    half_t* __restrict__ dV,
    int M, int N, int K_dim)
{
    // ---------- LDS память ----------
    __shared__ half_t s_K[BN][BK];
    __shared__ half_t s_V[BN][BK];
    __shared__ half_t s_VT[BK][BN];
    __shared__ half_t s_Q[BM][BK];
    __shared__ half_t s_dO[BM][BK];
    __shared__ half_t s_P[BM][BN];
    __shared__ half_t s_dP[BM][BN];
    __shared__ half_t s_dS[BM][BN];
    __shared__ half_t s_dST[BN][BM];
    __shared__ half_t s_PT[BN][BM];
    __shared__ float s_rowsum[BM];

    // ---------- WMMA фрагменты ----------
    using FragA  = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB  = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC  = fragment<accumulator, 16, 16, 16, float>;
    using FragOut= fragment<accumulator, 16, 16, 16, half_t>;

    // ---------- Индексы ----------
    int block_m = blockIdx.y, block_n = blockIdx.x;
    int m_start = block_m * BM, n_start = block_n * BN;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int warp_m = warp_id / 4;
    int warp_n = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_n_start = n_start + warp_n * 16;

    // ---------- 1. Pre-load K, V ----------
    for (int i = tid; i < BN * BK; i += blockDim.x) {
        int n = i / BK, k = i % BK;
        int gn = n_start + n;
        s_K[n][k] = (gn < N && k < K_dim) ? K[gn * K_dim + k] : half_t(0);
        s_V[n][k] = (gn < N && k < K_dim) ? V[gn * K_dim + k] : half_t(0);
    }
    __syncthreads();

    // ---------- 2. Load Q, dO, P ----------
    for (int i = tid; i < BM * BK; i += blockDim.x) {
        int m = i / BK, k = i % BK;
        int gm = m_start + m;
        s_Q[m][k]  = (gm < M && k < K_dim) ? Q[gm * K_dim + k] : half_t(0);
        s_dO[m][k] = (gm < M && k < K_dim) ? dO[gm * K_dim + k] : half_t(0);
    }
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        int gm = m_start + m, gn = n_start + n;
        s_P[m][n] = (gm < M && gn < N) ? P[gm * N + gn] : half_t(0);
    }
    __syncthreads();

    // ---------- 3. dP = dO @ V^T ----------
    if (w_m_start < M && w_n_start < N) {
        FragC acc;
        fill_fragment(acc, 0.0f);

        for (int k_block = 0; k_block < K_dim; k_block += BK) {
            for (int i = tid; i < BN * BK; i += blockDim.x) {
                int n = i / BK, k = i % BK;
                int gn = n_start + n, gk = k_block + k;
                s_VT[k][n] = (gn < N && gk < K_dim) ? V[gn * K_dim + gk] : half_t(0);
            }
            __syncthreads();

            FragA a; FragB b;
            half_t* a_ptr = reinterpret_cast<half_t*>(&a);
            half_t* b_ptr = reinterpret_cast<half_t*>(&b);
            
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                for (int j = 0; j < 16; ++j) {
                    a_ptr[i*16 + j] = (m < BM && j < BK) ? s_dO[m][k_block + j] : half_t(0);
                }
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int n = warp_n * 16 + j;
                    b_ptr[i*16 + j] = (i < BK && n < BN) ? s_VT[i][n] : half_t(0);
                }
            }
            mma_sync(acc, a, b, acc);
            __syncthreads();
        }

        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int m = warp_m * 16 + i;
            for (int j = 0; j < 16; ++j) {
                int n = warp_n * 16 + j;
                if (m < BM && n < BN) s_dP[m][n] = out.x[i*16+j];
            }
        }
    }
    __syncthreads();

    // ---------- 4. Softmax backward ----------
    if (tid < BM) {
        int gm = m_start + tid;
        if (gm < M) {
            float sum = 0.0f;
            for (int n = 0; n < BN; ++n)
                sum += (float)s_P[tid][n] * (float)s_dP[tid][n];
            s_rowsum[tid] = sum;
        }
    }
    __syncthreads();

    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        int gm = m_start + m, gn = n_start + n;
        if (gm < M && gn < N) {
            float p  = s_P[m][n];
            float dp = s_dP[m][n];
            float rs = s_rowsum[m];
            s_dS[m][n] = static_cast<half_t>(p * (dp - rs));
        }
    }
    __syncthreads();

    // Транспонируем dS и P для dK и dV
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int m = i / BN, n = i % BN;
        s_dST[n][m] = s_dS[m][n];
        s_PT[n][m] = s_P[m][n];
    }
    __syncthreads();

    // ---------- 5. dQ = dS @ K ----------
    if (w_m_start < M && w_n_start * 16 < K_dim) {
        FragC acc;
        fill_fragment(acc, 0.0f);
        for (int k = 0; k < N; k += BK) {
            FragA a; FragB b;
            half_t* a_ptr = reinterpret_cast<half_t*>(&a);
            half_t* b_ptr = reinterpret_cast<half_t*>(&b);
            
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                for (int j = 0; j < 16; ++j) {
                    a_ptr[i*16 + j] = (m < BM && k + j < BN) ? s_dS[m][k + j] : half_t(0);
                }
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    b_ptr[i*16 + j] = (k + i < BN && warp_n * 16 + j < BK) ? s_K[k + i][warp_n * 16 + j] : half_t(0);
                }
            }
            mma_sync(acc, a, b, acc);
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = warp_n * 16 + j;
                if (m < M && k < K_dim) dQ[m * K_dim + k] = out.x[i*16+j];
            }
        }
    }

    // ---------- 6. dK = dS^T @ Q ----------
    if (w_n_start < N && warp_m * 16 < K_dim) {
        FragC acc;
        fill_fragment(acc, 0.0f);
        for (int m = 0; m < M; m += BK) {
            FragA a; FragB b;
            half_t* a_ptr = reinterpret_cast<half_t*>(&a);
            half_t* b_ptr = reinterpret_cast<half_t*>(&b);
            
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    a_ptr[i*16 + j] = (m + j < BM && warp_n * 16 + i < BN) ? s_dST[warp_n * 16 + i][m + j] : half_t(0);
                }
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    b_ptr[i*16 + j] = (m + i < BM && warp_m * 16 + j < BK) ? s_Q[m + i][warp_m * 16 + j] : half_t(0);
                }
            }
            mma_sync(acc, a, b, acc);
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = warp_m * 16 + j;
                if (n < N && k < K_dim) dK[n * K_dim + k] = out.x[i*16+j];
            }
        }
    }

    // ---------- 7. dV = P^T @ dO ----------
    if (w_n_start < N && warp_m * 16 < K_dim) {
        FragC acc;
        fill_fragment(acc, 0.0f);
        for (int m = 0; m < M; m += BK) {
            FragA a; FragB b;
            half_t* a_ptr = reinterpret_cast<half_t*>(&a);
            half_t* b_ptr = reinterpret_cast<half_t*>(&b);
            
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    a_ptr[i*16 + j] = (m + j < BM && warp_n * 16 + i < BN) ? s_PT[warp_n * 16 + i][m + j] : half_t(0);
                }
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    b_ptr[i*16 + j] = (m + i < BM && warp_m * 16 + j < BK) ? s_dO[m + i][warp_m * 16 + j] : half_t(0);
                }
            }
            mma_sync(acc, a, b, acc);
        }
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int n = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = warp_m * 16 + j;
                if (n < N && k < K_dim) dV[n * K_dim + k] = out.x[i*16+j];
            }
        }
    }
}

// ------------------------------------------------------------------
// CPU reference
// ------------------------------------------------------------------
void cpu_reference(
    const half_t* Q, const half_t* K, const half_t* V,
    const half_t* P, const half_t* dO,
    half_t* dQ, half_t* dK, half_t* dV,
    int M, int N, int K_dim)
{
    std::vector<float> dP(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K_dim; ++k)
                sum += (float)dO[m * K_dim + k] * (float)V[n * K_dim + k];
            dP[m * N + n] = sum;
        }

    std::vector<float> rowsum(M, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            rowsum[m] += (float)P[m * N + n] * dP[m * N + n];

    std::vector<float> dS(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float p = P[m * N + n];
            dS[m * N + n] = p * (dP[m * N + n] - rowsum[m]);
        }

    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n)
                sum += dS[m * N + n] * (float)K[n * K_dim + k];
            dQ[m * K_dim + k] = (half_t)sum;
        }

    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m)
                sum += dS[m * N + n] * (float)Q[m * K_dim + k];
            dK[n * K_dim + k] = (half_t)sum;
        }

    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m)
                sum += (float)P[m * N + n] * (float)dO[m * K_dim + k];
            dV[n * K_dim + k] = (half_t)sum;
        }
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
int main() {
    constexpr int M = 256, N = 256, K_dim = 64;
    std::cout << "\n=== FUSED FLASH ATTENTION BACKWARD (COMPLETE) ===\n";
    std::cout << "M=" << M << ", N=" << N << ", K_dim=" << K_dim << "\n";

    size_t size_Q = M * K_dim, size_K = N * K_dim, size_V = N * K_dim, size_P = M * N;
    std::vector<half_t> h_Q(size_Q), h_K(size_K), h_V(size_V), h_P(size_P), h_dO(size_Q);
    std::vector<half_t> h_dQ_gpu(size_Q), h_dK_gpu(size_K), h_dV_gpu(size_V);
    std::vector<half_t> h_dQ_cpu(size_Q), h_dK_cpu(size_K), h_dV_cpu(size_V);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pos(0.1f, 1.0f);
    for (auto& v : h_Q) v = (half_t)dist(gen);
    for (auto& v : h_K) v = (half_t)dist(gen);
    for (auto& v : h_V) v = (half_t)dist(gen);
    for (auto& v : h_dO) v = (half_t)dist(gen);
    for (auto& v : h_P) v = (half_t)dist_pos(gen);

    for (int m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) sum += (float)h_P[m * N + n];
        for (int n = 0; n < N; ++n) h_P[m * N + n] = (half_t)((float)h_P[m * N + n] / sum);
    }

    half_t *d_Q, *d_K, *d_V, *d_P, *d_dO, *d_dQ, *d_dK, *d_dV;
    hipMalloc(&d_Q, size_Q*sizeof(half_t)); hipMalloc(&d_K, size_K*sizeof(half_t));
    hipMalloc(&d_V, size_V*sizeof(half_t)); hipMalloc(&d_P, size_P*sizeof(half_t));
    hipMalloc(&d_dO, size_Q*sizeof(half_t)); hipMalloc(&d_dQ, size_Q*sizeof(half_t));
    hipMalloc(&d_dK, size_K*sizeof(half_t)); hipMalloc(&d_dV, size_V*sizeof(half_t));

    hipMemcpy(d_Q, h_Q.data(), size_Q*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K.data(), size_K*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V.data(), size_V*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_P, h_P.data(), size_P*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO.data(), size_Q*sizeof(half_t), hipMemcpyHostToDevice);

    cpu_reference(h_Q.data(), h_K.data(), h_V.data(), h_P.data(), h_dO.data(),
                  h_dQ_cpu.data(), h_dK_cpu.data(), h_dV_cpu.data(), M, N, K_dim);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);

    hipLaunchKernelGGL(flash_attn_bwd_fused_complete, grid, block, 0, 0,
                       d_Q, d_K, d_V, d_P, d_dO, d_dQ, d_dK, d_dV, M, N, K_dim);
    hipDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(flash_attn_bwd_fused_complete, grid, block, 0, 0,
                       d_Q, d_K, d_V, d_P, d_dO, d_dQ, d_dK, d_dV, M, N, K_dim);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    hipMemcpy(h_dQ_gpu.data(), d_dQ, size_Q*sizeof(half_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_dK_gpu.data(), d_dK, size_K*sizeof(half_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_dV_gpu.data(), d_dV, size_V*sizeof(half_t), hipMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (size_t i = 0; i < size_Q; ++i) max_diff = std::max(max_diff, std::abs((float)h_dQ_gpu[i] - (float)h_dQ_cpu[i]));
    for (size_t i = 0; i < size_K; ++i) max_diff = std::max(max_diff, std::abs((float)h_dK_gpu[i] - (float)h_dK_cpu[i]));
    for (size_t i = 0; i < size_V; ++i) max_diff = std::max(max_diff, std::abs((float)h_dV_gpu[i] - (float)h_dV_cpu[i]));

    std::cout << "Time: " << time_us << " us\n";
    std::cout << "Max diff: " << max_diff << "\n";
    std::cout << (max_diff < 1.0f ? "✅ TEST PASSED\n" : "❌ TEST FAILED\n");

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_P); hipFree(d_dO);
    hipFree(d_dQ); hipFree(d_dK); hipFree(d_dV);
    return (max_diff < 1.0f) ? 0 : 1;
}

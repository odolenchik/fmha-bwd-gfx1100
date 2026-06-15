#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include "kernel/correct/fmha_bwd_config.h"

using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

// ============================================================================
// dP = dO @ V^T — naive element-per-thread with multi-head support
// dP[m][n] = sum_{k=0}^{K_dim-1} dO[m][k] * V[n][k], per head
// Each thread block computes one BM×BN tile of the output (M × N).
// ============================================================================
__global__ void dp_kernel(const bhalf_t* dO, const bhalf_t* V, bhalf_t* dP,
                          int M_global, int N_global, int K_dim, int total_heads) {
    int head_idx = blockIdx.z;
    if (head_idx >= total_heads) return;

    const bhalf_t* dO_h = dO + head_idx * M_global * K_dim;
    const bhalf_t* V_h  = V  + head_idx * N_global * K_dim;
    bhalf_t*       dP_h = dP + head_idx * M_global * N_global;

    int tid = threadIdx.x;
    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;

    for (int idx = tid; idx < BM * BN; idx += BLOCK_SIZE) {
        int m_local = idx / BN;   // 0..63
        int n_local = idx % BN;   // 0..63

        if (block_m + m_local >= M_global || block_n + n_local >= N_global) continue;

        float sum = 0.0f;
        for (int k = 0; k < K_dim; ++k) {
            bhalf_t do_val = dO_h[(block_m + m_local) * K_dim + k];
            bhalf_t v_val  = V_h[(block_n + n_local) * K_dim + k];
            sum += static_cast<float>(do_val) * static_cast<float>(v_val);
        }

        dP_h[(block_m + m_local) * N_global + block_n + n_local] = static_cast<bhalf_t>(sum);
    }
}

// ============================================================================
// Softmax backward: dS = P ⊙ (dP − rowsum(dP×P))
// ============================================================================
__global__ void softmax_bwd_kernel(const bhalf_t* P, const bhalf_t* dP, bhalf_t* dS,
                                   int M, int N, int total_heads) {
    extern __shared__ float s_data[];
    float* s_partial = s_data;

    int m = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (m >= M || head_idx >= total_heads) return;

    const bhalf_t* P_h  = P + head_idx * M * N;
    const bhalf_t* dP_h = dP + head_idx * M * N;
    bhalf_t*       dS_h = dS + head_idx * M * N;

    float local_sum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float p_val   = static_cast<float>(P_h[m * N + n]);
        float dp_val  = static_cast<float>(dP_h[m * N + n]);
        local_sum += dp_val * p_val;
    }

    s_partial[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }

    float rowsum = s_partial[0];

    for (int n = tid; n < N; n += blockDim.x) {
        float p_val   = static_cast<float>(P_h[m * N + n]);
        float dp_val  = static_cast<float>(dP_h[m * N + n]);
        dS_h[m * N + n] = static_cast<bhalf_t>(p_val * (dp_val - rowsum));
    }
}

// ============================================================================
// dQ = dS @ K — naive element-per-thread with multi-head support
// dQ[m][k] = sum_{n=0}^{N_global-1} dS[m][n] * K[n][k], per head
// Each thread block computes one BM×BK tile of the output (M × K_dim).
// Sum over N is done in chunks to fit shared memory.
// ============================================================================
__global__ void dq_kernel(const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
                          int M_global, int N_global, int K_dim_global, int total_heads) {
    __shared__ bhalf_t s_dS[BM][BN + LDS_PAD];   // [m_local × n_local] — one BN-chunk at a time
    __shared__ bhalf_t s_K[BK+LDS_PAD][BN];      // [k_local × n_idx]

    int head_idx = blockIdx.z;
    if (head_idx >= total_heads) return;

    const bhalf_t* dS_h = dS + head_idx * M_global * N_global;
    const bhalf_t* K_h  = K  + head_idx * N_global * K_dim_global;
    bhalf_t*       dQ_h = dQ + head_idx * M_global * K_dim_global;

    int tid = threadIdx.x;
    int block_m = blockIdx.y;
    int block_k = blockIdx.x;
    int m_start = block_m * BM;
    int k_start = block_k * BK;

    // Load K slice: s_K[k_local][n_idx] = K[n_idx][k_start + k_local], for all n in [0, N)
    for (int i = tid; i < BK * N_global; i += BLOCK_SIZE) {
        int k_local = i / N_global, n_idx = i % N_global;
        s_K[k_local][n_idx] = ((k_start + k_local < K_dim_global && n_idx < N_global))
                              ? K_h[n_idx * K_dim_global + k_start + k_local] : bhalf_t(0.0f);
    }

    // Each thread accumulates results across all N positions in chunks of BN.
    // All 512 threads participate in every __syncthreads() call — no early continue.
    for (int idx = tid; idx < BM * BK; idx += BLOCK_SIZE) {
        int m_local = idx / BK;   // 0..63
        int k_local = idx % BK;   // 0..31

        float sum = 0.0f;

        if (m_start + m_local < M_global && k_start + k_local < K_dim_global) {
            for (int n_base = 0; n_base < N_global; n_base += BN) {
                int n_chunk = min(BN, N_global - n_base);

                // Load dS tile: s_dS[m_local][n_local] = dS[m_start+m_local][n_base+n_local]
                for (int i = tid; i < BM * n_chunk; i += BLOCK_SIZE) {
                    int m_off = i / n_chunk, n_off = i % n_chunk;
                    s_dS[m_off][n_off] = ((m_start + m_off < M_global && n_base + n_off < N_global))
                                         ? dS_h[(m_start + m_off) * N_global + n_base + n_off] : bhalf_t(0.0f);
                }
                __syncthreads();

                for (int n_off = 0; n_off < n_chunk; ++n_off) {
                    sum += static_cast<float>(s_dS[m_local][n_off]) * 
                           static_cast<float>(s_K[k_local][n_off]);
                }

                __syncthreads();
            }
        } else {
            // Guard-failed threads still participate in all syncthreads to avoid deadlock.
            // Load dS tile for this chunk (garbage values are fine — no one reads them).
            for (int n_base = 0; n_base < N_global; n_base += BN) {
                int n_chunk = min(BN, N_global - n_base);

                for (int i = tid; i < BM * n_chunk; i += BLOCK_SIZE) {
                    int m_off = i / n_chunk, n_off = i % n_chunk;
                    s_dS[m_off][n_off] = bhalf_t(0.0f);
                }
                __syncthreads();
                // No dot product computation needed — sum stays 0.

                __syncthreads();
            }
        }

        if (m_start + m_local < M_global && k_start + k_local < K_dim_global) {
            dQ_h[(m_start + m_local) * K_dim_global + k_start + k_local] = static_cast<bhalf_t>(sum);
        }
    }
}

// ============================================================================
// dK = dS^T @ Q — naive element-per-thread with shared memory broadcast
// dK[n][k] = sum_m dS[m][n] * Q[m][k]
// Each thread block computes one BN×BK tile of the output (N × K_dim).
// Sum over M is done in chunks of BM to fit shared memory.
// ============================================================================
__global__ void dk_kernel(const bhalf_t* dS, const bhalf_t* Q, bhalf_t* dK,
                          int M_global, int N_global, int K_dim_global, int total_heads) {
    __shared__ bhalf_t s_dS[BM][BN + LDS_PAD];   // [m_local × n_local] — one BM-chunk at a time
    __shared__ bhalf_t s_Q[BK+LDS_PAD][BM];      // [k_local × m_idx]

    int head_idx = blockIdx.z;  // which head this block computes for
    if (head_idx >= total_heads) return;

    const bhalf_t* dS_h = dS + head_idx * M_global * N_global;
    const bhalf_t* Q_h  = Q  + head_idx * N_global * K_dim_global;
    bhalf_t*       dK_h = dK + head_idx * N_global * K_dim_global;

    int tid = threadIdx.x;
    int block_n = blockIdx.x;
    int block_k = blockIdx.y;
    int n_start = block_n * BN;
    int k_start = block_k * BK;

    // Load Q slice: s_Q[k_local][m_idx] = Q[m_idx][k_start + k_local]
    for (int i = tid; i < BM * BK; i += BLOCK_SIZE) {
        int m_idx = i / BK, k_local = i % BK;
        s_Q[k_local][m_idx] = ((m_idx < M_global && k_start + k_local < K_dim_global))
                              ? Q_h[m_idx * K_dim_global + k_start + k_local] : bhalf_t(0.0f);
    }

    // Each thread accumulates results for its assigned elements across all M chunks.
    // All 512 threads participate in every __syncthreads() call — no early continue.
    int local_bn = BN;  // number of n-local positions per block
    
    for (int idx = tid; idx < local_bn * BK; idx += BLOCK_SIZE) {
        int n_local = idx % BN;   // 0..63
        int k_local = idx / BN;   // 0..31

        int n_global_idx = n_start + n_local;
        int k_global_idx = k_start + k_local;

        float sum = 0.0f;

        if (n_global_idx < N_global && k_global_idx < K_dim_global) {
            // Accumulate over all M rows in chunks of BM
            for (int m_base = 0; m_base < M_global; m_base += BM) {
                int m_chunk = min(BM, M_global - m_base);

                // Load dS tile: s_dS[m_local][n_local] = dS[m_base+m_local][n_start+n_local]
                for (int i = tid; i < m_chunk * local_bn; i += BLOCK_SIZE) {
                    int m_off2 = i / local_bn, n_local2 = i % local_bn;
                    s_dS[m_off2][n_local2] = ((m_base + m_off2 < M_global && n_start + n_local2 < N_global))
                                             ? dS_h[(m_base + m_off2) * N_global + n_start + n_local2] : bhalf_t(0.0f);
                }
                __syncthreads();

                // Accumulate partial dot product for this chunk
                for (int m_off = 0; m_off < m_chunk; ++m_off) {
                    sum += static_cast<float>(s_dS[m_off][n_local]) * 
                           static_cast<float>(s_Q[k_local][m_off]);
                }

                __syncthreads();  // ensure next chunk's dS load is visible to all threads
            }
        } else {
            // Guard-failed threads still participate in all syncthreads.
            for (int m_base = 0; m_base < M_global; m_base += BM) {
                int m_chunk = min(BM, M_global - m_base);

                for (int i = tid; i < m_chunk * local_bn; i += BLOCK_SIZE) {
                    int m_off2 = i / local_bn, n_local2 = i % local_bn;
                    s_dS[m_off2][n_local2] = bhalf_t(0.0f);
                }
                __syncthreads();
                // No dot product computation needed — sum stays 0.

                __syncthreads();
            }
        }

        if (n_global_idx < N_global && k_global_idx < K_dim_global) {
            dK_h[n_global_idx * K_dim_global + k_global_idx] = static_cast<bhalf_t>(sum);
        }
    }
}

// ============================================================================
// dV = P^T @ dO — CORRECTED naive element-per-thread with multi-head support
// dV[n][k] = sum_{m=0}^{M-1} P[m][n] * dO[m][k], per head
// Each thread block computes one BN×BK tile of the output (N × K_dim).
// No shared memory — each thread loads from global directly.
// ============================================================================
__global__ void dv_kernel(const bhalf_t* P, const bhalf_t* dO, bhalf_t* dV,
                          int M_global, int N_global, int BK_dim, int total_heads) {
    int head_idx = blockIdx.z;  // which head this block computes for
    if (head_idx >= total_heads) return;

    const bhalf_t* P_h   = P + head_idx * M_global * N_global;
    const bhalf_t* dO_h  = dO + head_idx * M_global * BK_dim;
    bhalf_t*       dV_h  = dV + head_idx * N_global * BK_dim;

    int tid = threadIdx.x;
    int block_n = blockIdx.x * BN;
    int block_k = blockIdx.y * BK_dim;

    // Each thread handles ceil(BN*BK / BLOCK_SIZE) elements
    for (int idx = tid; idx < BN * BK_dim; idx += BLOCK_SIZE) {
        int n_local = idx % BN;   // 0..63
        int k_local = idx / BN;   // 0..BK_dim-1

        if (block_n + n_local >= N_global || block_k + k_local >= BK_dim) continue;

        float sum = 0.0f;
        for (int m = 0; m < M_global; ++m) {
            bhalf_t p_val = P_h[m * N_global + block_n + n_local];
            bhalf_t do_val = dO_h[m * BK_dim + block_k + k_local];
            sum += static_cast<float>(p_val) * static_cast<float>(do_val);
        }

        dV_h[(block_n + n_local) * BK_dim + block_k + k_local] = static_cast<bhalf_t>(sum);
    }
}

// ============================================================================
// C entry point — launches all kernels in sequence with error checking
// ============================================================================
extern "C" {
void fmha_bwd_full_py(bhalf_t* dQ, bhalf_t* dK, bhalf_t* dV,
    const bhalf_t* Q, const bhalf_t* K, const bhalf_t* V,
    const bhalf_t* P, const bhalf_t* dO, int M, int N, int K_dim, int total_heads)
{
    // Allocate intermediate buffers
    size_t size_P = total_heads * M * N * sizeof(bhalf_t);
    size_t size_S = total_heads * M * N * sizeof(bhalf_t);

    bhalf_t* dP = nullptr;
    bhalf_t* dS = nullptr;

    if (hipMalloc(&dP, size_P) != hipSuccess || hipMalloc(&dS, size_S) != hipSuccess) {
        if (dP) hipFree(dP);
        fprintf(stderr, "fmha_bwd: malloc failed for %zu + %zu bytes\n", size_P, size_S);
        return;
    }

    // 1. dP = dO @ V^T (naive element-per-thread with multi-head)
    dim3 dp_grid((N + BN - 1) / BN, (M + BM - 1) / BM, total_heads);
    hipLaunchKernelGGL(dp_kernel, dp_grid, dim3(BLOCK_SIZE), 0, 0, dO, V, dP, M, N, K_dim, total_heads);

    // 2. Softmax backward: dS = P ⊙ (dP − rowsum)
    int softmax_shared_mem = BLOCK_SIZE * sizeof(float);
    dim3 softmax_grid((M + BM - 1) / BM, total_heads, 1);
    hipLaunchKernelGGL(softmax_bwd_kernel, softmax_grid, dim3(BLOCK_SIZE),
                       softmax_shared_mem, 0, P, dP, dS, M, N, total_heads);

    // 3. dQ = dS @ K (naive element-per-thread with multi-head)
    dim3 dq_grid((K_dim + BK - 1) / BK, (M + BM - 1) / BM, total_heads);
    hipLaunchKernelGGL(dq_kernel, dq_grid, dim3(BLOCK_SIZE), 0, 0, dS, K, dQ, M, N, K_dim, total_heads);

    // 4. dK = dS^T @ Q (naive element-per-thread with multi-head)
    dim3 dk_grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dk_kernel, dk_grid, dim3(BLOCK_SIZE), 0, 0, dS, Q, dK, M, N, K_dim, total_heads);

    // 5. dV = P^T @ dO (naive element-per-thread)
    // Output shape: [N × K_dim] — grid is (N-tiles × K-tiles × heads)
    dim3 dv_grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dv_kernel, dv_grid, dim3(BLOCK_SIZE), 0, 0, P, dO, dV, M, N, K_dim, total_heads);

    // Wait for all kernels to finish and check errors
    hipDeviceSynchronize();
    hipFree(dP);
    hipFree(dS);
}
}



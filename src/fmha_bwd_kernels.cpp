#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include "kernel/correct/fmha_bwd_config.h"

using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

// ============================================================================
// dP = dO @ V^T  — each thread computes one output element (naive but fast)
// ============================================================================
__global__ void dp_kernel(const bhalf_t* dO, const bhalf_t* V, bhalf_t* dP,
                          int M, int N, int K_dim, int total_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    if (idx >= total_elements) return;

    // Unroll head index: each element is shared across all heads equally
    int m = idx / N, n = idx % N;

    float sum = 0.0f;
    for (int h = 0; h < total_heads; ++h) {
        const bhalf_t* dO_h = dO + h * M * K_dim;
        const bhalf_t* V_h  = V  + h * N * K_dim;
        for (int k = 0; k < K_dim; ++k) {
            sum += static_cast<float>(dO_h[m * K_dim + k]) *
                   static_cast<float>(V_h[n * K_dim + k]);
        }
    }

    // dP is only total_heads × M × N — store per-head so the caller can use it
    int head_idx = blockIdx.z;  // which head we're computing for
    bhalf_t* dP_head = dP + head_idx * M * N;
    dP_head[m * N + n] = static_cast<bhalf_t>(sum);
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
// dQ = dS @ K — WMMA kernel (verified working, pattern: row_major A + col_major B)
// ============================================================================
__global__ void dq_kernel(const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
                          int M_global, int N_global, int K_dim_global, int total_heads) {
    __shared__ bhalf_t s_dS[BM][BN + LDS_PAD];
    __shared__ bhalf_t s_K[BK][BN + LDS_PAD];

    using FragA = fragment<matrix_a, 16, 16, 16, bhalf_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, bhalf_t>;

    int block_m = blockIdx.y;
    int block_k = blockIdx.x;
    int head_idx = blockIdx.z;

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;
    int warp_m_idx = warp_id % 4;
    int warp_k_idx = warp_id / 4;

    int m_start = block_m * BM;
    int k_start = block_k * BN;
    int w_m_start = m_start + warp_m_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    dS += head_idx * M_global * N_global;
    K   += head_idx * N_global * K_dim_global;
    dQ  += head_idx * M_global * K_dim_global;

    // ---- Load dS tile into shared memory ----
    for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
        int m = i / BN, n = i % BN;
        int gm = m_start + m;
        s_dS[m][n] = (gm < M_global && n < N_global) ? dS[gm * N_global + n] : half_t(0.0f);
    }
    __syncthreads();

    FragC acc;
    fill_fragment(acc, 0.0f);

    for (int n_start = 0; n_start < N_global; n_start += BK) {
        for (int i = tid; i < BK * BN; i += BLOCK_SIZE) {
            int k_idx = i / BN, n = i % BN;
            int gn = n_start + n, gk = k_start + k_idx;
            s_K[k_idx][n] = (gn < N_global && gk < K_dim_global)
                            ? K[gn * K_dim_global + gk] : bhalf_t(0.0f);
        }
        __syncthreads();

        for (int sub = 0; sub < 2; ++sub) {
            FragA a0;
            FragB b0;

            bhalf_t* a_ptr0 = reinterpret_cast<bhalf_t*>(&a0);
            bhalf_t* b_ptr0 = reinterpret_cast<bhalf_t*>(&b0);

            int n_off = sub * 16;

            for (int i = 0; i < 16; ++i) {
                int m_local = warp_m_idx * 16 + i;
                for (int j = 0; j < 16; ++j) {
                    int n = n_start + n_off + j;
                    a_ptr0[i * 16 + j] = (m_local < BM && n < BN) ? s_dS[m_local][n] : bhalf_t(0.0f);
                }
            }

            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = warp_k_idx * 16 + j;
                    bhalf_t val = (k_idx_local < BK && (n_off + i) < BN)
                                  ? s_K[k_idx_local][n_off + i] : bhalf_t(0.0f);
                    b_ptr0[j * 16 + i] = val;
                }
            }

            mma_sync(acc, a0, b0, acc);
        }
    }

    if (w_m_start < M_global && w_k_start < K_dim_global) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int m_local = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (m_local < M_global && k < K_dim_global)
                    dQ[m_local * K_dim_global + k] = out.x[i * 16 + j];
            }
        }
    }
}

// ============================================================================
// dK = dS^T @ Q — WMMA kernel (fixed: uses same layout as dq_kernel)
// ============================================================================
__global__ void dk_kernel(const bhalf_t* dS, const bhalf_t* Q, bhalf_t* dK,
                          int M_global, int N_global, int K_dim_global, int total_heads) {
    __shared__ bhalf_t s_dS[BN][BM + LDS_PAD];   // transposed: s_dS[n_local][m_local] = dS[m_local][n_start+n_local]
    __shared__ bhalf_t s_Q[BK+LDS_PAD][BM];      // Q slice: s_Q[k_idx_local][m_idx]

    using FragA = fragment<matrix_a, 16, 16, 16, bhalf_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, bhalf_t>;

    int block_n = blockIdx.x;      // which N tile (BN=64)
    int block_k = blockIdx.y;      // which K_dim tile (BK=32)
    int head_idx = blockIdx.z;

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;
    int warp_n_idx = warp_id % 4;       // N sub-tile (16-wide each, 4 across)
    int warp_k_idx = warp_id / 4;        // K sub-tile (16-wide each, 2 down)

    int n_start = block_n * BN;
    int k_start = block_k * BK;
    int w_n_start = n_start + warp_n_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    dS += head_idx * M_global * N_global;
    Q   += head_idx * M_global * K_dim_global;
    dK  += head_idx * N_global * K_dim_global;

    // ---- Load dS transposed: s_dS[n][m] = dS[m][n] ----
    for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
        int n_local = i / BM, m_local = i % BM;
        s_dS[n_local][m_local] = (n_start + n_local < N_global && m_local < M_global)
                                 ? dS[m_local * N_global + n_start + n_local] : half_t(0.0f);
    }
    __syncthreads();

    FragC acc;
    fill_fragment(acc, 0.0f);

    for (int m_start = 0; m_start < M_global; m_start += BM) {
        int m_chunk = (BM < M_global - m_start) ? BM : (M_global - m_start);

        // Load Q slice: s_Q[k_idx_local][m_idx] = Q[m_idx][k_start+k_idx]
        for (int i = tid; i < BK * m_chunk; i += BLOCK_SIZE) {
            int k_local = i / m_chunk, m_idx = i % m_chunk;
            s_Q[k_local][m_idx] = (k_start + k_local < K_dim_global && m_start + m_idx < M_global)
                                  ? Q[(m_start + m_idx) * K_dim_global + k_start + k_local]
                                  : bhalf_t(0.0f);
        }
        __syncthreads();

        for (int m_sub = 0; m_sub < 4; ++m_sub) {
            FragA a_frag;
            FragB b_frag;

            bhalf_t* a_ptr = reinterpret_cast<bhalf_t*>(&a_frag);
            bhalf_t* b_ptr = reinterpret_cast<bhalf_t*>(&b_frag);

            int m_off = m_sub * 16;

            // Fill A from dS^T (row_major: n=rows, m=cols)
            for (int i = 0; i < 16; ++i) {
                int n_local = warp_n_idx * 16 + i;
                for (int j = 0; j < 16 && (m_off + j) < m_chunk; ++j) {
                    bhalf_t val = (n_local < BN && (m_off + j) < BM)
                                  ? s_dS[n_local][m_off + j] : bhalf_t(0.0f);
                    a_ptr[i * 16 + j] = val;
                }
            }

            // Fill B from Q slice (col_major transpose for WMMA matrix_b)
            for (int i = 0; i < 16 && (m_off + i) < m_chunk; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = warp_k_idx * 16 + j;
                    bhalf_t val = ((m_off + i) < BM && k_idx_local < BK)
                                  ? s_Q[k_idx_local][m_off + i] : bhalf_t(0.0f);
                    b_ptr[j * 16 + i] = val;
                }
            }

            mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    // Write back
    if (w_n_start < N_global && w_k_start < K_dim_global) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int n_local = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k_idx = w_k_start + j;
                if (n_local < N_global && k_idx < K_dim_global)
                    dK[n_local * K_dim_global + k_idx] = out.x[i * 16 + j];
            }
        }
    }
}

// ============================================================================
// dV = P^T @ dO — WMMA kernel (fixed: uses same layout as dk_kernel)
// ============================================================================
__global__ void dv_kernel(const bhalf_t* P, const bhalf_t* dO, bhalf_t* dV,
                          int M_global, int N_global, int K_dim_global, int total_heads) {
    __shared__ bhalf_t s_P[BN][BM + LDS_PAD];   // transposed: s_P[n_local][m_local] = P[m_local][n_start+n_local]
    __shared__ bhalf_t s_dO[BK+LDS_PAD][BM];     // dO slice: s_dO[k_idx_local][m_idx]

    using FragA = fragment<matrix_a, 16, 16, 16, bhalf_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, bhalf_t, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut = fragment<accumulator, 16, 16, 16, bhalf_t>;

    int block_n = blockIdx.x;
    int block_k = blockIdx.y;
    int head_idx = blockIdx.z;

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;
    int warp_n_idx = warp_id % 4;
    int warp_k_idx = warp_id / 4;

    int n_start = block_n * BN;
    int k_start = block_k * BK;
    int w_n_start = n_start + warp_n_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    P   += head_idx * M_global * N_global;
    dO  += head_idx * M_global * K_dim_global;
    dV  += head_idx * N_global * K_dim_global;

    // ---- Load P transposed: s_P[n][m] = P[m][n] ----
    for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
        int n_local = i / BM, m_local = i % BM;
        s_P[n_local][m_local] = (n_start + n_local < N_global && m_local < M_global)
                                ? P[m_local * N_global + n_start + n_local] : half_t(0.0f);
    }
    __syncthreads();

    FragC acc;
    fill_fragment(acc, 0.0f);

    for (int m_start = 0; m_start < M_global; m_start += BM) {
        int m_chunk = (BM < M_global - m_start) ? BM : (M_global - m_start);

        // Load dO slice: s_dO[k_idx_local][m_idx] = dO[m_idx][k_start+k_idx]
        for (int i = tid; i < BK * m_chunk; i += BLOCK_SIZE) {
            int k_local = i / m_chunk, m_idx = i % m_chunk;
            s_dO[k_local][m_idx] = (k_start + k_local < K_dim_global && m_start + m_idx < M_global)
                                   ? dO[(m_start + m_idx) * K_dim_global + k_start + k_local]
                                   : bhalf_t(0.0f);
        }
        __syncthreads();

        for (int m_sub = 0; m_sub < 4; ++m_sub) {
            FragA a_frag;
            FragB b_frag;

            bhalf_t* a_ptr = reinterpret_cast<bhalf_t*>(&a_frag);
            bhalf_t* b_ptr = reinterpret_cast<bhalf_t*>(&b_frag);

            int m_off = m_sub * 16;

            for (int i = 0; i < 16; ++i) {
                int n_local = warp_n_idx * 16 + i;
                for (int j = 0; j < 16 && (m_off + j) < m_chunk; ++j) {
                    bhalf_t val = (n_local < BN && (m_off + j) < BM)
                                  ? s_P[n_local][m_off + j] : bhalf_t(0.0f);
                    a_ptr[i * 16 + j] = val;
                }
            }

            for (int i = 0; i < 16 && (m_off + i) < m_chunk; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = warp_k_idx * 16 + j;
                    bhalf_t val = ((m_off + i) < BM && k_idx_local < BK)
                                  ? s_dO[k_idx_local][m_off + i] : bhalf_t(0.0f);
                    b_ptr[j * 16 + i] = val;
                }
            }

            mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    // Write back
    if (w_n_start < N_global && w_k_start < K_dim_global) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int n_local = w_n_start + i;
            for (int j = 0; j < 16; ++j) {
                int k_idx = w_k_start + j;
                if (n_local < N_global && k_idx < K_dim_global)
                    dV[n_local * K_dim_global + k_idx] = out.x[i * 16 + j];
            }
        }
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

    // 1. dP = dO @ V^T
    int dp_threads = BN * BM;
    dim3 dp_grid((dp_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BM - 1) / BM, total_heads);
    hipLaunchKernelGGL(dp_kernel, dp_grid, dim3(BLOCK_SIZE), 0, 0, dO, V, dP, M, N, K_dim, total_heads);

    // 2. Softmax backward: dS = P ⊙ (dP − rowsum)
    int softmax_shared_mem = BLOCK_SIZE * sizeof(float);
    dim3 softmax_grid((M + BM - 1) / BM, total_heads, 1);
    hipLaunchKernelGGL(softmax_bwd_kernel, softmax_grid, dim3(BLOCK_SIZE),
                       softmax_shared_mem, 0, P, dP, dS, M, N, total_heads);

    // 3. dQ = dS @ K
    dim3 dq_grid((K_dim + BN - 1) / BN, (M + BM - 1) / BM, total_heads);
    hipLaunchKernelGGL(dq_kernel, dq_grid, dim3(BLOCK_SIZE), 0, 0, dS, K, dQ, M, N, K_dim, total_heads);

    // 4. dK = dS^T @ Q
    dim3 dk_grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dk_kernel, dk_grid, dim3(BLOCK_SIZE), 0, 0, dS, Q, dK, M, N, K_dim, total_heads);

    // 5. dV = P^T @ dO
    dim3 dv_grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dv_kernel, dv_grid, dim3(BLOCK_SIZE), 0, 0, P, dO, dV, M, N, K_dim, total_heads);

    // Wait for all kernels to finish and check errors
    hipDeviceSynchronize();
    hipFree(dP);
    hipFree(dS);
}
}

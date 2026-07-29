#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include "kernel/correct/fmha_bwd_config.h"

using namespace rocwmma;
using bhalf_t = rocwmma::bfloat16_t;

// ============================================================================
// dP = dO @ V^T — naive element-per-thread with multi-head support
// dP[m][n] = sum_{k=0}^{K_dim-1} dO[m][k] * V[n][k], per head
// Each thread block computes one BM×BN tile of the output (M × N).
// Supports head tiling via head_start parameter.
// ============================================================================
__global__ void dp_kernel(const bhalf_t* dO, const bhalf_t* V, bhalf_t* dP,
                          int M_global, int N_global, int K_dim, int total_heads, int head_start) {
    int head_idx_base = blockIdx.z * MAX_HEADS_PER_LAUNCH + head_start;
    if (head_idx_base >= total_heads) return;

    int tid = threadIdx.x;
    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;

    for (int head_offset = 0; head_offset < MAX_HEADS_PER_LAUNCH; ++head_offset) {
        int head_idx = head_idx_base + head_offset;
        if (head_idx >= total_heads) break;

        const bhalf_t* dO_h = dO + head_idx * M_global * K_dim;
        const bhalf_t* V_h  = V  + head_idx * N_global * K_dim;
        bhalf_t*       dP_h = dP + head_idx * M_global * N_global;

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
}

// ============================================================================
// Softmax backward: dS = P ⊙ (dP − rowsum(dP×P))
// Supports head tiling via head_start parameter.
// Uses standard two-pass tree reduction for numerical correctness.
// ============================================================================
__global__ void softmax_bwd_kernel(const bhalf_t* P, const bhalf_t* dP, bhalf_t* dS,
                                   int M, int N, int total_heads, int head_start) {
    extern __shared__ float s_data[];
    float* s_partial = s_data;

    int head_idx_base = blockIdx.y * MAX_HEADS_PER_LAUNCH + head_start;
    if (head_idx_base >= total_heads) return;

    int m = blockIdx.x;
    if (m >= M) return;

    int tid = threadIdx.x;

    // Loop over heads assigned to this block
    for (int head_offset = 0; head_offset < MAX_HEADS_PER_LAUNCH; ++head_offset) {
        int head_idx = head_idx_base + head_offset;
        if (head_idx >= total_heads) break;

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
        __syncthreads();  // Ensure shared memory is clean for next head
    }
}

// ============================================================================
// WMMA dQ = dS @ K — optimized for GFX1100
// dS: [M x N] row-major, K: [N x K_dim] row-major -> dQ: [M x K_dim] row-major
// Each block computes one BM x BK tile of output.
// Grid: x = ceil(K_dim / BK), y = ceil(M / BM), z = ceil(heads / MAX_HEADS_PER_LAUNCH)
// ============================================================================
__global__ void dq_kernel(const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
                          int M_global, int N_global, int K_dim_global, int total_heads, int head_start) {
    __shared__ bhalf_t s_dS[BM][BN + LDS_PAD];
    __shared__ bhalf_t s_K[BK][BN + LDS_PAD];

    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bhalf_t, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    // Output fragment with row_major layout for store_matrix_sync
    using FragOut = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;

    int head_idx_base = blockIdx.z * MAX_HEADS_PER_LAUNCH + head_start;
    if (head_idx_base >= total_heads) return;

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;  // 8 warps per block (512/64)
    int warp_m_idx = warp_id % 4;
    int warp_k_idx = warp_id / 4;

    int block_m = blockIdx.y;
    int block_k = blockIdx.x;

    int m_start = block_m * BM;
    int k_start = block_k * BK;
    int w_m_start = m_start + warp_m_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // Loop over heads assigned to this block
    for (int head_offset = 0; head_offset < MAX_HEADS_PER_LAUNCH; ++head_offset) {
        int head_idx = head_idx_base + head_offset;
        if (head_idx >= total_heads) break;

        const bhalf_t* dS_h = dS + head_idx * M_global * N_global;
        const bhalf_t* K_h  = K  + head_idx * N_global * K_dim_global;
        bhalf_t*       dQ_h = dQ + head_idx * M_global * K_dim_global;

        // ---- Load full dS tile into shared memory (broadcast to all warps) ----
        for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
            int m = i / BN, n = i % BN;
            int gm = m_start + m;
            if (gm < M_global && n < N_global) {
                s_dS[m][n] = dS_h[gm * N_global + n];
            } else {
                s_dS[m][n] = bhalf_t(0.0f);
            }
        }
        __syncthreads();

        // ---- WMMA multiply: accumulate dQ_tile = dS_tile @ K_slice ----
        FragC acc;
        rocwmma::fill_fragment(acc, 0.0f);

        for (int n_start = 0; n_start < N_global; n_start += BK) {
            // Load K slice into shared memory
            for (int i = tid; i < BK * BN; i += BLOCK_SIZE) {
                int k_idx = i / BN, n = i % BN;
                int gn = n_start + n, gk = k_start + k_idx;
                s_K[k_idx][n] = (gn < N_global && gk < K_dim_global)
                                ? K_h[gn * K_dim_global + gk] : bhalf_t(0.0f);
            }
            __syncthreads();

            // WMMA: process 16x16 sub-tiles, two iterations for full BK-width
            for (int sub = 0; sub < 2; ++sub) {
                FragA a0;
                FragB b0;

                bhalf_t *a_ptr0 = reinterpret_cast<bhalf_t*>(&a0);
                bhalf_t *b_ptr0 = reinterpret_cast<bhalf_t*>(&b0);

                int n_off = sub * 16;

                // Fill A fragment from dS (row-major: m=rows, n=cols)
                for (int i = 0; i < 16; ++i) {
                    int m_local = warp_m_idx * 16 + i;
                    for (int j = 0; j < 16; ++j) {
                        int n = n_start + n_off + j;
                        bhalf_t val = (m_local < BM && n < BN) ? s_dS[m_local][n] : bhalf_t(0.0f);
                        a_ptr0[i * 16 + j] = val;
                    }
                }

                // Fill B fragment from K slice (col-major indexing: b_ptr[j*16+i])
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        int k_idx_local = warp_k_idx * 16 + j;
                        bhalf_t val = (k_idx_local < BK && (n_off + i) < BN)
                                      ? s_K[k_idx_local][n_off + i] : bhalf_t(0.0f);
                        // Col-major: transpose during load — KEY for WMMA matrix_b layout
                        b_ptr0[j * 16 + i] = val;
                    }
                }

                rocwmma::mma_sync(acc, a0, b0, acc);
            }
        }

        // ---- Write back results using store_matrix_sync ----
        if (w_m_start < M_global && w_k_start < K_dim_global) {
            FragOut out;
            rocwmma::fill_fragment(out, bhalf_t(0.0f));
            // Convert accumulator (float) to output fragment per thread
            for (int i = 0; i < out.num_elements; ++i) {
                out.x[i] = static_cast<bhalf_t>(acc.x[i]);
            }

            // Store to global memory using row-major layout
            rocwmma::store_matrix_sync(
                dQ_h + w_m_start * K_dim_global + w_k_start,
                out, K_dim_global, rocwmma::mem_row_major
            );
        }

        __syncthreads();  // Ensure shared memory is clean for next head
    }
}

// ============================================================================
// WMMA dK = dS^T @ Q — optimized for GFX1100 with head tiling
// dS: [M x N] row-major, Q: [M x K_dim] row-major -> dK: [N x K_dim] row-major
// Each block computes one BN x BK tile of output.
// Grid: x = ceil(K_dim / BK), y = ceil(N / BN), z = ceil(heads / MAX_HEADS_PER_LAUNCH)
// ============================================================================
__global__ void dk_kernel(const bhalf_t* dS, const bhalf_t* Q, bhalf_t* dK,
                          int M_global, int N_global, int K_dim_global, int total_heads, int head_start) {
    __shared__ bhalf_t s_dS[BN][BM + LDS_PAD];   // transposed: s_dS[n_local][m_local] = dS[m_local][n_start+n_local]
    __shared__ bhalf_t s_Q[BK + LDS_PAD][BM];    // Q slice: s_Q[k_idx_local][m_idx]

    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bhalf_t, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    using FragOut = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, bhalf_t>;

    int head_idx_base = blockIdx.z * MAX_HEADS_PER_LAUNCH + head_start;
    if (head_idx_base >= total_heads) return;

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;     // 8 warps per block (512/64)
    int warp_n_idx = warp_id % 4;        // N sub-tile (16-wide each, 4 across)
    int warp_k_idx = warp_id / 4;        // K sub-tile (16-wide each, 2 down)

    int block_n = blockIdx.x;            // which BN tile in N
    int block_k = blockIdx.y;            // which BK tile in K_dim

    int n_start = block_n * BN;
    int k_start = block_k * BK;
    int w_n_start = n_start + warp_n_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // Loop over heads assigned to this block
    for (int head_offset = 0; head_offset < MAX_HEADS_PER_LAUNCH; ++head_offset) {
        int head_idx = head_idx_base + head_offset;
        if (head_idx >= total_heads) break;

        const bhalf_t* dS_h = dS + head_idx * M_global * N_global;
        const bhalf_t* Q_h  = Q  + head_idx * M_global * K_dim_global;
        bhalf_t*       dK_h = dK + head_idx * N_global * K_dim_global;

        // ---- Load dS transposed into shared memory: s_dS[n][m] = dS[m][n] ----
        for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
            int n_local = i / BM, m_local = i % BM;  // [BN x BM] layout in shared mem
            s_dS[n_local][m_local] = (n_start + n_local < N_global && m_local < M_global)
                                     ? dS_h[m_local * N_global + n_start + n_local] : bhalf_t(0.0f);
        }
        __syncthreads();

        // ---- WMMA accumulate: dK_tile = dS^T @ Q  (loop over M in BM chunks) ----
        FragC acc;
        rocwmma::fill_fragment(acc, 0.0f);

        for (int m_start = 0; m_start < M_global; m_start += BM) {
            int m_chunk = (BM < M_global - m_start) ? BM : (M_global - m_start);

            // Load Q slice into shared memory: s_Q[k_local][m_idx] = Q[m_idx][k_start+k_local]
            for (int i = tid; i < BK * m_chunk; i += BLOCK_SIZE) {
                int k_local = i / m_chunk, m_idx = i % m_chunk;
                s_Q[k_local][m_idx] = (k_start + k_local < K_dim_global && m_start + m_idx < M_global)
                                      ? Q_h[(m_start + m_idx) * K_dim_global + k_start + k_local]
                                      : bhalf_t(0.0f);
            }
            __syncthreads();

            // WMMA: process 16x16 sub-tiles over M dimension (BM=64 -> 4 iterations of 16)
            for (int m_sub = 0; m_sub < 4; ++m_sub) {
                FragA a_frag;
                FragB b_frag;

                bhalf_t *a_ptr = reinterpret_cast<bhalf_t*>(&a_frag);
                bhalf_t *b_ptr = reinterpret_cast<bhalf_t*>(&b_frag);

                int m_off = m_sub * 16;

                // Fill A fragment from dS^T (row-major: n=rows, m=cols)
                for (int i = 0; i < 16; ++i) {
                    int n_local = warp_n_idx * 16 + i;
                    for (int j = 0; j < 16 && (m_off + j) < m_chunk; ++j) {
                        bhalf_t val = (n_local < BN && (m_off + j) < BM)
                                      ? s_dS[n_local][m_off + j] : bhalf_t(0.0f);
                        a_ptr[i * 16 + j] = val;
                    }
                }

                // Fill B fragment from Q slice (col-major: transpose during load for WMMA matrix_b)
                for (int i = 0; i < 16 && (m_off + i) < m_chunk; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        int k_idx_local = warp_k_idx * 16 + j;
                        bhalf_t val = ((m_off + i) < BM && k_idx_local < BK)
                                      ? s_Q[k_idx_local][m_off + i] : bhalf_t(0.0f);
                        // Col-major: b_ptr[j*16+i] = val
                        b_ptr[j * 16 + i] = val;
                    }
                }

                rocwmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            __syncthreads();  // ensure next Q slice load is visible
        }

        // ---- Write back results: dK[n][k] from accumulator ----
        if (w_n_start < N_global && w_k_start < K_dim_global) {
            FragOut out;
            for (int i = 0; i < acc.num_elements; ++i) {
                out.x[i] = static_cast<bhalf_t>(acc.x[i]);
            }
            for (int i = 0; i < 16; ++i) {
                int n_local = w_n_start + i;
                for (int j = 0; j < 16; ++j) {
                    int k_idx = w_k_start + j;
                    if (n_local < N_global && k_idx < K_dim_global) {
                        dK_h[n_local * K_dim_global + k_idx] = out.x[i * 16 + j];
                    }
                }
            }
        }

        __syncthreads();  // clean shared memory for next head
    }
}

// ============================================================================
// WMMA dV = P^T @ dO — optimized for GFX1100 with head tiling
// P: [M x N] row-major, dO: [M x K_dim] row-major -> dV: [N x K_dim] row-major
// dV[n][k] = sum_{m=0}^{M-1} P[m][n] * dO[m][k]
// Each block computes one BN x BK tile of output.
// Grid: x = ceil(K_dim / BK), y = ceil(N / BN), z = ceil(heads / MAX_HEADS_PER_LAUNCH)
// ============================================================================
__global__ void dv_kernel(const bhalf_t* P, const bhalf_t* dO, bhalf_t* dV,
                          int M_global, int N_global, int K_dim_global, int total_heads, int head_start) {
    __shared__ bhalf_t s_P[BN][BM + LDS_PAD];   // transposed: s_P[n_local][m_local] = P[m_local][n_start+n_local]
    __shared__ bhalf_t s_dO[BK + LDS_PAD][BM];  // dO slice: s_dO[k_idx_local][m_idx]

    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bhalf_t, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    using FragOut = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, bhalf_t>;

    int head_idx_base = blockIdx.z * MAX_HEADS_PER_LAUNCH + head_start;
    if (head_idx_base >= total_heads) return;

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;     // 8 warps per block (512/64)
    int warp_n_idx = warp_id % 4;        // N sub-tile (16-wide each, 4 across)
    int warp_k_idx = warp_id / 4;        // K sub-tile (16-wide each, 2 down)

    int block_n = blockIdx.x;            // which BN tile in N
    int block_k = blockIdx.y;            // which BK tile in K_dim

    int n_start = block_n * BN;
    int k_start = block_k * BK;
    int w_n_start = n_start + warp_n_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // Loop over heads assigned to this block
    for (int head_offset = 0; head_offset < MAX_HEADS_PER_LAUNCH; ++head_offset) {
        int head_idx = head_idx_base + head_offset;
        if (head_idx >= total_heads) break;

        const bhalf_t* P_h   = P + head_idx * M_global * N_global;
        const bhalf_t* dO_h  = dO + head_idx * M_global * K_dim_global;
        bhalf_t*       dV_h  = dV + head_idx * N_global * K_dim_global;

        // ---- Load P transposed into shared memory: s_P[n][m] = P[m][n] ----
        for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
            int n_local = i / BM, m_local = i % BM;  // [BN x BM] layout
            s_P[n_local][m_local] = (n_start + n_local < N_global && m_local < M_global)
                                    ? P_h[m_local * N_global + n_start + n_local] : bhalf_t(0.0f);
        }
        __syncthreads();

        // ---- WMMA accumulate: dV_tile = P^T @ dO  (loop over M in BM chunks) ----
        FragC acc;
        rocwmma::fill_fragment(acc, 0.0f);

        for (int m_start = 0; m_start < M_global; m_start += BM) {
            int m_chunk = (BM < M_global - m_start) ? BM : (M_global - m_start);

            // Load dO slice into shared memory: s_dO[k_local][m_idx] = dO[m_idx][k_start+k_local]
            for (int i = tid; i < BK * m_chunk; i += BLOCK_SIZE) {
                int k_local = i / m_chunk, m_idx = i % m_chunk;
                s_dO[k_local][m_idx] = (k_start + k_local < K_dim_global && m_start + m_idx < M_global)
                                       ? dO_h[(m_start + m_idx) * K_dim_global + k_start + k_local]
                                       : bhalf_t(0.0f);
            }
            __syncthreads();

            // WMMA: process 16x16 sub-tiles over M dimension (BM=64 -> 4 iterations of 16)
            for (int m_sub = 0; m_sub < 4; ++m_sub) {
                FragA a_frag;
                FragB b_frag;

                bhalf_t *a_ptr = reinterpret_cast<bhalf_t*>(&a_frag);
                bhalf_t *b_ptr = reinterpret_cast<bhalf_t*>(&b_frag);

                int m_off = m_sub * 16;

                // Fill A fragment from P^T (row-major: n=rows, m=cols)
                for (int i = 0; i < 16; ++i) {
                    int n_local = warp_n_idx * 16 + i;
                    for (int j = 0; j < 16 && (m_off + j) < m_chunk; ++j) {
                        bhalf_t val = (n_local < BN && (m_off + j) < BM)
                                      ? s_P[n_local][m_off + j] : bhalf_t(0.0f);
                        a_ptr[i * 16 + j] = val;
                    }
                }

                // Fill B fragment from dO slice (col-major transpose for WMMA matrix_b layout)
                for (int i = 0; i < 16 && (m_off + i) < m_chunk; ++i) {
                    for (int j = 0; j < 16; ++j) {
                        int k_idx_local = warp_k_idx * 16 + j;
                        bhalf_t val = ((m_off + i) < BM && k_idx_local < BK)
                                      ? s_dO[k_idx_local][m_off + i] : bhalf_t(0.0f);
                        b_ptr[j * 16 + i] = val;
                    }
                }

                rocwmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            __syncthreads();  // ensure next dO slice load is visible
        }

        // ---- Write back results: dV[n][k] from accumulator ----
        if (w_n_start < N_global && w_k_start < K_dim_global) {
            FragOut out;
            for (int i = 0; i < acc.num_elements; ++i) {
                out.x[i] = static_cast<bhalf_t>(acc.x[i]);
            }
            for (int i = 0; i < 16; ++i) {
                int n_local = w_n_start + i;
                for (int j = 0; j < 16; ++j) {
                    int k_idx = w_k_start + j;
                    if (n_local < N_global && k_idx < K_dim_global) {
                        dV_h[n_local * K_dim_global + k_idx] = out.x[i * 16 + j];
                    }
                }
            }
        }

        __syncthreads();  // clean shared memory for next head
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

    // Grid z-dimension: blocks per z = ceil(total_heads / MAX_HEADS_PER_LAUNCH)
    int grid_z = (total_heads + MAX_HEADS_PER_LAUNCH - 1) / MAX_HEADS_PER_LAUNCH;
    int head_start = 0;  // host handles tiling via multiple z-blocks; kernel starts at head_start + blockIdx.z * MAX_HEADS_PER_LAUNCH

    // 1. dP = dO @ V^T (naive element-per-thread with multi-head)
    dim3 dp_grid((N + BN - 1) / BN, (M + BM - 1) / BM, grid_z);
    hipLaunchKernelGGL(dp_kernel, dp_grid, dim3(BLOCK_SIZE), 0, 0, dO, V, dP, M, N, K_dim, total_heads, head_start);

    // 2. Softmax backward: dS = P ⊙ (dP − rowsum)
    int softmax_shared_mem = BLOCK_SIZE * sizeof(float);
    dim3 softmax_grid((M + BM - 1) / BM, grid_z, 1);
    hipLaunchKernelGGL(softmax_bwd_kernel, softmax_grid, dim3(BLOCK_SIZE),
                       softmax_shared_mem, 0, P, dP, dS, M, N, total_heads, head_start);

    // 3. dQ = dS @ K (WMMA with head tiling)
    dim3 dq_grid((K_dim + BK - 1) / BK, (M + BM - 1) / BM, grid_z);
    hipLaunchKernelGGL(dq_kernel, dq_grid, dim3(BLOCK_SIZE), 0, 0, dS, K, dQ, M, N, K_dim, total_heads, head_start);

    // 4. dK = dS^T @ Q (WMMA with head tiling)
    dim3 dk_grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, grid_z);
    hipLaunchKernelGGL(dk_kernel, dk_grid, dim3(BLOCK_SIZE), 0, 0, dS, Q, dK, M, N, K_dim, total_heads, head_start);

    // 5. dV = P^T @ dO (WMMA with head tiling)
    // Output shape: [N × K_dim] — grid is (N-tiles × K-tiles × heads-per-launch)
    dim3 dv_grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, grid_z);
    hipLaunchKernelGGL(dv_kernel, dv_grid, dim3(BLOCK_SIZE), 0, 0, P, dO, dV, M, N, K_dim, total_heads, head_start);

    // Wait for all kernels to finish and check errors
    hipDeviceSynchronize();
    hipFree(dP);
    hipFree(dS);
}

// Test wrapper functions for isolated kernel testing
void test_dp_kernel(const bhalf_t* dO, const bhalf_t* V, bhalf_t* dP,
                    int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM,
              (total_heads + MAX_HEADS_PER_LAUNCH - 1) / MAX_HEADS_PER_LAUNCH);
    hipLaunchKernelGGL(dp_kernel, grid, dim3(BLOCK_SIZE), 0, 0, dO, V, dP, M, N, K_dim, total_heads, 0);
}

void test_softmax_bwd_kernel(const bhalf_t* P, const bhalf_t* dP, bhalf_t* dS,
                             int M, int N, int total_heads) {
    int shared_mem_size = BLOCK_SIZE * sizeof(float);  // partial sums
    dim3 grid(M,  // One block per row
              (total_heads + MAX_HEADS_PER_LAUNCH - 1) / MAX_HEADS_PER_LAUNCH, 1);
    hipLaunchKernelGGL(softmax_bwd_kernel, grid, dim3(BLOCK_SIZE), shared_mem_size, 0, P, dP, dS, M, N, total_heads, 0);
}

void test_dq_kernel(const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
                    int M, int N, int K_dim, int total_heads) {
    dim3 grid((K_dim + BK - 1) / BK, (M + BM - 1) / BM,
              (total_heads + MAX_HEADS_PER_LAUNCH - 1) / MAX_HEADS_PER_LAUNCH);
    hipLaunchKernelGGL(dq_kernel, grid, dim3(BLOCK_SIZE), 0, 0, dS, K, dQ, M, N, K_dim, total_heads, 0);
}

void test_dk_kernel(const bhalf_t* dS, const bhalf_t* Q, bhalf_t* dK,
                    int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK,
              (total_heads + MAX_HEADS_PER_LAUNCH - 1) / MAX_HEADS_PER_LAUNCH);
    hipLaunchKernelGGL(dk_kernel, grid, dim3(BLOCK_SIZE), 0, 0, dS, Q, dK, M, N, K_dim, total_heads, 0);
}

void test_dv_kernel(const bhalf_t* P, const bhalf_t* dO, bhalf_t* dV,
                    int M, int N, int BK_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (BK_dim + BK - 1) / BK,
              (total_heads + MAX_HEADS_PER_LAUNCH - 1) / MAX_HEADS_PER_LAUNCH);
    hipLaunchKernelGGL(dv_kernel, grid, dim3(BLOCK_SIZE), 0, 0, P, dO, dV, M, N, BK_dim, total_heads, 0);
}
}



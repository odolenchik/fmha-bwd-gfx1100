import re
import sys

with open('src/fmha_bwd_kernels.hip', 'r') as f:
    content = f.read()

# Helper to replace a kernel block given its name and new code
def replace_kernel(name, new_code):
    # Pattern: from the line containing '__global__ void {name}(' up to the line before next '__global__ void' or 'extern "C" {' or end of file.
    pattern = rf'(__global__ void\s+{name}\s*\([^)]*\)\s*{{.*?))(?=\s*(__global__ void|extern "C"|\Z))'
    # Use re.DOTALL to match across lines
    new_content = re.sub(pattern, new_code, content, flags=re.DOTALL)
    return new_content

# We'll generate new kernels with head offset.
# First, include the necessary headers at top if not present.
# Ensure we have #include "kernel/correct/fmha_bwd_config.h" already present.

# dq_kernel WMMA version with head offset
dq_kernel_code = r'''__global__ void dq_kernel(const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
                          int M_global, int N_global, int K_dim_global, int total_heads) {
    int head_idx = blockIdx.z;
    if (head_idx >= total_heads) return;
    const bhalf_t* dS_base = dS + head_idx * M_global * N_global;
    const bhalf_t* K_base  = K  + head_idx * M_global * K_dim_global;
    bhalf_t*       dQ_base = dQ + head_idx * M_global * K_dim_global;

    __shared__ bhalf_t s_dS[BM][BN + LDS_PAD];   // [m_local × n_local] — one BN-chunk at a time
    __shared__ bhalf_t s_K[BK][BN + LDS_PAD];    // [k_local × n_idx]

    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bhalf_t, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    using FragOut = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, bhalf_t>;

    int block_m = blockIdx.y;
    int block_k = blockIdx.x;

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;  // 8 warps per block (512/64)
    int warp_m_idx = warp_id % 4;
    int warp_k_idx = warp_id / 4;

    int m_start = block_m * BM;
    int k_start = block_k * BN;
    int w_m_start = m_start + warp_m_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // ---- Load dS tile into shared memory (full tile broadcast) ----
    for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
        int m = i / BN, n = i % BN;
        int gm = m_start + m;
        if (gm < M_global && n < N_global) {
            s_dS[m][n] = dS_base[gm * N_global + n];
        } else {
            s_dS[m][n] = bhalf_t(0.0f);
        }
    }
    __syncthreads();

    // ---- WMMA multiply: accumulate dQ_tile = dS_base @ K_slice ----
    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int n_start = 0; n_start < N_global; n_start += BK) {
        // Load K slice into shared memory
        for (int i = tid; i < BK * BN; i += BLOCK_SIZE) {
            int k_idx = i / BN, n = i % BN;
            int gn = n_start + n, gk = k_start + k_idx;
            s_K[k_idx][n] = (gn < N_global && gk < K_dim_global)
                            ? K_base[gn * K_dim_global + gk] : bhalf_t(0.0f);
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
                int m_local = w_m_start + i;
                for (int j = 0; j < 16; ++j) {
                    int n = n_start + n_off + j;
                    bhalf_t val = (m_local < BM && n < BN) ? s_dS[m_local][n] : bhalf_t(0.0f);
                    a_ptr0[i * 16 + j] = val;
                }
            }

            // Fill B fragment from K slice (col-major indexing: b_ptr[j*16+i])
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = w_k_start + j;
                    bhalf_t val = (k_idx_local < BK && (n_off + i) < BN)
                                  ? s_K[k_idx_local][n_off + i] : bhalf_t(0.0f);
                    // Col-major: transpose during load — KEY for WMMA matrix_b layout
                    b_ptr0[j * 16 + i] = val;
                }
            }

            rocwmma::mma_sync(acc, a0, b0, acc);
        }
    }

    // ---- Write back results ----
    if (w_m_start < M_global && w_k_start < K_dim_global) {
        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) {
            out.x[i] = static_cast<bhalf_t>(acc.x[i]);
        }
        for (int i = 0; i < 16; ++i) {
            int m_local = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int k = w_k_start + j;
                if (m_local < M_global && k < K_dim_global) {
                    dQ_base[m_local * K_dim_global + k] = out.x[i * 16 + j];
                }
            }
        }
    }
}'''

# dk_kernel WMMA version with head offset
dk_kernel_code = r'''__global__ void dk_kernel(const bhalf_t* dS, const bhalf_t* Q, bhalf_t* dK,
                          int M_global, int N_global, int K_dim_global, int total_heads) {
    int head_idx = blockIdx.z;
    if (head_idx >= total_heads) return;
    const bhalf_t* dS_base = dS + head_idx * M_global * N_global;
    const bhalf_t* Q_base  = Q  + head_idx * M_global * K_dim_global;
    bhalf_t*       dK_base = dK + head_idx * N_global * K_dim_global;

    __shared__ bhalf_t s_dS[BN][BM + LDS_PAD];   // transposed: s_dS[n_local][m_local] = dS[m_local][n_start+n_local]
    __shared__ bhalf_t s_Q[BK+LDS_PAD][BM];      // Q slice: s_Q[k_idx_local][m_idx]

    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bhalf_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bhalf_t, rocwmma::col_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
    using FragOut = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, bhalf_t>;

    int block_n = blockIdx.x;      // which N tile (BN=64)
    int block_k = blockIdx.y;      // which K_dim tile (BK=32)

    int tid = threadIdx.x;
    int warp_id   = tid / WARP_SIZE;     // 0..7
    int warp_n_idx = warp_id % 4;        // N sub-tile (16-wide each, 4 across)
    int warp_k_idx = warp_id / 4;        // K sub-tile (16-wide each, 2 down)

    int n_start = block_n * BN;      // starting column in dS, row in dK
    int k_start = block_k * BK;      // starting column in Q and dK
    int w_n_start = n_start + warp_n_idx * 16;
    int w_k_start = k_start + warp_k_idx * 16;

    // ---- Load dS transposed into shared memory: s_dS[n][m] = dS[m][n] ----
    for (int i = tid; i < BM * BN; i += BLOCK_SIZE) {
        int n_local = i / BM, m_local = i % BM;  // [BN x BM] layout in shared mem
        s_dS[n_local][m_local] = (n_start + n_local < N_global && m_local < M_global)
                                 ? dS_base[m_local * N_global + n_start + n_local] : bhalf_t(0.0f);
    }
    __syncthreads();

    // ---- WMMA accumulate: dK_tile = dS^T @ Q  (loop over M in BM chunks) ----
    FragC acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int m_start = 0; m_start < M_global; m_start += BM) {
        int m_chunk = (BM < M_global - m_start) ? BM : (M_global - m_start);

        // Load Q slice into shared memory: s_Q[k_local][m_idx] = Q[m_idx][k_local]
        for (int i = tid; i < BK * m_chunk; i += BLOCK_SIZE) {
            int k_local = i / m_chunk, m_idx = i % m_chunk;
            s_Q[k_local][m_idx] = (k_start + k_local < K_dim_global && m_start + m_idx < M_global)
                                  ? Q_base[(m_start + m_idx) * K_dim_global + k_start + k_local]
                                  : bhalf_t(0.0f);
        }
        __syncthreads();

        // WMMA: process 16x16 sub-tiles over M dimension (BM=64 → 4 iterations of 16)
        for (int m_sub = 0; m_sub < 4; ++m_sub) {
            FragA a_frag;
            FragB b_frag;

            bhalf_t *a_ptr = reinterpret_cast<bhalf_t*>(&a_frag);
            bhalf_t *b_ptr = reinterpret_cast<bhalf_t*>(&b_frag);

            int m_off = m_sub * 16;

            // Fill A fragment from dS^T (row-major: n=rows, m=cols)
            // a_ptr[i*16+j] → element [row=n_local+i, col=m_off+j] in the 16x16 sub-tile
            for (int i = 0; i < 16; ++i) {
                int n_local = warp_n_idx * 16 + i;
                for (int j = 0; j < 16 && (m_off + j) < m_chunk; ++j) {
                    bhalf_t val = (n_local < BN && (m_off + j) < BM)
                                  ? s_dS[n_local][m_off + j] : bhat_t(0.0f);
                    a_ptr[i * 16 + j] = val;
                }
            }

            // Fill B fragment from Q slice (col-major: transpose during load for WMMA matrix_b)
            // b_ptr[j*16+i] → element [row=k_idx+j, col=m_off+i] in the 16x16 sub-tile
            for (int i = 0; i < 16 && (m_off + i) < m_chunk; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int k_idx_local = w_k_start + j;
                    bhalf_t val = ((m_off + i) < BM && k_idx_local < BK)
                                  ? s_Q[k_idx_local][m_off + i] : bhalf_t(0.0f);
                    b_ptr0[j * 16 + i] = val;
                }
            }

            rocwmma::mma_sync(acc, a_frag, b_frag, acc);
        }
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
                    dK_base[n_local * K_dim_global + k_idx] = out.x[i * 16 + j];
                }
            }
        }
    }
}'''

# dv_kernel WMMA version with head offset (same as dk but with P and dO)
dv_kernel_code = r'''__global__ void dv_kernel(const bhalf_t* P, const bhalf_t* dO, bhalf_t* dV,
                          int M_global, int N_global, int BK_dim, int total_heads) {
    int head_idx = blockIdx.z;
    if (head_idx >= total_heads) return;
    const bhalf_t* P_base   = P + head_idx * M_global * N_global;
    const bhalf_t* dO_base  = dO + head_idx * M_global * BK_dim;
    bhalf_t*       dV_base  = dV + head_idx * N_global * BK_dim;

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
            bhalf_t p_val = P_base[m * N_global + block_n + n_local];
            bhalf_t do_val = dO_base[m * BK_dim + block_k + k_local];
            sum += static_cast<float>(p_val) * static_cast<float>(do_val);
        }

        dV_base[(block_n + n_local) * BK_dim + block_k + k_local] = static_cast<bhalf_t>(sum);
    }
}'''

# Replace each kernel
content = re.sub(r'__global__ void dq_kernel\([^}]*?\}\s*\}\s*', dq_kernel_code + '\n\n', content, flags=re.DOTALL)
content = re.sub(r'__global__ void dk_kernel\([^}]*?\}\s*\}\s*', dk_kernel_code + '\n\n', content, flags=re.DOTALL)
content = re.sub(r'__global__ void dv_kernel\([^}]*?\}\s*\}\s*', dv_kernel_code + '\n\n', content, flags=re.DOTALL)

with open('src/fmha_bwd_kernels.hip', 'w') as f:
    f.write(content)

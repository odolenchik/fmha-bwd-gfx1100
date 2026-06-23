#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include "kernel/correct/fmha_bwd_config.h"

using namespace rocwmma;
using half_t = half;  // FP16 type

// Forward declarations of test functions for FP16
extern "C" {
void test_dp_kernel_fp16(const half_t* dO, const half_t* V, half_t* dP,
                         int M_global, int N_global, int K_dim, int total_heads);
void test_softmax_bwd_kernel_fp16(const half_t* P, const half_t* dP, half_t* dS,
                                  int M, int N, int total_heads);
void test_dq_kernel_fp16(const half_t* dS, const half_t* K, half_t* dQ,
                         int M_global, int N_global, int K_dim_global, int total_heads);
void test_dk_kernel_fp16(const half_t* dS, const half_t* Q, half_t* dK,
                         int M_global, int N_global, int K_dim_global, int total_heads);
void test_dv_kernel_fp16(const half_t* P, const half_t* dO, half_t* dV,
                         int M_global, int N_global, int BK_dim, int total_heads);
}

// ============================================================================
// dP = dO @ V^T — naive element-per-thread with multi-head support
// dP[m][n] = sum_{k=0}^{K_dim-1} dO[m][k] * V[n][k], per head
// Each thread block computes one BM×BN tile of the output (M × N).
// ============================================================================
__global__ void dp_kernel_fp16(const half_t* dO, const half_t* V, half_t* dP,
                               int M_global, int N_global, int K_dim, int total_heads) {
    int head_idx = blockIdx.z;
    if (head_idx >= total_heads) return;

    const half_t* dO_h = dO + head_idx * M_global * K_dim;
    const half_t* V_h  = V  + head_idx * N_global * K_dim;
    half_t*       dP_h = dP + head_idx * M_global * N_global;

    int tid = threadIdx.x;
    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;

    for (int idx = tid; idx < BM * BN; idx += BLOCK_SIZE) {
        int m_local = idx / BN;   // 0..63
        int n_local = idx % BN;   // 0..63

        if (block_m + m_local >= M_global || block_n + n_local >= N_global) continue;

        float sum = 0.0f;
        for (int k = 0; k < K_dim; ++k) {
            half_t do_val = dO_h[(block_m + m_local) * K_dim + k];
            half_t v_val  = V_h[(block_n + n_local) * K_dim + k];
            sum += __half2float(do_val) * __half2float(v_val);
        }

        dP_h[(block_m + m_local) * N_global + block_n + n_local] = __float2half_rn(sum);
    }
}

// ============================================================================
// Softmax backward: dS = P ⊙ (dP − rowsum(dP×P))
// Each block processes BM rows and all N columns for one head.
// Uses shared memory for row sum reduction.
// ============================================================================
__global__ void softmax_bwd_kernel_fp16(const half_t* P, const half_t* dP, half_t* dS,
                                        int M, int N, int total_heads) {
    extern __shared__ float s_data[];
    int head_idx = blockIdx.y;        // Head index is in y-dimension
    int block_m_start = blockIdx.x * BM;  // Starting row for this block (x-dimension)
    int tid = threadIdx.x;

    if (head_idx >= total_heads) return;

    const half_t* P_h  = P + head_idx * M * N;
    const half_t* dP_h = dP + head_idx * M * N;
    half_t*       dS_h = dS + head_idx * M * N;

    // We'll use shared memory for two purposes:
    // 1. Partial sums for row reduction (size: blockDim.x)
    // 2. Final row sums (size: BM)
    float* s_partial = s_data;                 // Size: blockDim.x (for reduction)
    float* s_rowsums = &s_data[blockDim.x];    // Size: BM (for storing row sums)

    // Phase 1: Compute row sums of P * dP with reduction
    for (int row_offset = 0; row_offset < BM; row_offset++) {
        int m = block_m_start + row_offset;
        if (m >= M) continue;  // Out of bounds

        // Each thread computes partial sum for its assigned columns
        float local_sum = 0.0f;
        for (int n = tid; n < N; n += blockDim.x) {
            float p_val   = __half2float(P_h[m * N + n]);
            float dp_val  = __half2float(dP_h[m * N + n]);
            local_sum += p_val * dp_val;
        }

        // Store partial sum in shared memory for reduction
        s_partial[tid] = local_sum;
        __syncthreads();

        // Reduction in shared memory (tree reduction)
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_partial[tid] += s_partial[tid + s];
            }
            __syncthreads();
        }

        // Thread 0 stores the final row sum
        if (tid == 0) {
            s_rowsums[row_offset] = s_partial[0];
        }
        __syncthreads();
    }

    // Phase 2: Compute dS = P * (dP - rowsum)
    for (int row_offset = 0; row_offset < BM; row_offset++) {
        int m = block_m_start + row_offset;
        if (m >= M) continue;  // Out of bounds

        float rowsum = s_rowsums[row_offset];

        // Each thread processes multiple columns in this row
        for (int n = tid; n < N; n += blockDim.x) {
            float p_val   = __half2float(P_h[m * N + n]);
            float dp_val  = __half2float(dP_h[m * N + n]);
            dS_h[m * N + n] = __float2half_rn(p_val * (dp_val - rowsum));
        }
    }
}

// ============================================================================
// dQ = dS @ K — naive element-per-thread with multi-head support
// dQ[m][k] = sum_{n=0}^{N_global-1} dS[m][n] * K[n][k], per head
// Each thread computes one output element.
// ============================================================================
__global__ void dq_kernel_fp16(const half_t* dS, const half_t* K, half_t* dQ,
                               int M_global, int N_global, int K_dim_global, int total_heads) {
    int head_idx = blockIdx.z;
    if (head_idx >= total_heads) return;

    const half_t* dS_h = dS + head_idx * M_global * N_global;
    const half_t* K_h  = K  + head_idx * N_global * K_dim_global;
    half_t*       dQ_h = dQ + head_idx * M_global * K_dim_global;

    int tid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
    int total_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x;

    // Each thread processes multiple elements if needed
    for (int idx = tid; idx < M_global * K_dim_global; idx += total_threads) {
        int m = idx / K_dim_global;   // 0..M_global-1
        int k = idx % K_dim_global;   // 0..K_dim_global-1

        float sum = 0.0f;
        for (int n = 0; n < N_global; ++n) {
            half_t ds_val = dS_h[m * N_global + n];
            half_t k_val  = K_h[n * K_dim_global + k];
            sum += __half2float(ds_val) * __half2float(k_val);
        }

        dQ_h[m * K_dim_global + k] = __float2half_rn(sum);
    }
}

// ============================================================================
// dK = dS^T @ Q — naive element-per-thread with multi-head support
// dK[n][k] = sum_{m=0}^{M_global-1} dS[m][n] * Q[m][k], per head
// Each thread computes one output element.
// ============================================================================
__global__ void dk_kernel_fp16(const half_t* dS, const half_t* Q, half_t* dK,
                               int M_global, int N_global, int K_dim_global, int total_heads) {
    int head_idx = blockIdx.z;
    if (head_idx >= total_heads) return;

    const half_t* dS_h = dS + head_idx * M_global * N_global;
    const half_t* Q_h  = Q  + head_idx * M_global * K_dim_global;
    half_t*       dK_h = dK + head_idx * N_global * K_dim_global;

    int tid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
    int total_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x;

    // Each thread processes multiple elements if needed
    for (int idx = tid; idx < N_global * K_dim_global; idx += total_threads) {
        int n = idx / K_dim_global;   // 0..N_global-1
        int k = idx % K_dim_global;   // 0..K_dim_global-1

        float sum = 0.0f;
        for (int m = 0; m < M_global; ++m) {
            half_t ds_val = dS_h[m * N_global + n];
            half_t q_val  = Q_h[m * K_dim_global + k];
            sum += __half2float(ds_val) * __half2float(q_val);
        }

        dK_h[n * K_dim_global + k] = __float2half_rn(sum);
    }
}

// ============================================================================
// dV = P^T @ dO — naive element-per-thread with multi-head support
// dV[n][k] = sum_{m=0}^{M-1} P[m][n] * dO[m][k], per head
// Each thread block computes one BN×BK tile of the output (N × K_dim).
// No shared memory — each thread loads from global directly.
// ============================================================================
__global__ void dv_kernel_fp16(const half_t* P, const half_t* dO, half_t* dV,
                               int M_global, int N_global, int BK_dim, int total_heads) {
    int head_idx = blockIdx.z;  // which head this block computes for
    if (head_idx >= total_heads) return;

    const half_t* P_h   = P + head_idx * M_global * N_global;
    const half_t* dO_h  = dO + head_idx * M_global * BK_dim;
    half_t*       dV_h  = dV + head_idx * N_global * BK_dim;

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
            half_t p_val = P_h[m * N_global + block_n + n_local];
            half_t do_val = dO_h[m * BK_dim + block_k + k_local];
            sum += __half2float(p_val) * __half2float(do_val);
        }

        dV_h[(block_n + n_local) * BK_dim + block_k + k_local] = __float2half_rn(sum);
    }
}

// ============================================================================
// C entry point — launches all kernels in sequence with error checking
// ============================================================================
extern "C" {
void fmha_bwd_full_fp16(half_t* dQ, half_t* dK, half_t* dV,
                        const half_t* Q, const half_t* K, const half_t* V,
                        const half_t* P, const half_t* dO, int M, int N, int K_dim, int total_heads)
{
    // Allocate intermediate buffers
    size_t size_P = total_heads * M * N * sizeof(half_t);
    size_t size_S = total_heads * M * N * sizeof(half_t);

    half_t* dP = nullptr;
    half_t* dS = nullptr;

    if (hipMalloc(&dP, size_P) != hipSuccess || hipMalloc(&dS, size_S) != hipSuccess) {
        if (dP) hipFree(dP);
        fprintf(stderr, "fmha_bwd_fp16: malloc failed for %zu + %zu bytes\n", size_P, size_S);
        return;
    }

    // 1. dP = dO @ V^T (naive element-per-thread with multi-head)
    dim3 dp_grid((N + BN - 1) / BN, (M + BM - 1) / BM, total_heads);
    hipLaunchKernelGGL(dp_kernel_fp16, dp_grid, dim3(BLOCK_SIZE), 0, 0, dO, V, dP, M, N, K_dim, total_heads);

    // 2. Softmax backward: dS = P ⊙ (dP − rowsum)
    int softmax_shared_mem = (BLOCK_SIZE + BM) * sizeof(float);  // partial sums + row sums
    dim3 softmax_grid((M + BM - 1) / BM, total_heads, 1);
    hipLaunchKernelGGL(softmax_bwd_kernel_fp16, softmax_grid, dim3(BLOCK_SIZE),
                       softmax_shared_mem, 0, P, dP, dS, M, N, total_heads);

    // 3. dQ = dS @ K (naive element-per-thread with multi-head)
    dim3 dq_grid((K_dim + BK - 1) / BK, (M + BM - 1) / BM, total_heads);
    hipLaunchKernelGGL(dq_kernel_fp16, dq_grid, dim3(BLOCK_SIZE), 0, 0, dS, K, dQ, M, N, K_dim, total_heads);

    // 4. dK = dS^T @ Q (naive element-per-thread with multi-head)
    dim3 dk_grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dk_kernel_fp16, dk_grid, dim3(BLOCK_SIZE), 0, 0, dS, Q, dK, M, N, K_dim, total_heads);

    // 5. dV = P^T @ dO (naive element-per-thread)
    // Output shape: [N × K_dim] — grid is (N-tiles × K-tiles × heads)
    dim3 dv_grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dv_kernel_fp16, dv_grid, dim3(BLOCK_SIZE), 0, 0, P, dO, dV, M, N, K_dim, total_heads);

    // Wait for all kernels to finish and check errors
    hipDeviceSynchronize();
    hipFree(dP);
    hipFree(dS);
}
// Close extern "C" block
}

// Test wrapper functions
void test_dq_kernel_fp16(const half_t* dS, const half_t* K, half_t* dQ,
                         int M, int N, int K_dim, int total_heads) {
    dim3 grid((K_dim + BK - 1) / BK, (M + BM - 1) / BM, total_heads);
    hipLaunchKernelGGL(dq_kernel_fp16, grid, dim3(BLOCK_SIZE), 0, 0, dS, K, dQ, M, N, K_dim, total_heads);
}

void test_dk_kernel_fp16(const half_t* dS, const half_t* Q, half_t* dK,
                         int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dk_kernel_fp16, grid, dim3(BLOCK_SIZE), 0, 0, dS, Q, dK, M, N, K_dim, total_heads);
}

void test_dv_kernel_fp16(const half_t* P, const half_t* dO, half_t* dV,
                         int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dv_kernel_fp16, grid, dim3(BLOCK_SIZE), 0, 0, P, dO, dV, M, N, K_dim, total_heads);
}

void test_dp_kernel_fp16(const half_t* dO, const half_t* V, half_t* dP,
                         int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, total_heads);
    hipLaunchKernelGGL(dp_kernel_fp16, grid, dim3(BLOCK_SIZE), 0, 0, dO, V, dP, M, N, K_dim, total_heads);
}

void test_softmax_bwd_kernel_fp16(const half_t* P, const half_t* dP, half_t* dS,
                                  int M, int N, int total_heads) {
    int shared_mem_size = (BLOCK_SIZE + BM) * sizeof(float);  // partial sums + row sums
    dim3 grid((M + BM - 1) / BM, total_heads, 1);
    hipLaunchKernelGGL(softmax_bwd_kernel_fp16, grid, dim3(BLOCK_SIZE), shared_mem_size, 0, P, dP, dS, M, N, total_heads);
}
# dK WMMA Kernel Debug Log

## Problem Statement
dK = dS^T @ Q kernel using rocWMMA on gfx1100 (RDNA3) produces constant -4704.0 for all elements in each warp's output block, instead of correct varying values.

## Environment
- GPU: Radeon RX 7900 XTX (gfx1100)
- ROCm: 7.2.1
- rocWMMA: header-only (`/opt/rocm/include/rocwmma/`)
- Data type: bfloat16 (rocwmma::bfloat16_t)
- Dimensions: M=64, N=64, K_dim=32

## Reference Implementation (Working)
Naive kernel (one element per thread): max error ~14.5 from CPU reference (acceptable for bfloat16).

Formula: dK[n][k] = Σ_m dS[m][n] * Q[m][k]

## Working dQ Kernel Pattern
```cpp
// dQ = dS @ K, s_dS[BM][BN+LDS_PAD], s_K[BK][BN+LDS_PAD]
// A from dS (row-major): a_ptr[i*16+j] = s_dS[m_local][n]
// B from K (col-major transpose): b_ptr[j*16+i] = s_K[k_idx_local][n_off+i]
```

## Attempted Fixes for dK Kernel

### Attempt 1: Shared memory layout [BK][BM+LDS_PAD] → [BN][BM+LDS_PAD]
**Change:** `s_dS[BK][BM+LDS_PAD]` (32×72=2304) → `s_dS[BN][BM+LDS_PAD]` (64×72=4608)  
**Also removed `% BK` from indexing.**
**Result:** FAILED — max error huge, dK[0][*] = -4704 constant.

### Attempt 2: Shared memory [BN][BM+LDS_PAD], B fragment = s_Q[BK+LDS_PAD][BM]
**Change:** Added transposed Q shared memory `s_Q[BK+LDS_PAD][BM]`  
Filled as: `s_Q[k_local][m_idx] = Q[m_idx][k_local]` (transposed)
B fragment filled with col-major transpose: `b_ptr[j*16+i] = s_Q[k_idx_local][m_off+i]`
**Result:** FAILED — 2048/2048 mismatches, max error huge.

### Attempt 3: Same as attempt 2 (reconfirmed)
**Result:** FAILED — same results.

### Attempt 4: B fragment without col-major transpose
**Change:** `b_ptr[i*16+j]` instead of `b_ptr[j*16+i]`
**Result:** FAILED — dK[0][*] = -4704 constant (same as before).

### Attempt 5: A fragment with col_major layout
**Change:** `rocwmma::matrix_a, ..., rocwmma::col_major` instead of `row_major`
Kept B fill as `b_ptr[i*16+j] = s_Q[k_idx_local][m_off+i]` (no transpose)
**Result:** FAILED — dK[0][*] = -4704 constant.

## Key Observations

1. **dK[0..15][*] = -4704 for ALL k values** in warp_n_idx=0 block. This means all 16 rows of the A fragment produce the SAME output, which should not happen with WMMA.

2. The constant value -4704 appears across multiple attempts even after changing shared memory layout and fragment filling logic.

3. Naive kernel produces reasonable results (max error ~14.5 from CPU ref), confirming the math formula is correct.

4. dQ kernel works with:
   - `s_dS[BM][BN+LDS_PAD]` = [64][72], loaded as s_dS[m][n]
   - `s_K[BK][BN+LDS_PAD]` = [32][72], loaded as s_K[k_idx][n]  
   - A fill: `a_ptr[i*16+j] = s_dS[warp_m_idx*16+i][n_start+n_off+j]`
   - B fill (col-major transpose): `b_ptr[j*16+i] = s_K[warp_k_idx*16+j][n_off+i]`

5. For dK, the mathematical equivalent should be:
   - A from dS^T: rows=n, cols=m → `s_dS[n_local][m_off+j]`, fill as `a_ptr[i*16+j]`
   - B from Q^T: rows=k, cols=m → `s_Q[k_idx_local][m_off+i]`, fill with col-major transpose

6. **Root cause suspected:** rocWMMA's handling of matrix_a row_major on AMD GCN may not match the expected row-major layout when using raw pointer access via reinterpret_cast. The internal fragment storage is in packed registers, and element order/access pattern may differ from simple 2D array indexing.

## Next Steps
- Study rocWMMA source code to understand how matrix_a row_major fragments are stored internally
- Try using rocWMMA's load/store/transform functions instead of direct pointer access
- Consider using col_major for both A and B with appropriate data layout
- May need to transpose the entire computation: dK = (Q^T @ dS)^T and use dQ kernel pattern directly

## Current dK Kernel State (src/kernel/correct/fmha_bwd_dk_kernel.hpp)
Last attempt used col_major for FragA, row_major for FragB.
Both attempts produce identical wrong output: constant -4704 per warp block.

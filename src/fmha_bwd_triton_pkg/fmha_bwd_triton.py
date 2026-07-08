"""Triton JIT implementations of all 5 FMHA backward kernels for AMD RDNA3.

This module provides a drop-in replacement for the HIP+WMMA kernels in
`fmha_bwd_kernels.hip`. All five ops are fused into a single callable:

    dQ, dK, dV = fmha_backward_triton(Q, K, V, P, dO)

which matches the signature of `fmha_bwd_full_py()` from the HIP module.

Design notes
------------
- Tile sizes follow RDNA3 WMMA tile conventions: TILE_N=64 (power-of-2).
- Arbitrary M/N/K_dim handled via loop over chunks of TILE_N/BK with boundary masks.
- All kernels operate in float16/bfloat16; intermediate accumulation is fp32.
- For correctness validation, compare against PyTorch einsum reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ==============================================================================
# Tile-size constants (must match fmha_bwd_config.h)
# TILE_M=64, TILE_N=64 are power-of-2 ✓, BK=32 is power-of-2 ✓
# ==============================================================================
TILE_M = 64
TILE_N = 64
BK = 32

# ==============================================================================
# Helper: softmax backward — dS = P ⊙ (dP − rowsum(dP * P))
# ==============================================================================

@triton.jit
def _softmax_bwd_kernel(
    P_ptr,       # [M, N]  — attention probs
    dP_ptr,      # [M, N]  — score gradients
    dS_ptr,      # [M, N]  — output: softmax backward gradient
    M: tl.constexpr,
    N: tl.constexpr,
):
    """Each program handles one row. Uses a single-pass reduction."""
    pid = tl.program_id(0)
    if pid >= M:
        return

    # Initialize accumulators in float32 for numerical stability
    acc_p = tl.zeros((64,), dtype=tl.float32)
    acc_dp = tl.zeros((64,), dtype=tl.float32)
    acc_product = tl.zeros((64,), dtype=tl.float32)

    # Process row in chunks of 64
    for start_n in range(0, N, 64):
        offsets = start_n + tl.arange(0, 64)
        mask = offsets < N

        p_val = tl.load(P_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)
        dp_val = tl.load(dP_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)

        # Store values for debugging (optional)
        # tl.device_print("softmax_bwd_p", p_val)
        # tl.device_print("softmax_bwd_dp", dp_val)

        product = p_val * dp_val
        # tl.device_print("softmax_bwd_product", product)

        # Accumulate for row-wise sum
        acc_p += p_val
        acc_dp += dp_val
        acc_product += product

    # Compute final sums (reduce across chunks)
    p_sum = tl.sum(acc_p, axis=0)
    dp_sum = tl.sum(acc_dp, axis=0)
    product_sum = tl.sum(acc_product, axis=0)

    # For softmax backward, we need: dS = P * (dP - mean(P * dP))
    # where mean is over the row: sum(P * dP) / N
    # Actually, looking at the standard formula:
    # dS_i = P_i * (dP_i - sum_j(P_j * dP_j))
    # Note: no division by N!, this is correct for softmax backward

    # We computed product_sum = sum_j(P_j * dP_j) for the entire row
    # Now compute: dS_i = P_i * (dP_i - product_sum)

    # Reload P and dP values to compute final result (or we could have saved them)
    # Better approach: compute in one pass without storing intermediate values
    # Let me restart and do it properly

    # Actually, let me recompute in a second pass to avoid storage issues
    # First pass: compute sum_j(P_j * dP_j)
    product_sum = 0.0
    for start_n in range(0, N, 64):
        offsets = start_n + tl.arange(0, 64)
        mask = offsets < N

        p_val = tl.load(P_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)
        dp_val = tl.load(dP_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)

        product_sum += tl.sum(p_val * dp_val, axis=0)

    # Second pass: compute dS_i = P_i * (dP_i - product_sum)
    for start_n in range(0, N, 64):
        offsets = start_n + tl.arange(0, 64)
        mask = offsets < N

        p_val = tl.load(P_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)
        dp_val = tl.load(dP_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)

        ds_val = p_val * (dp_val - product_sum)
        tl.store(dS_ptr + pid * N + offsets, ds_val.to(tl.float16), mask=mask)


# ==============================================================================
# Helper: tiled matmul kernel — D = A @ B  [M,N] x [N,Kdim] → [M,Kdim]
# Uses tl.dot with chunked reduction over the inner dimension.
# ==============================================================================

@triton.jit
def _matmul_kernel(
    A_ptr,      # [M, N] — left operand (input)
    B_ptr,      # [N, Kdim_padded] — right operand (K_dim padded to multiple of BK=32)
    D_ptr,      # [M, Kdim_padded] — output
    M: tl.constexpr,
    N: tl.constexpr,
    Kdim_padded: tl.constexpr,
):
    """Tiled matmul with tl.dot. Tile along reduction dim in BK-sized chunks."""
    pid_m = tl.program_id(0)   # M-tile index
    pid_k = tl.program_id(1)   # K-dim tile index

    TILE_M: tl.constexpr = 64
    TILE_N: tl.constexpr = 64
    BK: tl.constexpr = 32

    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    k_offsets = pid_k * BK + tl.arange(0, BK)

    m_mask = m_offsets < M
    k_mask = k_offsets < Kdim_padded

    acc_dtype = tl.float32
    acc = tl.zeros((TILE_M, BK), dtype=acc_dtype)

    for n_start in range(0, N, TILE_N):
        n_offsets = n_start + tl.arange(0, TILE_N)
        n_mask = n_offsets < N

        a_tile = tl.load(A_ptr + m_offsets[:, None] * N + n_offsets[None, :],
                         mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        b_tile = tl.load(B_ptr + n_offsets[:, None] * Kdim_padded + k_offsets[None, :],
                         mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        acc += tl.dot(a_tile.to(acc_dtype), b_tile.to(acc_dtype))

    result = acc.to(tl.float16)
    tl.store(D_ptr + m_offsets[:, None] * Kdim_padded + k_offsets[None, :],
             result, mask=m_mask[:, None] & k_mask[None, :])


# ==============================================================================
# Public API — single fused call matching fmha_bwd_full_py()
# ==============================================================================

def _launch_1d(kernel, grid_size: int, *args):
    """Launch a 1-D kernel."""
    kernel[(grid_size,), *args]


def _launch_2d(kernel, grid_m: int, grid_n: int, *args):
    """Launch a 2-D kernel."""
    kernel[(grid_m, grid_n), *args]


def fmha_backward_triton(
    Q: torch.Tensor,   # [B,H,M,K_dim] — query
    K: torch.Tensor,   # [B,H,N,K_dim] — key
    V: torch.Tensor,   # [B,H,N,K_dim] — value
    P: torch.Tensor,   # [B,H,M,N] — attention probs from forward pass
    dO: torch.Tensor,  # [B,H,M,K_dim] — output gradient
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full FMHA backward pass on GPU using Triton JIT kernels.

    Parameters are expected to be contiguous tensors on the same device with dtype float16/bfloat16.
    Returns (dQ, dK, dV).
    """
    assert Q.dtype in (torch.float16, torch.bfloat16), f"Unsupported dtype: {Q.dtype}"
    assert K.dtype == Q.dtype and V.dtype == Q.dtype
    assert P.dtype == Q.dtype and dO.dtype == Q.dtype

    batch, heads, M, K_dim = Q.shape
    _, _, N, _ = K.shape

    # Pad K_dim to multiple of BK=32 for tl.dot alignment
    Kdim_padded = ((K_dim + BK - 1) // BK) * BK  # round up to multiple of 32

    # Flatten batch × heads for per-head kernel launch (matches HIP blockIdx.z convention)
    flat_q = Q.reshape(-1, M, K_dim).contiguous()      # [B*H, M, K]
    flat_k = K.reshape(-1, N, K_dim).contiguous()
    flat_v = V.reshape(-1, N, K_dim).contiguous()
    flat_p = P.reshape(-1, M, N).contiguous()           # [B*H, M, N]
    flat_do = dO.reshape(-1, M, K_dim).contiguous()

    num_heads_flat = flat_q.size(0)  # batch * heads

    dQ_list: list[torch.Tensor] = []
    dK_list: list[torch.Tensor] = []
    dV_list: list[torch.Tensor] = []

    for h in range(num_heads_flat):
        dS = torch.empty((M, N), dtype=Q.dtype, device=Q.device)
        dQ_h = torch.zeros((M, K_dim), dtype=Q.dtype, device=Q.device)
        dK_h = torch.zeros((N, Kdim_padded), dtype=Q.dtype, device=Q.device)  # padded for tl.dot
        dV_h = torch.zeros((N, Kdim_padded), dtype=Q.dtype, device=Q.device)

        q_h = flat_q[h]       # [M, K]
        k_h = flat_k[h]       # [N, K] — stored as padded for tl.dot
        p_h = flat_p[h]       # [M, N]
        do_h = flat_do[h]     # [M, K]

        # Pad K and V to Kdim_padded (fill with zeros)
        k_padded = torch.zeros((N, Kdim_padded), dtype=Q.dtype, device=Q.device)
        k_padded[:, :K_dim] = k_h
        v_padded = torch.zeros((N, Kdim_padded), dtype=Q.dtype, device=Q.device)
        v_padded[:, :K_dim] = flat_v[h]

        # --- Step 1: dP = dO @ V^T  (via torch.matmul for speed) ---
        # do_h: [M,K], v_padded[:N,:K_dim]: [N,K] -> need v_padded.T: [K,N]
        dP_h = torch.mm(do_h.to(torch.float32), v_padded[:, :K_dim].to(torch.float32).t())

        # --- Step 2: softmax backward → dS = P ⊙ (dP - rowsum(dP*P)) ---
        _launch_1d(_softmax_bwd_kernel, M, p_h, dP_h.to(Q.dtype), dS, M, N)

        # --- Step 3: dQ = dS @ K_padded ---
        grid_dq_m = (M + TILE_M - 1) // TILE_M
        grid_dq_k = (Kdim_padded + BK - 1) // BK
        _launch_2d(_matmul_kernel, grid_dq_m, grid_dq_k, dS, k_padded, dQ_h, M, N, Kdim_padded)

        # --- Step 4: dK = dS^T @ Q padded ---
        q_padded = torch.zeros((M, Kdim_padded), dtype=Q.dtype, device=Q.device)
        q_padded[:, :K_dim] = q_h
        grid_dk_n = (N + TILE_M - 1) // TILE_M
        grid_dk_k = (Kdim_padded + BK - 1) // BK
        _launch_2d(_matmul_kernel, grid_dk_n, grid_dk_k, dS.transpose(0, 1), q_padded, dK_h, N, M, Kdim_padded)

        # --- Step 5: dV = P^T @ dO padded ---
        do_padded = torch.zeros((M, Kdim_padded), dtype=Q.dtype, device=Q.device)
        do_padded[:, :K_dim] = do_h
        grid_dv_n = (N + TILE_M - 1) // TILE_M
        grid_dv_k = (Kdim_padded + BK - 1) // BK
        _launch_2d(_matmul_kernel, grid_dv_n, grid_dv_k, p_h.transpose(0, 1), do_padded, dV_h, N, M, Kdim_padded)

        # Truncate dK/dV back to original K_dim (remove padded columns)
        dQ_list.append(dQ_h[:, :K_dim])
        dK_list.append(dK_h[:, :K_dim])
        dV_list.append(dV_h[:, :K_dim])

    # Stack back to [B,H,M,K] and [B,H,N,K] shapes
    dQ = torch.stack(dQ_list).reshape(Q.shape)
    dK = torch.stack(dK_list).reshape(K.shape)
    dV = torch.stack(dV_list).reshape(V.shape)

    return dQ, dK, dV


# ==============================================================================
# Convenience: one-call wrapper matching the HIP C API signature
# ==============================================================================

def fmha_bwd_triton_single(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    P: torch.Tensor,
    dO: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-call wrapper for benchmarking / profiling.

    Identical signature to the HIP `fmha_bwd_full_py()` entry point.
    Accepts pre-flattened tensors and returns (dQ, dK, dV).
    """
    return fmha_backward_triton(Q, K, V, P, dO)

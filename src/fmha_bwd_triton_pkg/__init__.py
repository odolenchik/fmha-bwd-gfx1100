"""Triton-based FMHA backward kernels for AMD RDNA3 (gfx1100).

Provides Triton JIT implementations of all 5 backward ops:
  dP = dO @ V^T        — attention score gradients
  dS = P ⊙ (dP − rowsum) — softmax backward
  dQ = dS @ K           — query gradient
  dK = dS^T @ Q         — key gradient
  dV = P^T @ dO         — value gradient

Usage:
    from src.fmha_bwd_triton_pkg import fmha_backward_triton
    dQ, dK, dV = fmha_backward_triton(Q, K, V, P, dO)
"""

from .fmha_bwd_triton import fmha_backward_triton

__all__ = ["fmha_backward_triton"]

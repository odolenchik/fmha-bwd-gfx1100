#!/usr/bin/env python3
"""Benchmark custom FMHA backward against PyTorch einsum.

Works on both ROCm (AMD) and CUDA (NVIDIA) backends.
Usage:
    python benchmark_pytorch_final.py          # auto-detect device
    HIP_VISIBLE_DEVICES=0 python benchmark_pytorch_final.py  # pick GPU
"""

import torch
import ctypes
import time

# ============================================================================
# Device setup — ROCm or CUDA
# ============================================================================
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    USE_HIP = not hasattr(torch.version, 'hip') or not torch.version.hip
    # If torch was built with ROCm, use hipify-aware sync
    HAS_HIP = hasattr(torch.version, 'hip') and torch.version.hip is not None
else:
    raise RuntimeError("CUDA/ROCm device not available. Install PyTorch with GPU support.")

print(f"Device: {DEVICE}  (HIP={HAS_HIP})")

# ============================================================================
# Load the shared library
# ============================================================================
lib = ctypes.CDLL('./build/libfmha_bwd_bf16.so')

lib.fmha_bwd_full_py.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

def custom_flash_attn_backward(Q, K, V, P, dO):
    """Run the custom HIP C-kernel backward pass."""
    batch, heads, M, K_dim = Q.shape
    _, _, N, _ = K.shape
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    lib.fmha_bwd_full_py(
        dQ.data_ptr(), dK.data_ptr(), dV.data_ptr(),
        Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        P.data_ptr(), dO.data_ptr(),
        M, N, K_dim, heads
    )
    return dQ, dK, dV

# ============================================================================
# PyTorch reference (naive einsum)
# ============================================================================
def pytorch_naive_backward(Q, K, V, P, dO):
    """Reference backward using PyTorch einsum ops."""
    # dP = dO @ V^T  — shape [batch, heads, M, N]
    dP = torch.einsum('bhmk,bhnk->bhmn', dO, V)
    # Softmax backward: dS = P ⊙ (dP − rowsum(dP × P))
    dP_mul_P = dP * P
    rowsum = dP_mul_P.sum(dim=-1, keepdim=True)
    dS = P * (dP - rowsum)
    # Gradients w.r.t. Q, K, V
    dQ = torch.einsum('bhmn,bhnk->bhmk', dS, K)
    dK = torch.einsum('bhnm,bhmk->bhnk', dS.transpose(-2, -1), Q)
    dV = torch.einsum('bhnm,bhmk->bhnk', P.transpose(-2, -1), dO)
    return dQ, dK, dV

# ============================================================================
# Benchmark helper
# ============================================================================
def sync():
    """Synchronize and check for errors."""
    if HAS_HIP:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize()

def benchmark_size(batch, heads, M, N, K_dim, dtype=torch.float16, warmup=5, reps=20):
    """Run a full benchmark for one configuration."""
    device = DEVICE

    # Generate ONE set of data per call (seeded for reproducibility)
    torch.manual_seed(42)
    Q  = torch.randn(batch, heads, M, K_dim, dtype=dtype, device=device)
    K  = torch.randn(batch, heads, N, K_dim, dtype=dtype, device=device)
    V  = torch.randn(batch, heads, N, K_dim, dtype=dtype, device=device)
    dO = torch.randn(batch, heads, M, K_dim, dtype=dtype, device=device)
    P  = torch.softmax(torch.randn(batch, heads, M, N, dtype=dtype, device=device), dim=-1)

    # Warmup — forces kernel launches and memory allocation
    for _ in range(warmup):
        _ = custom_flash_attn_backward(Q, K, V, P, dO)
        _ = pytorch_naive_backward(Q, K, V, P, dO)
    sync()

    # --- Custom HIP kernel ---
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        dQ_custom, dK_custom, dV_custom = custom_flash_attn_backward(Q, K, V, P, dO)
    end.record()
    sync()
    custom_time = start.elapsed_time(end) / reps

    # --- PyTorch reference ---
    start.record()
    for _ in range(reps):
        dQ_pt, dK_pt, dV_pt = pytorch_naive_backward(Q, K, V, P, dO)
    end.record()
    sync()
    pt_time = start.elapsed_time(end) / reps

    # --- Accuracy check ---
    max_diff = max(
        (dQ_custom - dQ_pt).abs().max().item(),
        (dK_custom - dK_pt).abs().max().item(),
        (dV_custom - dV_pt).abs().max().item(),
    )

    flops_per_matmul = 2.0 * M * N * K_dim
    total_flops = flops_per_matmul * 4
    custom_tflops = total_flops / (custom_time * 1e-3) / 1e12 if custom_time > 0 else 0

    return {
        'M': M, 'N': N, 'K': K_dim,
        'custom_time_ms': custom_time,
        'pytorch_time_ms': pt_time,
        'speedup': pt_time / custom_time if custom_time > 0 else 0,
        'max_diff': max_diff,
        'custom_tflops': custom_tflops,
    }

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print(f"\n{'='*72}")
    print(f"FMHA Backward Benchmark — {DEVICE}")
    print(f"{'='*72}\n")

    configs = [
        (1, 16, 256,   256,   64),
        (1, 16, 512,   512,   64),
        (1, 16, 1024,  1024,  64),
        (1, 8,  2048,  2048,  64),
    ]

    print(f"{'M':>5} {'N':>5} {'K':>3} | Custom(ms) | PyTorch(ms) | Speedup | TFLOPS | Max diff")
    print("-" * 72)

    for batch, heads, M, N, K_dim in configs:
        res = benchmark_size(batch, heads, M, N, K_dim)
        print(
            f"{res['M']:>5} {res['N']:>5} {res['K']:>3} | "
            f"{res['custom_time_ms']:>9.3f} | "
            f"{res['pytorch_time_ms']:>10.3f} | "
            f"{res['speedup']:>7.2f}x  | "
            f"{res['custom_tflops']:>6.2f} | "
            f"{res['max_diff']:.6f}"
        )

    print(f"\n{'='*72}\n")

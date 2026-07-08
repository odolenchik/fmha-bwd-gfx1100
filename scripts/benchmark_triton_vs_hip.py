#!/usr/bin/env python3
"""Benchmark Triton FMHA backward vs HIP+WMMA kernels.

Usage:
    cd /home/odolen/fmha-bwd-gfx1100 && python scripts/benchmark_triton_vs_hip.py [--hip-only | --triton-only]
"""

import torch
import ctypes
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Load HIP shared library (already compiled)
# ---------------------------------------------------------------------------
HIP_LIB_PATH = Path(__file__).parent.parent / "build" / "libfmha_bwd.so"

_lib_hip = None
if HIP_LIB_PATH.exists():
    _lib_hip = ctypes.CDLL(str(HIP_LIB_PATH))
    _lib_hip.fmha_bwd_full_py.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

# ---------------------------------------------------------------------------
# Triton import (same module we just wrote)
# ---------------------------------------------------------------------------
import sys
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))
from src.fmha_bwd_triton_pkg import fmha_backward_triton


def _hip_custom_backward(Q, K, V, P, dO):
    """Run HIP kernel backward via ctypes call."""
    batch, heads, M, K_dim = Q.shape
    _, _, N, _ = K.shape

    # Flatten for HIP API (expects per-head tensors)
    flat_q = Q.reshape(-1, M, K_dim).contiguous()
    flat_k = K.reshape(-1, N, K_dim).contiguous()
    flat_v = V.reshape(-1, N, K_dim).contiguous()
    flat_p = P.reshape(-1, M, N).contiguous()
    flat_do = dO.reshape(-1, M, K_dim).contiguous()

    num_heads = flat_q.size(0)
    dQ_list, dK_list, dV_list = [], [], []

    for h in range(num_heads):
        dq_h = torch.empty((M, K_dim), dtype=Q.dtype, device="cuda")
        dk_h = torch.empty((N, K_dim), dtype=Q.dtype, device="cuda")
        dv_h = torch.empty((N, K_dim), dtype=Q.dtype, device="cuda")

        _lib_hip.fmha_bwd_full_py(
            dq_h.data_ptr(), dk_h.data_ptr(), dv_h.data_ptr(),
            flat_q[h].data_ptr(), flat_k[h].data_ptr(), flat_v[h].data_ptr(),
            flat_p[h].data_ptr(), flat_do[h].data_ptr(),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K_dim),
            ctypes.c_int(heads),
        )
        dQ_list.append(dq_h)
        dK_list.append(dk_h)
        dV_list.append(dv_h)

    return (torch.stack(dQ_list).reshape(Q.shape),
            torch.stack(dK_list).reshape(K.shape),
            torch.stack(dV_list).reshape(V.shape))


def _sync():
    if hasattr(torch.version, 'hip') and torch.version.hip:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# PyTorch reference (einsum) for accuracy check
# ---------------------------------------------------------------------------
def pytorch_einsum_backward(Q, K, V, P, dO):
    dP = torch.einsum("bhmk,bhnk->bhmn", dO, V)
    rowsum = (dP * P).sum(dim=-1, keepdim=True)
    dS = P * (dP - rowsum)
    dQ = torch.einsum("bhmn,bhnk->bhmk", dS, K)
    dK = torch.einsum("bhnm,bhmk->bhnk", dS.transpose(-2, -1), Q)
    dV = torch.einsum("bhnm,bhmk->bhnk", P.transpose(-2, -1), dO)
    return dQ, dK, dV


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def benchmark_size(batch, heads, M, N, K_dim, backend="both", warmup=5, reps=20):
    """Run benchmark for one configuration."""
    dtype = torch.float16
    device = "cuda"

    torch.manual_seed(42)
    Q  = torch.randn(batch, heads, M, K_dim, dtype=dtype, device=device)
    K  = torch.randn(batch, heads, N, K_dim, dtype=dtype, device=device)
    V  = torch.randn(batch, heads, N, K_dim, dtype=dtype, device=device)

    S_logits = torch.randn(batch, heads, M, N, dtype=torch.float32, device=device)
    P = torch.softmax(S_logits, dim=-1).to(dtype)

    dO = torch.randn(batch, heads, M, K_dim, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        if backend in ("both", "hip"):
            _hip_custom_backward(Q, K, V, P, dO)
        if backend in ("both", "triton"):
            fmha_backward_triton(Q, K, V, P, dO)

    _sync()

    results = {}

    # --- HIP kernel ---
    if backend in ("both", "hip"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(reps):
            _hip_custom_backward(Q, K, V, P, dO)
        end.record()
        _sync()
        hip_time_ms = start.elapsed_time(end) / reps
        results["hip_time_ms"] = hip_time_ms

    # --- Triton kernel ---
    if backend in ("both", "triton"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(reps):
            fmha_backward_triton(Q, K, V, P, dO)
        end.record()
        _sync()
        triton_time_ms = start.elapsed_time(end) / reps
        results["triton_time_ms"] = triton_time_ms

    # --- Accuracy check (against einsum reference once per config) ---
    if backend in ("both", "hip"):
        _, _, _ = _hip_custom_backward(Q, K, V, P, dO)
        hip_dQ, hip_dK, hip_dV = pytorch_einsum_backward(Q, K, V, P, dO), None, None

    if backend in ("both", "triton"):
        triton_dQ, triton_dK, triton_dV = fmha_backward_triton(Q, K, V, P, dO)
        pt_dQ, pt_dK, pt_dV = pytorch_einsum_backward(Q, K, V, P, dO)

        dq_diff = (triton_dQ.float() - pt_dQ.float()).abs().max().item()
        dk_diff = (triton_dK.float() - pt_dK.float()).abs().max().item()
        dv_diff = (triton_dV.float() - pt_dV.float()).abs().max().item()

        results["dq_err"] = dq_diff
        results["dk_err"] = dk_diff
        results["dv_err"] = dv_diff

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FMHA backward benchmark")
    parser.add_argument("--hip-only", action="store_true", help="Benchmark HIP kernels only")
    parser.add_argument("--triton-only", action="store_true", help="Benchmark Triton kernels only")
    args = parser.parse_args()

    if args.hip_only and args.triton_only:
        print("Cannot use both --hip-only and --triton-only.")
        return 1

    backend_map = {"hip": "hip" if not args.triton_only else "both", "triton": "triton"}
    if args.hip_only:
        backend = "hip"
    elif args.triton_only:
        backend = "triton"
    else:
        backend = "both"

    print(f"\n{'='*72}")
    print(f"FMHA Backward Benchmark — {torch.cuda.get_device_name(0)}")
    print(f"Triton version: {torch.__version__}  (HIP={getattr(torch.version, 'hip', 'N/A')})")
    print(f"Backend mode: {backend}")
    print(f"{'='*72}\n")

    configs = [
        (1, 16, 64,   64,   32),
        (1, 16, 128,  128,  64),
        (1, 8,  256,  256,  64),
    ]

    headers = []
    if backend == "both":
        headers.append("M")
        headers.append("N")
        headers.append("K_dim")
        headers.append("HIP(ms)")
        headers.append("Triton(ms)")
        headers.append("Speedup")
        headers.append("dQ_err")
    elif backend == "hip":
        headers = ["M", "N", "K_dim", "HIP(ms)"]
    else:  # triton only
        headers.extend(["M", "N", "K_dim", "Triton(ms)", "dQ_err"])

    print(" | ".join(headers))
    print("-" * len(headers[0]) * 12)

    for M, N, K_dim in configs:
        res = benchmark_size(1, 8, M, N, K_dim, backend=backend)

        row = []
        if backend == "both":
            hip_t = res.get("hip_time_ms", float("inf"))
            tri_t = res.get("triton_time_ms", float("inf"))
            speedup = hip_t / tri_t if tri_t > 0 else float("inf")
            row.extend([f"{M:>5}", f"{N:>5}", f"{K_dim:>3d}",
                        f"{hip_t:<9.3f}", f"{tri_t:<10.3f}", f"{speedup:>7.2f}x",
                        f"{res.get('dq_err', 0):.2e}"])
        elif backend == "hip":
            row.extend([f"{M:>5}", f"{N:>5}", f"{K_dim:>3d}",
                        f"{res['hip_time_ms']:.3f}"])
        else:  # triton
            row.extend([f"{M:>5}", f"{N:>5}", f"{K_dim:>3d}",
                        f"{res['triton_time_ms']:.3f}",
                        f"{res.get('dq_err', float('inf')):.2e}"])

        print(" | ".join(row))

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()

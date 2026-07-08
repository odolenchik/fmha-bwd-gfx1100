#!/usr/bin/env python3
"""Debug dq_kernel"""

import torch
import ctypes
import os

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), 'build', 'libfmha_bwd.so')
lib = ctypes.CDLL(lib_path)

# Define the test_dq_kernel signature
lib.test_dq_kernel.argtypes = [
    ctypes.c_void_p,  # dS
    ctypes.c_void_p,  # K
    ctypes.c_void_p,  # dQ
    ctypes.c_int,     # M
    ctypes.c_int,     # N (actually N_global)
    ctypes.c_int,     # K_dim
    ctypes.c_int      # total_heads
]
lib.test_dq_kernel.restype = None

def test_dq_kernel(M=4, N=4, K_dim=2, heads=1, seed=42):
    torch.manual_seed(seed)
    # Create input tensors on GPU
    dS = torch.randn(heads, M, N, dtype=torch.bfloat16, device='cuda')
    K  = torch.randn(heads, N, K_dim, dtype=torch.bfloat16, device='cuda')
    # Allocate output
    dQ_custom = torch.zeros(heads, M, K_dim, dtype=torch.bfloat16, device='cuda')

    print(f"dS[0]: {dS[0]}")
    print(f"K[0]: {K[0]}")

    # Call the kernel via wrapper
    lib.test_dq_kernel(
        ctypes.c_void_p(dS.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(dQ_custom.data_ptr()),
        M, N, K_dim, heads
    )
    # Synchronize
    torch.cuda.synchronize()

    # Reference computation using torch.einsum or bmm
    # dS: [heads, M, N], K: [heads, N, K] -> dQ: [heads, M, K] = sum_n dS[:,:,n] * K[:,n,:]^T? Actually dQ[m][k] = sum_n dS[m][n] * K[n][k]
    # So per head: dQ[h] = dS[h] @ K[h]  (M,N) @ (N,K) -> (M,K)
    dQ_ref = torch.zeros_like(dQ_custom)
    for h in range(heads):
        dQ_ref[h] = torch.matmul(dS[h], K[h])  # (M,N) @ (N,K) = (M,K)

    print(f"dQ_custom[0]: {dQ_custom[0]}")
    print(f"dQ_ref[0]: {dQ_ref[0]}")

    # Compare
    diff = (dQ_custom.float() - dQ_ref.float()).abs()
    max_diff = diff.max().item()
    print(f"Test dq_kernel: M={M}, N={N}, K={K_dim}, heads={heads}")
    print(f"  Max absolute difference: {max_diff}")
    print(f"  Max diff location: {diff.argmax()}")
    print(f"  Custom dQ[0,0,:5]: {dQ_custom[0,0,:min(5,K_dim)].to(torch.float32).cpu().numpy()}")
    print(f"  Ref    dQ[0,0,:5]: {dQ_ref[0,0,:min(5,K_dim)].to(torch.float32).cpu().numpy()}")
    if max_diff < 1e-2:
        print("  PASS")
    else:
        print("  FAIL")
    return max_diff

if __name__ == '__main__':
    test_dq_kernel()

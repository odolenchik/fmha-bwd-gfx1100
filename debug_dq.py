import torch
import ctypes
import os
import numpy as np

lib_path = os.path.join(os.path.dirname('.'), 'build', 'libfmha_bwd_bf16.so')
lib = ctypes.CDLL(lib_path)
lib.test_dq_kernel.argtypes = [
    ctypes.c_void_p,  # dS
    ctypes.c_void_p,  # K
    ctypes.c_void_p,  # dQ
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K_dim
    ctypes.c_int      # total_heads
]
lib.test_dq_kernel.restype = None

def test_dq_kernel(M=2, N=2, K_dim=2, heads=1, seed=42):
    torch.manual_seed(seed)
    dS = torch.randn(heads, M, N, dtype=torch.bfloat16, device='cuda')
    K  = torch.randn(heads, N, K_dim, dtype=torch.bfloat16, device='cuda')
    dQ = torch.zeros(heads, M, K_dim, dtype=torch.bfloat16, device='cuda')
    print('dS[0]:')
    print(dS[0].float().cpu().numpy())
    print('K[0]:')
    print(K[0].float().cpu().numpy())
    lib.test_dq_kernel(
        ctypes.c_void_p(dS.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(dQ.data_ptr()),
        M, N, K_dim, heads
    )
    torch.cuda.synchronize()
    print('dQ result:')
    print(dQ[0].float().cpu().numpy())
    # reference
    dQ_ref = torch.zeros_like(dQ)
    for h in range(heads):
        dQ_ref[h] = torch.matmul(dS[h], K[h])
    print('dQ reference:')
    print(dQ_ref[0].float().cpu().numpy())
    diff = (dQ[0].float() - dQ_ref[0].float()).abs().max().item()
    print(f'Max abs diff: {diff}')
    return diff

if __name__ == '__main__':
    test_dq_kernel()

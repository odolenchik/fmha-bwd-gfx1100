#!/usr/bin/env python3
import torch
import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), 'build', 'libfmha_bwd.so')
lib = ctypes.CDLL(lib_path)

# Define the function signatures
lib.fmha_bwd_full_py.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.fmha_bwd_full_py.restype = None

def test_full():
    torch.manual_seed(42)
    batch = 1
    heads = 2
    M = 4
    N = 4
    K_dim = 4
    device = 'cuda'

    Q = torch.randn(batch, heads, M, K_dim, dtype=torch.bfloat16, device=device)
    K = torch.randn(batch, heads, N, K_dim, dtype=torch.bfloat16, device=device)
    V = torch.randn(batch, heads, N, K_dim, dtype=torch.bfloat16, device=device)
    dO = torch.randn(batch, heads, M, K_dim, dtype=torch.bfloat16, device=device)
    P = torch.softmax(torch.randn(batch, heads, M, N, dtype=torch.bfloat16, device=device), dim=-1)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    lib.fmha_bwd_full_py(
        dQ.data_ptr(), dK.data_ptr(), dV.data_ptr(),
        Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        P.data_ptr(), dO.data_ptr(),
        M, N, K_dim, heads
    )
    torch.cuda.synchronize()

    # Reference
    dP_ref = torch.einsum('bhmk,bhnk->bhmn', dO, V)
    dP_mul_P_ref = dP_ref * P
    rowsum_ref = dP_mul_P_ref.sum(dim=-1, keepdim=True)
    dS_ref = P * (dP_ref - rowsum_ref)
    dQ_ref = torch.einsum('bhmn,bhnk->bhmk', dS_ref, K)
    dK_ref = torch.einsum('bhnm,bhmk->bhnk', dS_ref.transpose(-2, -1), Q)
    dV_ref = torch.einsum('bhnm,bhmk->bhnk', P.transpose(-2, -1), dO)

    print("dP max diff:", (dP_ref - torch.zeros_like(dP_ref)).abs().max())  # dP_ref vs zero? Actually we didn't compute dP from custom
    # We need to extract intermediate dP and dS from custom? Not saved.
    # Let's compute each step by calling the isolated test functions.
    from test_dp_isolated import test_dp_kernel
    from test_softmax_isolated import test_softmax_bwd_kernel
    from test_dq_isolated import test_dq_kernel
    from test_dk_isolated import test_dk_kernel
    from test_dv_isolated import test_dv_kernel

    # Allocate intermediates
    dP = torch.zeros(batch, heads, M, N, dtype=torch.bfloat16, device=device)
    dS = torch.zeros(batch, heads, M, N, dtype=torch.bfloat16, device=device)

    # Step1 dp
    lib.test_dp_kernel(
        dO.data_ptr(), V.data_ptr(), dP.data_ptr(),
        M, N, K_dim, heads
    )
    torch.cuda.synchronize()
    print("Step1 dP max diff vs ref:", (dP - dP_ref).abs().max().item())

    # Step2 softmax
    lib.test_softmax_bwd_kernel(
        P.data_ptr(), dP.data_ptr(), dS.data_ptr(),
        M, N, heads
    )
    torch.cuda.synchronize()
    print("Step2 dS max diff vs ref:", (dS - dS_ref).abs().max().item())

    # Step3 dq
    lib.test_dq_kernel(
        dS.data_ptr(), K.data_ptr(), dQ.data_ptr(),
        M, N, K_dim, heads
    )
    torch.cuda.synchronize()
    print("Step3 dQ max diff vs ref:", (dQ - dQ_ref).abs().max().item())

    # Step4 dk
    lib.test_dk_kernel(
        dS.data_ptr(), Q.data_ptr(), dK.data_ptr(),
        M, N, K_dim, heads
    )
    torch.cuda.synchronize()
    print("Step4 dK max diff vs ref:", (dK - dK_ref).abs().max().item())

    # Step5 dv
    lib.test_dv_kernel(
        P.data_ptr(), dO.data_ptr(), dV.data_ptr(),
        M, N, K_dim, heads
    )
    torch.cuda.synchronize()
    print("Step5 dV max diff vs ref:", (dV - dV_ref).abs().max().item())

    # Full pass
    print("\nFull pass:")
    print("dQ max diff vs ref:", (dQ - dQ_ref).abs().max().item())
    print("dK max diff vs ref:", (dK - dK_ref).abs().max().item())
    print("dV max diff vs ref:", (dV - dV_ref).abs().max().item())

if __name__ == '__main__':
    test_full()
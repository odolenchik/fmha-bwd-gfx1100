import torch
import ctypes
import time
import numpy as np

# ============================================================================
# 1. Загружаем динамическую библиотеку (скомпилирована hipcc)
# ============================================================================
lib = ctypes.CDLL('./build/libfmha_bwd.so')

# Простая C-обёртка без stream
lib.fmha_bwd_full_py.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

def custom_flash_attn_backward(Q, K, V, P, dO):
    batch, heads, M, K_dim = Q.shape
    _, _, N, _ = K.shape
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    lib.fmha_bwd_full_py(
        dQ.data_ptr(), dK.data_ptr(), dV.data_ptr(),
        Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        P.data_ptr(), dO.data_ptr(),
        M, N, K_dim
    )
    return dQ, dK, dV

# ============================================================================
# 2. Наивный PyTorch
# ============================================================================
def pytorch_naive_backward(Q, K, V, P, dO):
    M, N, K_dim = Q.shape[2], K.shape[2], Q.shape[3]
    dP = torch.einsum('bhmk,bhnk->bhmn', dO, V)
    dP_mul_P = dP * P
    rowsum = dP_mul_P.sum(dim=-1, keepdim=True)
    dS = P * (dP - rowsum)
    dQ = torch.einsum('bhmn,bhnk->bhmk', dS, K)
    dK = torch.einsum('bhnm,bhmk->bhnk', dS.transpose(-2, -1), Q)
    dV = torch.einsum('bhnm,bhmk->bhnk', P.transpose(-2, -1), dO)
    return dQ, dK, dV

# ============================================================================
# 3. Бенчмарк
# ============================================================================
def benchmark_size(batch, heads, M, N, K_dim, dtype=torch.float16, warmup=5, reps=20):
    device = torch.device('cuda')
    
    # Генерируем ОДИН набор данных
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, M, K_dim, dtype=dtype, device=device)
    K = torch.randn(batch, heads, N, K_dim, dtype=dtype, device=device)
    V = torch.randn(batch, heads, N, K_dim, dtype=dtype, device=device)
    dO = torch.randn(batch, heads, M, K_dim, dtype=dtype, device=device)
    P = torch.softmax(torch.randn(batch, heads, M, N, dtype=dtype, device=device), dim=-1)

    # Прогрев
    for _ in range(warmup):
        _ = custom_flash_attn_backward(Q, K, V, P, dO)
        _ = pytorch_naive_backward(Q, K, V, P, dO)
    torch.cuda.synchronize()

    # Замер Custom
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        dQ_custom, dK_custom, dV_custom = custom_flash_attn_backward(Q, K, V, P, dO)
    end.record()
    torch.cuda.synchronize()
    custom_time = start.elapsed_time(end) / reps

    # Замер PyTorch
    start.record()
    for _ in range(reps):
        dQ_pt, dK_pt, dV_pt = pytorch_naive_backward(Q, K, V, P, dO)
    end.record()
    torch.cuda.synchronize()
    pt_time = start.elapsed_time(end) / reps

    # Точность (используем последние результаты)
    max_diff = max(
        (dQ_custom - dQ_pt).abs().max().item(),
        (dK_custom - dK_pt).abs().max().item(),
        (dV_custom - dV_pt).abs().max().item()
    )

    flops_per_matmul = 2.0 * M * N * K_dim
    total_flops = flops_per_matmul * 4
    custom_tflops = total_flops / (custom_time * 1e-3) / 1e12

    return {
        'M': M, 'N': N, 'K': K_dim,
        'custom_time_ms': custom_time,
        'pytorch_time_ms': pt_time,
        'speedup': pt_time / custom_time,
        'max_diff': max_diff,
        'custom_tflops': custom_tflops
    }

if __name__ == "__main__":
    print("Running benchmarks...\n")
    configs = [
        (1, 16, 256, 256, 64),
        (1, 16, 512, 512, 64),
        (1, 16, 1024, 1024, 64),
        (1, 8, 2048, 2048, 64),
    ]
    for batch, heads, M, N, K_dim in configs:
        res = benchmark_size(batch, heads, M, N, K_dim)
        print(f"M={M:4d}, N={N:4d}, K={K_dim:2d}: "
              f"Custom: {res['custom_time_ms']:.3f} ms, "
              f"PyTorch: {res['pytorch_time_ms']:.3f} ms, "
              f"Speedup: {res['speedup']:.2f}x, "
              f"TFLOPS: {res['custom_tflops']:.2f}, "
              f"Max diff: {res['max_diff']:.6f}")

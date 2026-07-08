import torch
import ctypes
import argparse

def bench_lib(lib_path, name):
    print(f"\nBenchmarking {name} using {lib_path}")
    lib = ctypes.CDLL(lib_path)
    # Define test_dp_kernel signature
    lib.test_dp_kernel.argtypes = [
        ctypes.c_void_p,  # dO
        ctypes.c_void_p,  # V
        ctypes.c_void_p,  # dP
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K_dim
        ctypes.c_int      # total_heads
    ]
    lib.test_dp_kernel.restype = None

    # Problem size
    M_sz, N_sz, K_sz, heads = 256, 256, 64, 16
    torch.manual_seed(0)
    dO = torch.randn(heads, M_sz, K_sz, dtype=torch.bfloat16, device='cuda')
    V  = torch.randn(heads, N_sz, K_sz, dtype=torch.bfloat16, device='cuda')
    dP = torch.zeros(heads, M_sz, N_sz, dtype=torch.bfloat16, device='cuda')

    # Warmup
    for _ in range(10):
        lib.test_dp_kernel(
            ctypes.c_void_p(dO.data_ptr()),
            ctypes.c_void_p(V.data_ptr()),
            ctypes.c_void_p(dP.data_ptr()),
            M_sz, N_sz, K_sz, heads
        )
    torch.cuda.synchronize()

    # Timing
    repeat = 100
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        lib.test_dp_kernel(
            ctypes.c_void_p(dO.data_ptr()),
            ctypes.c_void_p(V.data_ptr()),
            ctypes.c_void_p(dP.data_ptr()),
            M_sz, N_sz, K_sz, heads
        )
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)  # milliseconds
    avg_ms = elapsed_ms / repeat
    print(f"  Average time per dp_kernel launch: {avg_ms:.3f} ms")
    print(f"  Total time for {repeat} runs: {elapsed_ms:.3f} ms")
    return avg_ms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', required=True, help='Path to libfmha_bwd_bf16.so')
    parser.add_argument('--name', required=True, help='Name of variant')
    args = parser.parse_args()
    avg = bench_lib(args.lib, args.name)
    # Optionally write to file
    with open('dp_benchmark_results.txt', 'a') as f:
        f.write(f"{args.name}: {avg:.3f} ms per call\n")

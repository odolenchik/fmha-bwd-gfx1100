# fmha-bwd-gfx1100

**High-performance FlashAttention backward kernels for AMD RDNA3 (gfx1100)** using WMMA instructions — optimized for LoRA/DoRA training on Radeon RX 7900 XTX.

## Overview

This project implements a complete backward pass for Multi-Head Attention (MHA) as five fused HIP/WGMMa kernels:

| Kernel | Operation | Description |
|--------|-----------|-------------|
| `dp_kernel` | dP = dO @ V^T | Gradient of attention scores |
| `softmax_bwd_kernel` | dS = P ⊙ (dP − rowsum) | Softmax backward |
| `dq_kernel` | dQ = dS @ K | Query gradient via WMMA |
| `dk_kernel` | dK = dS^T @ Q | Key gradient via WMMA |
| `dv_kernel` | dV = P^T @ dO | Value gradient (naive) |

All kernels support arbitrary head counts via `blockIdx.z` and operate at BF16 precision. The unified entry point is `fmha_bwd_full_py()` — one C-call launches all five stages in sequence with shared intermediate buffers.

## Architecture

```
src/
├── fmha_bwd_kernels.hip          # All 5 fused kernels + C API (340 lines)
├── fmha_bwd_kernels.hpp          # Header declarations
└── kernel/correct/               # WMMA-specific modules
    ├── fmha_bwd_config.h         # Tile params: BM=64, BN=64, BK=32, LDS_PAD=8
    ├── fmha_bwd_dq_kernel.hpp    # dQ via WMMA (verified correct)
    ├── fmha_bwd_dk_kernel.hpp    # dK via WMMA (col_major FragB fixed)
    ├── fmha_bwd_dv_kernel.hpp    # dV via WMMA (col_major FragB fixed)
    ├── fmha_bwd_dp_kernel.hpp    # DP score computation
    ├── fmha_bwd_softmax_kernel.hpp  # Softmax backward
    ├── fmha_bwd_utils.hip        # HIP_CHECK / sync helpers
    └── cpu_reference.hpp         # CPU reference for validation
```

### WMMA Tile Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| BM (M-tile) | 64 | Rows per thread-block tile |
| BN (N-tile) | 64 | Columns per thread-block tile |
| BK (K-chunk) | 32 | WMMA inner dimension chunk size |
| LDS_PAD | 8 | Shared memory padding for bank conflict avoidance |
| BLOCK_SIZE | 512 | Threads per block (8 warps × 64) |

### Warp Mapping (512-thread block, 64 threads/warp = 8 warps)

```
warp_id   = tid / 64            // 0..7
warp_m    = warp_id % 4         // row sub-tile index (0..3) × 16
warp_k    = warp_id / 4         // col sub-tile index (0..1) × 16
```

Each warp computes a 16×16 output tile via WMMA. The unified pattern across dQ/dK/dV is: **FragA=row_major, FragB=col_major** with col-major transpose during B-fragment fill (`b_ptr[j*16+i] = s_B[k_local][m_off+i]`).

## Requirements

- **GPU**: AMD Radeon RX 7900 XTX (gfx1100) or compatible RDNA3
- **ROCm**: ≥ 7.2
- **CMake**: ≥ 3.18
- **Compiler**: hipcc (AMD HIP SDK)
- **rocWMMA**: Header-only, shipped with ROCm at `/opt/rocm/include`

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This produces `libfmha_bwd.so` — a shared library containing all five fused kernels. The library is compiled for gfx1100 with BF16 math enabled (`-DROCWMMA_ENABLE_BF16=ON`).

## Usage from Python

```python
import ctypes, torch

# Load the compiled library
lib = ctypes.CDLL('./build/libfmha_bwd.so')

# Set up signature
lib.fmha_bwd_full_py.argtypes = [
    ctypes.c_void_p,  # dQ
    ctypes.c_void_p,  # dK
    ctypes.c_void_p,  # dV
    ctypes.c_void_p,  # Q
    ctypes.c_void_p,  # K
    ctypes.c_void_p,  # V
    ctypes.c_void_p,  # P (softmax output)
    ctypes.c_void_p,  # dO (gradient of attention output)
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K_dim
    ctypes.c_int,     # total_heads
]

# Call with BF16 tensors on GPU
lib.fmha_bwd_full_py(
    dQ_ptr, dK_ptr, dV_ptr,
    Q_ptr, K_ptr, V_ptr, P_ptr, dO_ptr,
    M, N, K_dim, total_heads
)
```

Or use the bundled benchmark:

```bash
python3 benchmark_pytorch_final.py
```

## Validation & Testing

The project includes a CPU reference implementation (`cpu_reference.hpp`) for numerical verification. The forward+backward pass is tested against known analytical results.

### Known Status

| Kernel | Status | Notes |
|--------|--------|-------|
| `dp_kernel` | Working | Naive element-per-thread |
| `softmax_bwd_kernel` | Working | Parallel reduction via shared memory |
| `dq_kernel` | Verified | WMMA, error = 0.0 against reference |
| `dk_kernel` | Fixed | WMMA, FragB=col_major (was row_major) |
| `dv_kernel` | Fixed | WMMA, FragB=col_major (was row_major) |

## Performance Target

Optimized for LoRA/DoRA training on FLUX.2 / HV1.5 / WAN2.2 diffusion models where the backward pass bottleneck is MHA gradient computation on RDNA3 GPUs. BF16 precision matches the primary training dtype used by modern LoRA adapters.

## Related Work

- **AITER** (AMD): `fmha_v3_bwd` — known issue with BF16 `.data_ptr()` on ROCm ([#3724](https://github.com/ROCm/aiter/issues/3724))
- **rocWMMA**: Header-only WMMA abstraction for AMD GPUs
- **FlashAttention**: Original algorithm this project implements

## License

MIT — see [LICENSE](./LICENSE)

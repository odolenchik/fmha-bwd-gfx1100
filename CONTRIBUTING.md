# Contributing to fmha-bwd-gfx1100

## Development Workflow

### Prerequisites

- ROCm ≥ 7.2 with hipcc
- CMake ≥ 3.18
- AMD GPU (gfx1100/RDNA3 recommended)

```bash
git clone <repo> && cd fmha-bwd-gfx1100
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Adding a New Kernel

1. Add your kernel to `src/fmha_bwd_kernels.hip` or create a new `.hpp` under `src/kernel/correct/`
2. Update `fmha_bwd_config.h` if you introduce new tile parameters (BM, BN, BK)
3. Ensure your kernel uses the unified pattern: **FragA=row_major, FragB=col_major** for WMMA operations
4. Add a test in Python (`test_*.py`) or HIP (`test_*.hip`) under `src/kernel/correct/cpu_reference.hpp`

### Adding a New Test

1. Create `test_<name>.py` in the project root referencing `cpu_reference.hpp`
2. Verify against known analytical results (not just "doesn't crash")
3. Include both BF16 and FP32 test cases where applicable

## WMMA Conventions

All kernels must follow these conventions for interoperability:

- **Tile layout**: BM rows × BN columns, tiled at 64×64 with BK=32 inner chunks
- **Shared memory padding**: `LDS_PAD=8` to avoid bank conflicts
- **Warp mapping**: 512-thread block → 8 warps (64 threads each), warp_id = tid/64
- **FragB layout**: Always col_major with transpose: `b_ptr[j*16+i] = s_B[k_local][m_off+i]`

## Commit Style

```
feat(dk): fix FragB col_major transpose for WMMA
fix(dv): apply same FragB pattern as dk kernel
docs(readme): add architecture overview and usage examples
```

## Code Review Checklist

- [ ] All new kernels use `fmha_bwd_config.h` constants (no magic numbers)
- [ ] Shared memory padding applied (`LDS_PAD=8`)
- [ ] Warp deadlock-safe (all 512 threads participate in every `__syncthreads()`)
- [ ] BF16 precision maintained through WMMA operations
- [ ] Multi-head support via `blockIdx.z` when applicable

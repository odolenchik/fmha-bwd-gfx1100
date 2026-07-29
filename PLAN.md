# Development Plan: FMHA Backward Kernels for AMD RDNA3 (gfx1100)

## Project Goal
High-performance FlashAttention backward kernels for AMD Radeon RX 7900 XTX (RDNA3, gfx1100) using WMMA instructions — optimized for LoRA/DoRA training.

---

## Current State (Post-Audit, 2026-07-29)

**Production file**: `src/fmha_bwd_kernels.hip` — BF16 kernels with multi-head support
- ✅ Build system works (CMake → `libfmha_bwd_bf16.so`, `libfmha_bwd_fp16.so`)
- ✅ `dp_kernel`, `softmax_bwd_kernel` — naive but functional
- ❌ `dq_kernel`, `dk_kernel`, `dv_kernel` — **still naive element-per-thread** (NOT WMMA!)
- ✅ Reference WMMA implementations exist in `src/kernel/correct/` (verified working with 0.0 diff)

**Key Issue**: The optimized WMMA kernels for dQ/dK/dV are sitting in `kernel/correct/` but **not integrated** into the production file.

**Current Blocking Issue**: The `fmha_bwd_full_py` C API has a bug — dQ/dK produce garbage (dV works perfect). Root cause under investigation (intermediate buffer handling / kernel launch config mismatch).

---

## Phase 1: Integrate WMMA Kernels (IMMEDIATE PRIORITY) 🔥

### Task 1.1: Replace naive dQ/dK/dV with WMMA versions
**File**: `src/fmha_bwd_kernels.hip`
- Copy WMMA implementations from `kernel/correct/fmha_bwd_dq_kernel.hpp`, `dk_kernel.hpp`, `dv_kernel.hpp`
- Adapt signatures to match existing naive kernels (add `total_heads`, `head_start` parameters)
- Add `head_start` parameter + internal loop over `MAX_HEADS_PER_LAUNCH` heads per block
- Preserve existing test wrapper signatures (`head_start=0` default)

### Task 1.2: Update grid dimensions in `fmha_bwd_full_py`
- Z-dimension: `grid_z = (total_heads + MAX_HEADS_PER_LAUNCH - 1) / MAX_HEADS_PER_LAUNCH`
- Pass `head_start=0` to all kernel launches
- Remove `total_heads` from individual kernel calls (handled internally)

### Task 1.3: Verify correctness
```bash
cd build && cmake .. && make -j$(nproc)
python3 tests/test_dq_isolated.py
python3 tests/test_dk_isolated.py
python3 tests/test_dv_isolated.py
python3 tests/test_integration.py
python3 benchmark_pytorch_final.py
```

### Task 1.4: Fix `fmha_bwd_full_py` C API bug
**Current blocker**: dQ/dK produce garbage in full API (dV works). Debug steps:
1. ✅ Verified test wrappers + explicit sync work correctly (max diff 0.03)
2. ✅ Verified kernel launch configs match between test wrappers and full API
3. ❌ Full API `fmha_bwd_full_py` gives dQ/dK garbage (dV perfect)
4. Next: Add debug instrumentation to capture intermediate `dS` after softmax in full API

---

## Phase 2: Head Tiling & Multi-Head Optimization

### Task 2.1: Implement head tiling in all 5 kernels
- **dp_kernel**: Already uses `blockIdx.z`, add `MAX_HEADS_PER_LAUNCH` loop
- **softmax_bwd_kernel**: Restructure from `blockIdx.y` to z-dimension + head loop
- **dq/dk/dv WMMA kernels**: Already designed for head tiling in Phase 1

### Task 2.2: Update launch configurations
- All kernels: z-dimension = `ceil(total_heads / MAX_HEADS_PER_LAUNCH)`
- Each block processes up to 64 heads sequentially

---

## Phase 3: Softmax Kernel Optimization

### Task 3.1: Replace two-pass reduction
- Use warp-level reduce (`__shfl_down_sync`) + block-level shared memory reduce
- Add vectorized loads (`float2`/`float4`) for P and dP
- Use `__ldg` for read-only cache

---

## Phase 4: HIP Graph Launch

### Task 4.1: Create HIP graph for 5-kernel sequence
```cpp
hipGraphCreate(&graph);
hipGraphAddKernelNode(&node1, graph, ..., dp_kernel, ...);
// ... 4 more nodes for softmax, dq, dk, dv
hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
hipGraphLaunch(graphExec, 0);
```

---

## Phase 5: Cleanup & Documentation

### Task 5.1: Already done — removed 40+ redundant files
### Task 5.2: Update README.md with current build/run instructions
### Task 5.3: Create PERFORMANCE_RESULTS.md with benchmark data
### Task 5.4: Bump version in pyproject.toml (MINOR for WMMA integration)

---

## File Reference

### Production (keep)
- `src/fmha_bwd_kernels.hip` — main BF16 kernels (current focus)
- `src/fmha_bwd_kernels.hpp` — C headers
- `src/fmha_bwd_kernels_fp16.hip` — FP16 parallel implementation
- `src/kernel/correct/` — reference WMMA implementations & config
- `tests/test_integration.py` — full backward pass test
- `tests/test_dp_isolated.py` — dp kernel test
- `tests/test_dq_isolated.py` — dq kernel test
- `tests/test_dk_isolated.py` — dk kernel test
- `tests/test_dv_isolated.py` — dv kernel test
- `tests/test_softmax_isolated.py` — softmax test
- `tests/test_all_kernels_isolated.py` — combined test
- `benchmark_pytorch_final.py` — main benchmark
- `CMakeLists.txt` — build config
- `CLAUDE.md` — agent instructions

### Removed (cleanup complete)
- All `.backup`, `.no_test` files
- 20+ debug/verification scripts (`test_dp*.cpp`, `verify_*.cpp`, `debug_*.py`, etc.)
- One-off utility scripts (`fix_extern.py`, `replace_kernels.py`, etc.)
- `STATE.md`, `AITER_BF16_bug_report.md`, `.changeset/`

---

## Next Action (IMMEDIATE)
**Debug and fix `fmha_bwd_full_py` C API**: Add debug instrumentation to capture intermediate `dS` after softmax in the full C API, compare with test-wrapper-sequence results to isolate the kernel/dP/dS handling difference. Once fixed, proceed with Phase 1 Task 1.1 (WMMA integration).
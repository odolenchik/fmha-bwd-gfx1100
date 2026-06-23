# Optimization Plan for FMHA Backward Kernels (AMD ROCm)

**Target GPU:** AMD Radeon RX 7900 XTX (gfx1100/RDNA3)  
**Target Precision:** BF16 (with optional FP16 variant)  
**Goal:** Transform the current naïve element‑wise kernels into high‑performance, fused, memory‑efficient implementations that can approach or exceed PyTorch’s scaled‑dot‑product attention backward pass while preserving numerical correctness.

---

## 1. Prerequisites & Toolchain

| Tool | Purpose | Installation / Invocation |
|------|---------|---------------------------|
| ROCm ≥ 5.7 | HIP runtime, compilers, libraries | Already present in environment |
| hipcc | HIP compiler (based on clang) | `hipcc` |
| rocWMMA | Header‑only WMMA for matrix multiply | `/opt/rocm/include` |
| **rocprofiler** | Kernel‑level profiling (metrics, traces) | `rocprofiler` |
| **omnitrace** | End‑to‑end tracing (CPU+GPU, kernels, memory copies) | `omnitrace` |
| **Kernel Analyzer** (via `rocprofiler` `--kernel-stat`) | ISA analysis, occupancy, VALU/MFMA utilization | `rocprofiler --stats` |
| **LLVM‑MCA** | Static instruction‑level performance prediction | `llvm-mca` |
| **amdllpc** | Offline ISA disassembly & inspection | `amdllpc <object>` |
| **ROCm Debugger (rocgdb)** | Source‑level debugging of kernels | `rocgdb` |
| **ROCm System Profiler** | System‑wide utilization (GPU, PCIe, memory) | `rocm-smi` / `rocprofiler-system` |
| **rocBLAS / MIOpen** | Reference high‑performance GEMM for verification | `rocblas-bench`, `miopen-bench` |
| **HIP Graphs** | Low‑overhead kernel launch sequences | HIP API (`hipGraphCreate`, etc.) |
| **hipEvent** / **hipStreamSynchronize** | Precise timing | HIP runtime |

All tools are assumed to be available via the standard ROCm installation (`/opt/rocm/bin`).

---

## 2. Baseline Measurement

1. **Compile current code (debug)**  
   ```bash
   mkdir -p build_debug && cd build_debug
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make -j$(nproc)
   ```
2. **Run benchmark vs PyTorch**  
   ```bash
   cd ..
   python3 benchmark_pytorch_final.py   # capture baseline times
   ```
3. **Collect profiling data**  
   - **Kernel metrics:**  
     ```bash
     rocprofiler --metrics =sm__warps_active.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,mem__throughput.avg.pct_of_peak_sustained_elapsed \
        -d ./build_debug/libfmha_bwd.so -i 5 -- ./benchmark_pytorch_final.py
     ```
   - **Omnitrace timeline:**  
     ```bash
     omnitrace-python -o omnitrace_out python3 benchmark_pytorch_final.py
     ```
   - **Kernel ISA & occupancy:**  
     ```bash
     rocprofiler --kernel-stat --dump hwcounters.csv -d ./build_debug/libfmha_bwd.so -i 1 -- ./benchmark_pytorch_final.py
     ```

4. **Record baseline numbers** (kernel time, achieved occupancy, memory bandwidth, achieved FLOPS) in a table inside this document.

---

## 3. Roofline Analysis

- Compute **Operational Intensity (OI)** for each kernel analytically (FLOP / byte).  
- Using measured peak memory bandwidth (~960 GB/s) and peak compute (~122.9 TFLOPS BF16) draw the roofline.  
- Determine whether each kernel is **memory‑bound** or **compute‑bound** in its current form.

*Expected result:* All five kernels show OI ≈ 0.3‑0.8 FLOP/Byte → strongly memory‑bound.

---

## 4. Optimization Strategy (per kernel)

The following steps are to be applied iteratively. After each change, re‑run the baseline measurements and compare to the roofline.

### 4.1 Common Optimizations (apply to all kernels where relevant)

| Technique | How to apply | Expected impact |
|-----------|--------------|-----------------|
| **Blocked tiling + shared memory** | Load input tiles (e.g., dO and V for dp_kernel) into `__shared__` buffers, compute outer product, accumulate to output tile. | Increases data reuse → raises OI from ~0.5 to 10‑30+ FLOP/Byte. |
| **Vectorized loads/stores** | Use `float2`, `float4` (or `half2`, `half4`) via `__half2`, `__half4` or built‑in vector types (`half2`, `half4`). Ensure address alignment. | Improves memory coalescence, reduces instruction count. |
| **Optimal block/grid dimensions** | - Block size = 256 or 512 threads (multiple of warp size 32). <br>- Grid size: launch enough blocks to fully occupy all 96 CUs (target ≥ 4 blocks per CU). Use `hipOccupancyMaxPotentialBlockSize` to suggest block size. | Improves occupancy, hides latency. |
| **Register pressure reduction** | - Mark pointers `__restrict__`. <br>- Limit loop‑scoped variables, reuse registers. <br>- Use `-maxrregcount` compiler flag to tune. | Allows more warps per CU → higher occupancy. |
| **Loop unrolling** | Explicitly unroll inner loops (e.g., over BK) with `#pragma unroll`. | Reduces branch overhead, improves ILP. |
| **Use `__ldg` for read‑only data** | Annotate read‑only global loads with `__ldg(const ptr++)` to prefer L1 texture cache. | Caches read‑only matrices (e.g., Q, K, V, P). |
| **Constant memory for small read‑only tables** | If any kernel uses small lookup tables (e.g., softmax max), place in `__constant__`. | Faster access, reduces pressure on L1. |
| **Cooperative groups for reductions** | In softmax_bwd, use `cooperative_groups::reduce` for warp‑level then block‑level sum. | More efficient, less shared‑memory bank conflicts. |
| **HIP Graphs for launch sequence** | Encapsulate the five kernel launches (dp → softmax_bwd → dq → dk → dv) into a single hipGraph, instantiate and launch once per iteration. | Eliminates host‑side launch overhead, enables overlapping copies/compute via streams. |
| **Asynchronous memory copies** | Use `hipMemcpyAsync` with separate streams for H↔D transfers while GPU computes. | Overlaps PCIe transfer with kernel execution. |
| **Precise timing with hipEvent** | Wrap each kernel (or graph) with `hipEventRecord` before/after, compute elapsed time. | Accurate performance measurement for iteration. |
| **ISA inspection** | After each compile, run `amdllpc -mcpu=gfx1100 -dis <object>.hsaco` to review generated ISA, look for VALU/MFMA stalls, unnecessary moves, spillage. | Guides further register/alloc tweaks. |
| **Compiler flags** | Use `-O3 -ffast-math -fno-finite-math-only` (if safe), `-march=amdgcn -mcpu=gfx1100`. Enable `-DROCWMMA_ENABLE_BF16=ON` for BF16 WMMA path. | Maximizes instruction scheduling and vectorization. |

### 4.2 Kernel‑Specific Plans

#### dp_kernel (dP = dO @ Vᵀ)
- **Current:** naïve element‑wise, each thread computes one output dot product over K.  
- **Optimization:**
  1. **Tile size:** Choose `TM × TN` (e.g., 64×64) matching shared‑memory capacity.  
  2 Load a tile of dO `[TM × TK]` and V `[TN × TK]` into shared memory (using `__half2` loads).  
  3 Compute partial dot products inside the tile, accumulate to registers.  
  4 Loop over K in chunks of TK.  
  5 Use vectorized loads (`half2` or `half4`) for coalesced reads.  
  6 Apply `__ldg` for dO and V pointers.  
  7 After computing tile, store to dP with coalesced writes.  
- **Shared memory requirement:** `sizeof(half)*(TM*TK + TN*TK)`. With TM=TN=64, TK=32 → 2*64*32*2 = 8 KB (fits comfortably).  
- **Expected OI:** roughly `(2*TK) / (2*TK*sizeof + 2*TK*sizeof) ≈ 0.5 * (TK/2)` → with TK=32 gives ~8 FLOP/Byte (improves 16×).

#### dq_kernel (dQ = dS @ K) and dk_kernel (dK = dSᵀ @ Q)
- **Current:** naïve element‑wise with tiling in shared memory but not leveraging WMMA.  
- **Optimization Options:**
  - **Option A – WMMA‑based GEMM:** Replace innermost loops with rocWMMA matrix multiply fragments (16×16×16) using `bfloat16` (or `half`) fragments. This directly uses the hardware matrix pipeline.  
    - Declare `wmma::fragment<...>` for A, B, Accumulator.  
    - Load fragments from global memory using `wmma::load_matrix_sync` with appropriate strides (coalesced).  
    - Perform `wmma::mma_sync`.  
    - Store accumulator with `wmma::store_matrix_sync`.  
    - Tiles: WMMA native 16×16 (M,N) with inner dimension chunk BK=16/32 as supported.  
  - **Option B – Optimized shared‑memory GEMM** (if WMMA not desired):  
    - Tile dS (or dSᵀ) and K (or Q) into shared memory blocks (e.g., 64×64).  
    - Compute block product using inner product loops, accumulating in registers.  
    - Use vectorized loads and `__ldg`.  
- **Shared memory:** similar size to dp_kernel.  
- **Expected OI:** 20‑50 FLOP/Byte (depending on tile dimensions), moving towards compute‑bound.

#### dv_kernel (dV = Pᵀ @ dO)
- **Current:** naïve element‑wise with strided access to P (by N) and dO (by BK).  
- **Optimization:**
  1. **Transpose one operand** (if feasible) or load tiles of P and dO into shared memory to make accesses coalesced.  
  2. Use blocked tiling: load a tile of P `[TM × TN]` and dO `[TN × TK]` into shared memory, compute outer product, accumulate to dV tile `[TM × TK]`.  
  3. Apply vectorized loads (`half2/half4`).  
  4. Use `__ldg` for read‑only P and dO.  
- **Shared memory:** similar to dp_kernel (two tiles).  
- **Expected OI:** raised from ~0.5 to ~10‑20 FLOP/Byte.

#### softmax_bwd_kernel (dS = P ⊙ (dP − rowsum(dP×P)))
- **Current:** two passes over rows with a reduction in shared memory.  
- **Optimization:**
  1. **First pass (rowsum):** Use warp‑level reductions (`cooperative_groups::reduce_sum`) to compute partial sums, then a block‑level reduction in shared memory (avoiding bank conflicts via padding).  
  2. **Second pass (compute dS):** After storing rowsums, each warp loads its assigned row’s P and dP values via vectorized loads, computes `P * (dP - rowsum)`, writes dS.  
  3. Enable `__ldg` for P and dP (read‑only).  
  4. Ensure block size is a multiple of warp size (e.g., 256 threads → 8 warps) to fully utilize warp schedulers.  
- **Shared memory:** only need storage for rowsums (`BM` floats) plus reduction workspace (e.g., `blockDim.x` floats).  
- **Expected impact:** Reduction latency cut, better memory throughput → overall kernel time reduction ~20‑30%.

### 4.3 Fusion & Launch Overhead Reduction

- Already present: `fmha_bwd_full_py` launches five kernels sequentially with intermediate buffers.  
- **Enhancements:**
  - Allocate dP and dS **once** per call (already done).  
  - Use **hipStream_t** for each kernel to enable overlap (if dependencies allow). However true dependencies exist: softmax_bwd needs dP, dq needs dS, etc. So we can overlap independent parts: e.g., while dq kernel computes on a subset of rows, softmax_bwd can work on another stream if we split the work along M dimension (requires splitting dP/dS). This is more advanced; for now, keep sequential but use **hipGraphs** to capture the sequence and eliminate per‑launch host overhead.  
  - Record the graph once (after first successful run) and replay for each iteration; only update kernel arguments if they change (they don’t in benchmark).  
  - Use **hipEvent** to measure graph execution time.

### 4.4 Validation & Correctness

After each modification:
1. Run isolated unit tests (`test_*_isolated.py`) – must pass with tolerance < 1e‑2 (BF16) or < 1e‑3 (FP16).  
2. Run integration test (`test_integration.py`).  
3. Compare outputs against PyTorch reference using the benchmark’s diff reporting (max absolute difference).  
4. If any test fails, inspect the offending kernel with `rocgdb` or add debug prints (`printf`) guarded by a compile‑time flag.

## 5. Correctness Assurance (Best‑Practice Checklist)

To guarantee that optimizations do **not** silently break numerical correctness, follow this incremental verification workflow after **every** code change:

1. **Preserve a reference implementation**  
   - Keep the original naïve kernels (e.g., in `src/fmha_bwd_kernels_naive.hip`).  
   - After each change, compile both the naïve and the optimized versions.

2. **Use a high‑precision oracle**  
   - Compute the expected result in FP32 (or FP64) on CPU or via a separate HIP kernel that accumulates in FP32.  
   - Compare the optimized BF16/FP16 output **against this FP32 reference**, allowing only a small error (e.g., ≤ 1 ULP for BF16 ≈ 0.001, ≤ 2 ULP for FP16 ≈ 0.0005).  
   - This catches loss of precision from reduced‑precision accumulators.

3. **Automated unit‑ and integration tests**  
   - Existing `test_*_isolated.py` and `test_integration.py` already compare against PyTorch (FP32→BF16).  
   - Keep the tolerances tight (`rtol=1e-3, atol=1e-3` for BF16) and treat any failure as a blocker.

4. **Intermediate‑buffer diff**  
   - Save the intermediate tensors (`dP`, `dS`, `dQ`, `dK`, `dV`) from both naïve and optimized kernels to host memory (or to files).  
   - Perform element‑wise `allclose` checks.  
   - If an intermediate buffer differs, the error is localized to that kernel; otherwise the problem lies later in the pipeline.

5. **Deterministic seeding & fixed problem sizes**  
   - In all test scripts fix `np.random.seed(42)` and `torch.manual_seed(42)`.  
   - Run the naïve and optimized kernels back‑to‑back on the **exact same** input tensors to eliminate scheduler nondeterminism.

6. **Optional intra‑kernel debug prints**  
   - Guard `printf` statements with `#if DEBUG` to log a few values (e.g., first warp’s accumulator) from both kernels; compare logs visually or via a script.

7. **Compiler flag hygiene**  
   - Avoid `-ffast-math` (or similar) on stages where the order of reductions matters.  
   - If you need fast math for performance, build two variants: a “strict” version (no fast‑math) for verification and a “fast” version for benchmarking; only promote the fast version if the strict‑vs‑fast diff stays within the tolerated ULP range.

8. **Leverage AMD debugging tools**  
   - **rocgdb**: set breakpoints at kernel entry/exit, inspect register values or shared memory in a few warps to verify no race conditions.  
   - **amdllpc -dis**: review the generated ISA for unintended spills or extra moves that could alter precision.  
   - **Kernel Analyzer (via rocprofiler --kernel-stat)**: look for abnormal numbers of `s_waitcnt` (indicates unnecessary stalls from missing `__syncthreads()`).  
   - **rocprofiler --memory-traces**: confirm that global loads are truly coalesced (addresses increasing by 128 bytes per wavefront).  
   - **omnitrace**: verify that host‑device copies do not unintentionally serialize kernel execution.

9. **Commit granularity & bisect readiness**  
   - Each logical optimization (e.g., “add tiled shared memory to dp_kernel”, “switch dq_kernel to WMMA”) gets its own commit with a clear message.  
   - If a later commit breaks correctness, use `git bisect` to pinpoint the offending change, fix only that commit, and continue.

10. **CI regression gate**  
    - Add a step to your CI workflow that after building runs: unit tests, integration test, and a fixed‑size correctness comparison (naïve vs. optimized).  
    - Any failure blocks merging to `main`.

By adhering to this checklist, you guarantee that every performance improvement is **verified** numerically before moving on, eliminating the risk of silently degraded accuracy.

---

## 5. Iterative Workflow

```
repeat for each kernel (dp, softmax_bwd, dq, dk, dv):
    1. Apply a set of optimizations (start with tiling + vectorized loads).
    2. Re‑compile (release mode: -O3, -ffast-math).
    3. Run correctness tests.
    4. If pass:
         a. Run benchmark to get new timing.
         b. Profile with rocprofiler/omnitrace to capture:
            - Achieved occupancy (%)
            - Achieved memory bandwidth (%)
            - Achieved compute (VALU/MFMA) %
            - ELapsed time per kernel
         c. Compare to roofline; compute OI achieved.
         c. If still far from compute bound, add next optimization (e.g., WMMA, shared‑memory GEMM, better block sizing).
    5. If fail: debug, fix, then retest.
    6. Log results in a table (see section 6).
```

At the end, we should have a set of optimized kernels whose **combined backward pass time** is within 10‑20 % of PyTorch (or better) while consuming less global memory and having lower launch overhead.

---

## 6. Results Log (to be filled during optimization)

| Kernel | Version | Block size (Tx,Ty,Tz) | Shared mem (KB) | Achieved Occupancy | Achieved BW (GB/s) | Achieved FLOPS (TFLOP/s) | Speedup vs. PyTorch | Notes |
|--------|---------|-----------------------|-----------------|--------------------|--------------------|--------------------------|----------------------|-------|
| dp_kernel | naïve | – | – | – | – | – | – | Baseline |
| dp_kernel | tiled + vec | 16×16×1 | 8 | … | … | … | … | |
| … | … | … | … | … | … | … | … | |
| **Total backward** | naïve | – | – | – | – | – | 0.12× | Baseline |
| **Total backward** | optimized | – | – | … | … | … | 0.85× (goal) | After all optimizations |

*(Update after each iteration.)*

---

## 7. Final Deliverables

1. **Source code** with all optimizations applied (both BF16 and FP16 variants).  
2. **Updated CMakeLists.txt** (if any new compile flags or sources added).  
3. **Performance report** (`PERFORMANCE_RESULTS.md`) containing:  
   - Baseline vs. optimized numbers.  
   - Roofline plots (can be generated with `rocprofiler` data).  
   - Description of remaining bottlenecks (if any).  
4. **README update** with instructions to build/run the optimized version and expected speedup.  
5. **Commit** all changes to the repository.

---

## 8. Timeline (suggested)

| Day | Activity |
|-----|----------|
| 1 | Environment setup, baseline capture, roofline analysis. |
| 2‑3 | Optimize `dp_kernel` (tiling + vectorization). Validate, measure. |
| 4‑5 | Optimize `dq_kernel` / `dk_kernel` (WMMA or shared‑memory GEMM). Validate, measure. |
| 6 | Optimize `dv_kernel` (tiled shared memory). Validate, measure. |
| 7 | Optimize `softmax_bwd_kernel` (warp reductions + vector loads). Validate, measure. |
| 8 | Implement hipGraph launch sequence, overlap streams if possible. |
| 9‑10| Full integration testing, benchmark vs PyTorch, finalize documentation. |
| 11| Write results, commit, cleanup. |

---

## 9. Conclusion

By systematically applying the AMD‑provided profiling and optimization tools (rocprofiler, omnitrace, Kernel Analyzer, llvm-mca, amdllpc, rocgdb, hipGraphs, etc.) and following the transformations outlined above, we can transform the naïve FMHA backward kernels from a **memory‑bound, low‑occupancy** implementation into a **high‑throughput, fused** kernel that approaches the hardware’s roofline limit. The expected outcome is a **significant speedup** (target ≥ 0.8× PyTorch, with potential to match or exceed) together with **reduced memory footprint** and **lower launch overhead**, making the implementation suitable for real‑world LoRA/DoRA training on the Radeon RX 7900 XTX.

*Let’s begin!*
 Work Plan for FMHA Backward Kernel Optimization

 Head-dimension scalability with MAX_HEADS_PER_LAUNCH

  Based on the code inspection and the current state of the files, here is the
  summary of what has been accomplished for Phase 3:

  ✅ Completed:
  1. Added MAX_HEADS_PER_LAUNCH constant to
  src/kernel/correct/fmha_bwd_config.h:
  // Maximum number of heads to process per kernel launch (for head tiling)
  // Set empirically based on register pressure and occupancy requirements
  constexpr int MAX_HEADS_PER_LAUNCH = 64;

  🔧 Next Steps (to fully implement Phase 3):
  To complete Phase 3 according to the WORK_PLAN.md, the three WMMA kernels
  (dq_kernel, dk_kernel, dv_kernel) in src/fmha_bwd_kernels.hip need to be
  modified to:
  - Accept a head_start parameter (indicating the first head index to process
  for this block)
  - Internally loop over head_idx = head_start + head_offset for head_offset in
  [0, MAX_HEADS_PER_LAUNCH)
  - Process up to MAX_HEADS_PER_LAUNCH heads per block (instead of exactly 1
  head per block)
  - Keep the same function signatures so existing test wrappers continue to work
  - The outer loop over heads will reside in the test/benchmark scripts (or a
  thin wrapper), as specified in the plan

  📋 Current State of Kernels:
  Upon inspection, the kernels in src/fmha_bwd_kernels.hip are still the naive
  versions, not the WMMA versions from src/kernel/correct/. Therefore, the
  immediate next step is to:
  1. Replace the naive kernels with their WMMA-optimized counterparts (from
  src/kernel/correct/)
  2. Then modify each WMMA kernel to support head tiling via the head_start
  parameter and internal loop over MAX_HEADS_PER_LAUNCH

  📋 Current State of Kernels:
  Upon inspection, the kernels in src/fmha_bwd_kernels.hip are still the naive
  versions, not the WMMA versions from src/kernel/correct/. Therefore, the
  immediate next step is to:
  1. Replace the naive kernels with their WMMA-optimized counterparts (from
  src/kernel/correct/)
  2. Then modify each WMMA kernel to support head tiling via the head_start
  parameter and internal loop over MAX_HEADS_PER_LAUNCH

  🛠️  Recommended Action Plan:
  1. Replace dq_kernel, dk_kernel, dv_kernel in src/fmha_bwd_kernels.hip with
  the WMMA versions from:
    - src/kernel/correct/fmha_bwd_dq_kernel.hpp
    - src/kernel/correct/fmha_bwd_dk_kernel.hpp
    - src/kernel/correct/fmha_bwd_dv_kernel.hpp
  2. Modify each inserted WMMA kernel to:
    - Add int head_start parameter to the function signature
    - Compute head_idx_base = blockIdx.z * MAX_HEADS_PER_LAUNCH
    - Loop over head_offset in [0, MAX_HEADS_PER_LAUNCH), breaking if head_idx
  >= total_heads
    - Inside the loop, compute base pointers:
  base_ptr = original_ptr + head_idx * stride
    - Keep all shared memory and WMMA logic inside the head loop (so
  accumulation is per-head)
    - Write results immediately after computing each head (to avoid accumulation
  across heads)
  3. Update fmha_bwd_full_py (the extern "C" wrapper) to:
    - Pass head_start = 0 to each kernel launch (since the host now handles
  tiling via multiple block launches in the z-dimension)
    - Adjust the grid dimension in z to be:
  grid.z = (total_heads + MAX_HEADS_PER_LAUNCH - 1) / MAX_HEADS_PER_LAUNCH
    - Remove the total_heads argument from the kernel launches (since it's now
  handled by the head loop inside the kernel)

  📌 Example Modification (for dq_kernel):
  global void dq_kernel(const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
                            int M_global, int N_global, int K_dim_global, int
  total_heads,
                            int head_start)  // <-- Added head_start
  {
      int head_idx = blockIdx.z * MAX_HEADS_PER_LAUNCH + head_start;  // <--
  Head tiling
      if (head_idx >= total_heads) return;  // <-- Bounds check

  const bhalf_t* dS_base = dS + head_idx * M_global * N_global;
  const bhalf_t* K_base  = K  + head_idx * N_global * K_dim_global;
  bhalf_t*       dQ_base = dQ + head_idx * M_global * K_dim_global;

  // ... [shared memory and WMMA logic remains largely unchanged] ...
  // ... but all memory accesses use the *_base pointers above ...
  }

  ✅ Benefits of This Approach:
  - Backward compatibility: Existing test wrappers (e.g., test_dq_kernel) can
  still call the kernel with head_start = 0
  - Head tiling: Each block processes up to MAX_HEADS_PER_LAUNCH heads, reducing
  grid pressure and improving occupancy
  - Flexibility: The host (in fmha_bwd_full_py or test scripts) can launch
  multiple blocks in the z-dimension to process all heads
  - Performance: Better utilization of GPU resources when total_heads is large

  📝 Notes:
  - The dp_kernel and softmax_bwd_kernel are already naive and handle head_idx =
  blockIdx.z; they may also benefit from similar head tiling in a later
  optimization pass.
  - After implementing head tiling in the WMMA kernels, the grid dimensions in
  fmha_bwd_full_py should be updated to reflect the new z-dimension layout.

  🚀 Immediate Next Step:
  Replace the three naive kernels in src/fmha_bwd_kernels.hip with their WMMA
  counterparts from src/kernel/correct/, then modify each to include head tiling
  logic as described above.


  Phase 4: Optimise softmax_bwd_kernel

  - [ ] Replace the two‑pass reduction with a warp‑level + block‑level reduce
  (using cooperative_groups::reduce or manual shuffling).
  - [ ] Add vectorised loads (float2/float4) and __ldg for read‑only P and dP.
  - [ ] Verify correctness with test_softmax_isolated.py.
  - [ ] Benchmark and record improvement.

  Phase 5: HIP‑graph launch sequence (reduce launch overhead)

  - [ ] After all five kernels are correct and benchmarked,
  replace the sequential hipLaunchKernelGGL calls in
  fmha_bwd_full_py with a single HIP graph:
    - hipGraphCreate
    - hipGraphAddKernelNode for each kernel (in order)
    - hipGraphInstantiate
    - hipGraphLaunch
  - [ ] Verify correctness and benchmark again.
  - [ ] Expect additional speed‑up (especially for small problem sizes)
  due to eliminated launch overhead.

  Phase 6: Final Validation and Documentation

  - [ ] Run full benchmark suite across multiple problem sizes
  (as in benchmark_pytorch_final.py) and across a range of
  head_dim values (16, 32, 64, 128, 256).
  - [ ] Update PERFORMANCE_RESULTS.md with tables and, if possible,
  roofline plots (generated from rocprofiler data).
  - [ ] Update README.md with build/run instructions and performance
  numbers.
  - [ ] Ensure all tests pass and no regressions.

  Optional: Continuous Integration

  - Add a CI step (GitHub Actions) that builds, runs tests, and benchmarks
  on a ROCm agent.

  Notes

  - Keep all changes in src/fmha_bwd_kernels.hip unless integrating WMMA
  headers from kernel/correct/.
  - Preserve numeric correctness: compare against FP32 reference and allow
  ≤ 0.5 ULP (≈ 1e‑2 for bfloat16) error.
  - Commit after each major milestone with descriptive messages.

  ---

  **What to do next**

  1. Replace the three naïve kernels (`dq_kernel`, `dk_kernel`, `dv_kernel`) in
  `src/fmha_bwd_kernels.hip` with the WMMA versions provided in the previous
  answer (including the `head_idx` offset).
  2. Re‑build the library, copy it to `tests/build/`, and run the full test
  suite.
  3. If all tests pass, run `benchmark_pytorch_final.py` to obtain the updated
  performance numbers.
  4. After that, proceed with the head‑dimension scalability steps (introducing
  `MAX_HEADS_PER_LAUNCH`), then optimise `softmax_bwd` and finally add a
  HIP‑graph.


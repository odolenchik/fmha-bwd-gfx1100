# Performance Analysis of FMHA Backward Kernels

## Executive Summary

Based on benchmark results showing 0.09-0.15x speedup vs PyTorch (instead of expected >1x), the custom HIP kernels are significantly underperforming. This document analyzes potential performance bottlenecks and optimization opportunities that can be investigated without requiring GPU execution.

## Kernel-by-Kernel Analysis

### 1. Softmax Backward Kernel (`softmax_bwd_kernel`)

**Current Implementation:**
- 2D grid: `(M + BM - 1) / BM` x `total_heads`
- Each block processes one row (M dimension) for one head
- Uses shared memory for row-wise reduction
- Strided access pattern for column processing

**Potential Issues:**
- **Underutilization**: When M is small, grid size in x-dimension is small, leading to underutilization of GPU resources
- **Memory Access**: Strided access (`n += blockDim.x`) may not be optimal for memory coalescing
- **Shared Memory Usage**: Uses `BLOCK_SIZE * sizeof(float)` for partial sums, but could be optimized

**Optimization Opportunities:**
- Consider 1D grid over flattened (M * heads) dimension for better scalability
- Vectorized memory accesses (float2/float4) if applicable
- Optimize reduction pattern for better parallel efficiency

### 2. dQ Kernel (`dq_kernel`)

**Current Implementation:**
- Tiled approach with shared memory
- 3D grid: `(K_dim + BK - 1) / BK` x `(M + BM - 1) / BM` x `total_heads`
- Uses WMMA-inspired tiling (though current code appears to be naive element-wise, not WMMA)
- Each thread block computes one BM×BK tile

**Note**: The current implementation in the file appears to be a naive tiled approach, not the WMMA-optimized version mentioned in comments.

**Potential Issues:**
- **Shared Memory Bank Conflicts**: Access patterns to `s_dS` and `s_K` may cause bank conflicts
- **Inefficient Tile Loading**: Multiple __syncthreads() boundaries may create pipeline bubbles
- **Register Pressure**: High register usage from nested loops may limit occupancy

**Optimization Opportunities:**
- Verify actual WMMA implementation vs current naive tiling
- Optimize shared memory layout to avoid bank conflicts
- Consider loop unrolling and better instruction scheduling
- Optimize thread block shape for better memory coalescing

### 3. dK Kernel (`dk_kernel`)

Similar structure to dq_kernel with analogous optimization opportunities.

### 4. dV Kernel (`dv_kernel`)

**Current Implementation:**
- 2D grid tiling over N and BK_dim dimensions
- Each thread computes output element by reduction over M dimension
- No shared memory usage

**Potential Issues:**
- **Poor Memory Access Pattern for P**: Access `P_h[m * N_global + block_n + n_local]` strided by N_global
- **Poor Memory Access Pattern for dO**: Access `dO_h[m * BK_dim + block_k + k_local]` strided by BK_dim
- **Low Arithmetic Intensity**: Each thread does M global memory reads for one output element

**Optimization Opportunities:**
- **Add Shared Memory Tilinig**: Cache tiles of P and dO in shared memory to reduce global memory bandwidth
- **Reorganize Computation**: Change loop order or tiling strategy to improve data reuse
- **Vectorization**: Consider vectorized loads if access patterns allow

### 5. dP Kernel (`dp_kernel`)

**Current Implementation:**
- Similar to dV kernel but for dP = dO @ V^T
- 2D grid tiling
- No shared memory

**Similar Issues and Opportunities as dV kernel.**

## Cross-Kernel Observations

### 1. Grid Configuration Efficiency
All kernels use similar grid calculation patterns. Potential issue:
- For small problem sizes, grid dimensions may be very small (e.g., 1x1xheads), leading to severe underutilization
- Consider minimum grid size guarantees or persistent kernel approaches

### 2. Shared Memory Allocation
- `softmax_bwd_kernel` uses `(BLOCK_SIZE + BM) * sizeof(float)` - verify this size is correct
- Tiled kernels use fixed shared memory sizes - verify against actual usage

### 3. Occupancy Calculation
Theoretical analysis needed:
- Register usage per thread (from compilation)
- Shared memory usage per block
- Theoretical occupancy based on GPU resources (RDNA3: RX 7900 XTX)

### 4. Memory Bound vs Compute Bound Analysis
Given the poor performance relative to PyTorch, these kernels are likely memory bound. Focus should be on:
- Improving memory coalescing
- Increasing data reuse through tiling/shared memory
- Reducing global memory accesses

## Recommended Next Steps (CPU-Guest Only)

1. **Assembly Inspection**: Use `amdllpc` to inspect generated ISA for instruction efficiency
2. **Occupancy Calculator**: Manual calculation of theoretical occupancy
3. **Memory Access Tracing**: Static analysis of access patterns for coalescing
4. **Benchmark Alternatives**: Create micro-benchmarks that test memory bandwidth/compute independently
5. **Rofline Model Analysis**: Position kernels on memory-compute rofline

## Specific Code Improvement Suggestions

### For softmax_bwd_kernel:
```cpp
// Consider flattening grid for better scaling
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int total_elems = M * total_heads;
if (tid >= total_elems) return;

int m = tid / total_heads;
int head_idx = tid % total_heads;
// ... rest similar but with better scaling
```

### For dV/dP kernels:
Add shared memory tiling:
```cpp
__shared__ float s_P_tile[BM][BN];  // or appropriate dimensions
__shared__ float s_dO_tile[BM][BK];
// Load tiles cooperatively, then compute
```

## Conclusion

The primary performance limitations likely stem from:
1. Suboptimal grid configurations leading to underutilization
2. Memory access patterns that don't coalesce well
3. Lack of data reuse opportunities (no/shared insufficient shared memory tiling)
4. Potential instruction inefficiencies in the compiled code

These issues can be analyzed and improved through static code analysis and compiler inspection without requiring GPU benchmarks for initial improvements.
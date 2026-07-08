import sys
with open('src/fmha_bwd_kernels.hip', 'r') as f:
    lines = f.readlines()

# Find line with hipFree(dS);
for i, line in enumerate(lines):
    if line.strip() == 'hipFree(dS);':
        hipfree_idx = i
        break
else:
    sys.exit("hipFree(dS); not found")

# The next line should be the closing brace of extern "C"
# Insert our test function definitions before that line.
# We'll insert after hipfree_idx+1? Actually we want to put them before the '}' line.
# So we insert at hipfree_idx+1 (the line after hipFree) but before the '}' line.
# Let's see what line hipfree_idx+1 is.
if lines[hipfree_idx+1].strip() != '}':
    # If not, we still insert before the '}' line later.
    pass

# Find the index of the line that is exactly '}' (with possible spaces) after hipfree_idx.
    for j in range(hipfree_idx+1, len(lines)):
        if lines[j].strip() == '}':
            close_brace_idx = j
            break
    else:
        close_brace_idx = hipfree_idx+1  # fallback
else:
    close_brace_idx = hipfree_idx+1

# Define the test function blocks to insert
test_defs = '''// Test kernel for dp_kernel (tiled version)
void test_dp_kernel(const bhalf_t* dO, const bhalf_t* V, bhalf_t* dP,
                    int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, total_heads);
    size_t shared_mem = (BM * BK + BN * BK) * sizeof(bhalf_t);
    hipLaunchKernelGGL(dp_kernel, dim3(grid), dim3(BLOCK_SIZE), shared_mem, 0, dO, V, dP, M, N, K_dim, total_heads);
}

// Test kernel for softmax_bwd_kernel
void test_softmax_bwd_kernel(const bhalf_t* P, const bhalf_t* dP, bhalf_t* dS,
                             int M, int N, int total_heads)
{
    const int blockSize = 256;                     // must be a multiple of 32
    dim3 grid(M, total_heads, 1);                  // one block per (row, head)
    dim3 block(blockSize);
    size_t shared_mem = (blockSize / 32) * sizeof(float); // warps-per-block * sizeof(float)
    hipLaunchKernelGGL(softmax_bwd_kernel, dim3(grid), dim3(block), shared_mem, 0,
                       P, dP, dS, M, N, total_heads);
}
void dv_kernel_sym(const bhalf_t* P, const bhalf_t* dO, bhalf_t* dV,
                   int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    hipLaunchKernelGGL(dv_kernel, dim3(grid), dim3(BLOCK_SIZE), 0, 0, P, dO, dV, M, N, K_dim, total_heads);
}
'''

# Insert the lines before the closing brace line.
# We'll split the lines list into three parts: before insert point, the insert lines, after.
before = lines[:close_brace_idx]
after = lines[close_brace_idx:]  # includes the '}' line and everything after
# We want to keep the '}' line after the inserted definitions.
new_lines = before + [test_defs] + after
# Flatten list of lists
new_lines_flat = []
for item in new_lines:
    if isinstance(item, list):
        new_lines_flat.append(item)
    else:
        new_lines_flat.append(item)
# Actually test_defs is a string; we need to split into lines.
test_lines = test_defs.splitlines(keepends=True)
new_lines_flat = before + test_lines + after

with open('src/fmha_bwd_kernels.hip', 'w') as f:
    f.writelines(new_lines_flat)

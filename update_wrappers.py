import re
with open('src/fmha_bwd_kernels.hip', 'r') as f:
    content = f.read()

# Replace test_dq_kernel wrapper
new_test_dq = '''// Test kernel for dq_kernel (WMMA version)
void test_dq_kernel(const bhalf_t* dS, const bhalf_t* K, bhalf_t* dQ,
                    int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, total_heads);
    size_t shared_mem = (BM + BK) * (BN + LDS_PAD) * sizeof(bhalf_t);
    hipLaunchKernelGGL(dq_kernel, dim3(grid), dim3(BLOCK_SIZE), shared_mem, 0, dS, K, dQ, M, N, K_dim, total_heads);
}'''
# Replace from '// Test kernel for dq_kernel' to the end of that function (next newline before next function or end)
pattern = r'(// Test kernel for dq_kernel\(.*?\)\s*\{.*?\}\s*)'
# Use DOTALL to match across lines
content = re.sub(pattern, new_test_dq + '\n\n', content, flags=re.DOTALL)

# Replace test_dk_kernel wrapper
new_test_dk = '''// Test kernel for dk_kernel (WMMA version)
void test_dk_kernel(const bhalf_t* dS, const bhalf_t* Q, bhalf_t* dK,
                    int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    size_t shared_mem = (BN + BM) * (BM + LDS_PAD) * sizeof(bhalf_t); // s_dS: BN*(BM+LDS_PAD), s_Q: (BK+LDS_PAD)*BM
    hipLaunchKernelGGL(dk_kernel, dim3(grid), dim3(BLOCK_SIZE), shared_mem, 0, dS, Q, dK, M, N, K_dim, total_heads);
}'''
pattern = r'(// Test kernel for dk_kernel\(.*?\)\s*\{.*?\}\s*)'
content = re.sub(pattern, new_test_dk + '\n\n', content, flags=re.DOTALL)

# Replace test_dv_kernel_sym wrapper
new_test_dv = '''// Test kernel for dv_kernel (WMMA version)
void dv_kernel_sym(const bhalf_t* P, const bhalf_t* dO, bhalf_t* dV,
                   int M, int N, int K_dim, int total_heads) {
    dim3 grid((N + BN - 1) / BN, (K_dim + BK - 1) / BK, total_heads);
    size_t shared_mem = 0; // WMMA dv_kernel uses no shared memory
    hipLaunchKernelGGL(dv_kernel, dim3(grid), dim3(BLOCK_SIZE), 0, 0, P, dO, dV, M, N, K_dim, total_heads);
}'''
pattern = r'(// Test kernel for dv_kernel\(.*?\)\s*\{.*?\}\s*)'
content = re.sub(pattern, new_test_dv + '\n\n', content, flags=re.DOTALL)

with open('src/fmha_bwd_kernels.hip', 'w') as f:
    f.write(content)

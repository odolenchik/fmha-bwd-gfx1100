"""Debug why matmul returns zeros for float16 inputs."""
import torch, triton, triton.language as tl

TILE_N = tl.constexpr(64)
BK = tl.constexpr(32)

@triton.jit
def debug_matmul_kernel(A_ptr, B_ptr, D_ptr, M: tl.constexpr, N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_offsets = pid_m * TILE_N + tl.arange(0, TILE_N)
    k_offsets = tl.arange(0, BK)

    m_mask = m_offsets < M
    k_mask = k_offsets < 32

    acc_dtype = tl.float32
    acc = tl.zeros((TILE_N, BK), dtype=acc_dtype)

    for n_start in range(0, N, TILE_N):
        n_offsets = n_start + tl.arange(0, TILE_N)
        n_mask = n_offsets < N

        a_tile = tl.load(A_ptr + m_offsets[:, None] * N + n_offsets[None, :],
                         mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        b_tile = tl.load(B_ptr + n_offsets[:, None] * 32 + k_offsets[None, :],
                         mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        acc += tl.dot(a_tile.to(acc_dtype), b_tile.to(acc_dtype))

    result = acc.to(tl.float16)
    tl.store(D_ptr + m_offsets[:, None] * 32 + k_offsets[None, :],
             result, mask=m_mask[:, None] & k_mask[None, :])


M_test = 4
N_test = 4

A_fp16 = torch.randn(M_test, N_test, dtype=torch.float16, device='cuda')
B_fp32 = torch.randn(N_test, 32, dtype=torch.float32, device='cuda')
D_fp16 = torch.zeros((M_test, 32), dtype=torch.float16, device='cuda')

print(f'A_fp16[0] = {A_fp16[0]}')
print(f'B_fp32[:, :4][0] = {B_fp32[0,:4]}')

debug_matmul_kernel[(1, 1)](A_fp16, B_fp32, D_fp16, M_test, N_test)
print(f'D_triton[0, :4] = {D_fp16[0, :4]}')

# Expected: A @ B[:M,:32] — but since A is [4,4], we only use first 4 cols of B
expected = torch.mm(A_fp16.to(torch.float32), B_fp32.to(torch.float32))
print(f'Expected[0, :4] = {expected[0, :4]}')

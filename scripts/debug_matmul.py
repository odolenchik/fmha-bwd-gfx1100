"""Debug Triton matmul kernel to find why it returns zeros."""
import torch, triton, triton.language as tl

TILE_N = tl.constexpr(64)
BK = tl.constexpr(32)

@triton.jit
def debug_matmul_kernel(A_ptr, B_ptr, D_ptr, M: tl.constexpr, N: tl.constexpr):
    """Debug version that prints intermediate values."""
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

A = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [-1.0, -2.0, -3.0, -4.0],
    [0.5, 1.5, -0.5, 2.5]
], dtype=torch.float32, device='cuda')

B = (torch.eye(4, dtype=torch.float32, device='cuda') * 10.0 + torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda').unsqueeze(1) / 10.0).t()

D = torch.zeros((M_test, 32), dtype=torch.float16, device='cuda')

grid_m = (M_test + TILE_N - 1) // TILE_N
print(f'Grid: ({grid_m}, 1)')
debug_matmul_kernel[(grid_m, 1)](A, B, D, M_test, N_test)

print('A:\\n', A)
print('B:\\n', B)
print('D_triton[:, :4]:\\n', D[:, :4])
expected = torch.mm(A.to(torch.float32), torch.cat([B, torch.zeros(4, 28, dtype=torch.float16, device='cuda')], dim=1))
print('Expected first row dot product:', A[0] @ B[:, 0])

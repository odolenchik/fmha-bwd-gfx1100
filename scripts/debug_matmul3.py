"""Minimal Triton matmul on RDNA3 to verify tl.dot correctness."""
import torch, triton, triton.language as tl


@triton.jit
def tiny_matmul(A_ptr, B_ptr, D_ptr):
    pid_m = tl.program_id(0)  # only tile: pid=0
    pid_k = tl.program_id(1)

    m_offsets = tl.arange(0, 64)
    k_offsets = tl.arange(0, 32)

    m_mask = m_offsets < M_test
    k_mask = k_offsets < Kdim_padded

    acc_dtype = tl.float32
    acc = tl.zeros((64, 32), dtype=acc_dtype)

    for n_start in range(0, N_test, 64):
        n_offsets = n_start + tl.arange(0, 64)
        n_mask = n_offsets < N_test

        a_tile = tl.load(A_ptr + m_offsets[:, None] * N_test + n_offsets[None, :],
                         mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        b_tile = tl.load(B_ptr + n_offsets[:, None] * Kdim_padded + k_offsets[None, :],
                         mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        acc += tl.dot(a_tile.to(acc_dtype), b_tile.to(acc_dtype))

    result = acc.to(tl.float16)
    tl.store(D_ptr + m_offsets[:, None] * Kdim_padded + k_offsets[None, :],
             result, mask=m_mask[:, None] & k_mask[None, :])


M_test = 64
N_test = 64
Kdim_padded = 32

A = torch.randn(M_test, N_test, dtype=torch.float16, device='cuda')
B = torch.randn(N_test, Kdim_padded, dtype=torch.float16, device='cuda')
D = torch.zeros((M_test, Kdim_padded), dtype=torch.float16, device='cuda')

tiny_matmul[(1, 1)](A, B, D)

expected = torch.mm(A.to(torch.float32), B.to(torch.float32))
diff = (D.float() - expected).abs().max().item()
print(f'Max diff: {diff:.4e}')
if diff > 0.1:
    print('A[0,:5]:', A[0, :5])
    print('B[:5,0]:', B[:5, 0])
    print('D_triton[0,:5]:', D[0, :5])
    print('Expected[0,:5]:', expected[0, :5])

"""Debug Triton load behavior."""
import torch, triton, triton.language as tl

TILE_N = tl.constexpr(64)

@triton.jit
def debug_load_kernel(A_ptr, D_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, TILE_N)
    mask = offsets < 4
    
    loaded = tl.load(A_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(D_ptr + pid * N + offsets, loaded.to(tl.float16), mask=mask)


A = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [-1.0, -2.0, -3.0, -4.0],
    [0.5, 1.5, -0.5, 2.5]
], dtype=torch.float32, device='cuda')

D = torch.zeros((4, 4), dtype=torch.float16, device='cuda')

debug_load_kernel[(4,)](A, D, 4)
print('A:\\n', A)
print('D_triton:\\n', D)
print('Match:', torch.allclose(A.half(), D))

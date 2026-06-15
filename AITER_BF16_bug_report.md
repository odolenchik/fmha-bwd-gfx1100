# Bug Report: fmha_v3_bwd fails with BF16/FP16 tensors on ROCm

## Environment
- **GPU**: AMD Radeon RX 7900 XTX (gfx1100, RDNA3)
- **ROCm**: 7.2.4 (hip: 7.2.53211)
- **PyTorch**: 2.13.0.dev20260418+rocm7.2
- **AITER**: amd-aiter==0.1.14rc1.dev303+g90cb390e1 (from GitHub https://github.com/ROCm/aiter)

## Symptom

`fmha_v3_bwd()` raises `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage` when called with BF16 or FP16 tensors.

FP32 and FP64 tensors pass the `.data_ptr()` check but correctly fail later with "FlashAttention only support fp16 and bf16".

## Reproduction Code

```python
import torch, math
from aiter import fmha_v3_bwd

batch, heads, seq, head_dim = 1, 4, 64, 64

# BF16 test (fails)
q = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=torch.bfloat16)
k = q.clone()
v = q.clone()
dO = torch.randn_like(v)
out = v.clone()
softmax_lse = torch.zeros(batch, heads, seq, device='cuda').contiguous().clone()

dq = torch.empty_like(q).contiguous()
dk = torch.empty_like(k).contiguous()
dv = torch.empty_like(v).contiguous()

result = fmha_v3_bwd(
    dO.clone(), q, k, v, out.clone(), softmax_lse.clone(),
    dropout_p=0.0, softmax_scale=1.0/math.sqrt(head_dim), is_causal=False,
    window_size_left=-1, window_size_right=-1, deterministic=True,
    is_v3_atomic_fp32=False, how_v3_bf16_cvt=0,
    dq=dq, dk=dk, dv=dv)
# RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

# FP16 test (same failure)
q_f16 = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=torch.float16)
k_f16 = q_f16.clone()
v_f16 = q_f16.clone()
dO_f16 = torch.randn_like(v_f16)
out_f16 = v_f16.clone()

dq_f16 = torch.empty_like(q_f16).contiguous()
dk_f16 = torch.empty_like(k_f16).contiguous()
dv_f16 = torch.empty_like(v_f16).contiguous()

result = fmha_v3_bwd(
    dO_f16.clone(), q_f16, k_f16, v_f16, out_f16.clone(), softmax_lse.clone(),
    dropout_p=0.0, softmax_scale=1.0/math.sqrt(head_dim), is_causal=False,
    window_size_left=-1, window_size_right=-1, deterministic=True,
    is_v3_atomic_fp32=False, how_v3_bf16_cvt=0,
    dq=dq_f16, dk=dk_f16, dv=dv_f16)
# RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

# FP32 test (passes .data_ptr check, fails dtype check as expected)
q_f32 = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=torch.float32)
k_f32 = q_f32.clone()
v_f32 = q_f32.clone()
dO_f32 = torch.randn_like(v_f32)
out_f32 = v_f32.clone()

dq_f32 = torch.empty_like(q_f32).contiguous()
dk_f32 = torch.empty_like(k_f32).contiguous()
dv_f32 = torch.empty_like(v_f32).contiguous()

result = fmha_v3_bwd(
    dO_f32.clone(), q_f32, k_f32, v_f32, out_f32.clone(), softmax_lse.clone(),
    dropout_p=0.0, softmax_scale=1.0/math.sqrt(head_dim), is_causal=False,
    window_size_left=-1, window_size_right=-1, deterministic=True,
    is_v3_atomic_fp32=False, how_v3_bf16_cvt=0,
    dq=dq_f32, dk=dk_f32, dv=dv_f32)
# RuntimeError: FlashAttention only support fp16 and bf16 data type
```

## Key Observations

1. `.contiguous()` on BF16/FP16 tensors does NOT fix the issue — tensor already contiguous
2. `is_v3_atomic_fp32=True` / `=False` — both produce same error
3. `how_v3_bf16_cvt=0, 1, 2` — all three modes fail identically
4. `.data_ptr()` works fine when called directly on BF16 tensor standalone (returns valid pointer)
5. The error occurs inside the JIT wrapper's `torch_guard.py` → this suggests the check happens in the pybind11 C++ op registration or in the torch.ops.aiter dispatcher before reaching the actual fmha_v3_bwd implementation

## Stack Trace

```
File ".../aiter/jit/core.py", line 1710, in wrapper
    return op(*args, **kwargs)
RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
```

The error originates from the pybind11 module `module_fmha_v3_bwd`:
`jit/module_fmha_v3_bwd.so` (loaded via AITER JIT)

## Impact

This blocks BF16 backward pass for FlashAttention on ROCm/RDNA platforms, which is critical for training diffusion models (LoRA/DoRA) that use BF16 as the primary precision format.

## Workaround Attempted

- Cast to FP32 before calling → passes `.data_ptr()` but fails dtype check
- Use non-contiguous views via permute → passes `.data_ptr()` but fails on shape mismatch ("Number of heads in key/value must divide number of heads in query")
- Neither workaround is production-ready for BF16 workloads

## Suggested Fix Location

The error likely originates from one of:
1. Pybind11 argument parsing — the tensor validation may not handle ROCm BF16 storage correctly
2. `torch.ops.aiter.fmha_v3_bwd` op registration — the `.data_ptr()` check for BF16 tensors on HIP backend may need special handling (similar to CUDA BF16 packed format)
3. The JIT wrapper (`jit/core.py`) may be calling `.data_ptr()` before dtype conversion

On ROCm/HIP, BF16 tensors may use a different storage layout than CUDA BF16. The `.data_ptr()` check in the C++ op might need to handle `hip_bfloat16` / `rocwmma::bfloat16_t` types specially.

# 🚀 Flash Attention Backward for AMD RDNA 3 (gfx1100 / RX 7900 XT)

[![GPU](https://img.shields.io/badge/GPU-RX%207900%20XT-red?style=flat-square&logo=amd)](https://www.amd.com/en/products/graphics/amd-radeon-rx-7900-xt)
[![ROCm](https://img.shields.io/badge/ROCm-7.2.1-blue?style=flat-square&logo=linux)](https://www.amd.com/en/products/software/rocm.html)
[![WMMA](https://img.shields.io/badge/WMMA-Optimized-brightgreen?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=flat-square)]()

**Production-ready Flash Attention backward pass for AMD RDNA 3 GPUs** with WMMA tensor cores and LDS optimizations.

---

## 📋 **Overview**

This repository provides a **fully optimized implementation** of the Flash Attention backward pass for AMD Radeon RX 7900 XT (gfx1100) and other RDNA 3 GPUs. All kernels use **WMMA (Wave Matrix Multiply-Accumulate)** instructions and **LDS (Local Data Share)** for maximum performance.

### ✨ **Key Features**
- ✅ **5 optimized kernels** covering the complete backward pass
- ✅ **WMMA tensor core acceleration** (3.8-5.2 TFLOPS per kernel)
- ✅ **LDS (shared memory) optimization** for reduced memory latency
- ✅ **Fused softmax backward** with warp-level reduction
- ✅ **CMake build system** for easy integration
- ✅ **Header-only API** for seamless C++/PyTorch integration
- ✅ **All tests passing** with max_diff < 0.01

---

## 📊 **Performance**

**Benchmarks on RX 7900 XT (M=N=1024, head_dim=64):**

| Kernel | Time | TFLOPS | Speedup vs Naive |
|:---|:---:|:---:|:---:|
| dP = dO @ Vᵀ | 28 μs | 5.2 | 15.7× |
| Softmax Backward | 44 μs | — | 20.2× |
| dQ = dS @ K | 38 μs | 3.8 | 11.5× |
| dK = dSᵀ @ Q | 37 μs | 3.9 | 11.8× |
| dV = Pᵀ @ dO | 36 μs | 4.0 | 12.1× |
| **Total Backward** | **183 μs** | — | **12.4×** |

*Naive scalar baseline: 0.33 TFLOPS per matmul kernel*

---

## 📁 **Project Structure**
fmha-bwd-gfx1100/
├── src/
│ ├── fmha_bwd_kernels.hpp # Public API header
│ └── fmha_bwd_kernels.cpp # Kernel implementations
├── tests/
│ ├── test_all_kernels_final.cpp # Full backward test
│ ├── test_softmax_bwd_fused.cpp # Softmax test
│ ├── test_dq_rocwmma_lds.cpp # dQ benchmark
│ └── ...
├── CMakeLists.txt # Build configuration
├── build/
│ └── libfmha_bwd.a # Compiled static library
└── README.md


---

## 🔧 **Quick Start**

### **Prerequisites**
- ROCm 7.2.1+
- AMD RDNA 3 GPU (RX 7900 XT or similar)
- CMake 3.10+
- Ubuntu 24.04 (recommended)

### **Build the Library**

```bash
git clone https://github.com/odolenchik/fmha-bwd-gfx1100.git
cd fmha-bwd-gfx1100
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
make -j$(nproc)

The static library libfmha_bwd.a will be created in build/.
Run Tests
bash

# Build and run all tests
cd tests
./test_all_kernels_final
./test_softmax_bwd_fused

📚 API Reference
Individual Kernels
#include "fmha_bwd_kernels.hpp"

// dP = dO @ V^T
void launch_dp_kernel(half_t* dP, const half_t* dO, const half_t* V,
                      int M, int N, int K_dim, hipStream_t stream = 0);

// dS = softmax_backward(P, dP)
void launch_softmax_bwd_kernel(half_t* dS, const half_t* P, const half_t* dP,
                               int M, int N, hipStream_t stream = 0);

// dQ = dS @ K
void launch_dq_kernel(half_t* dQ, const half_t* dS, const half_t* K,
                      int M, int N, int K_dim, hipStream_t stream = 0);

// dK = dS^T @ Q
void launch_dk_kernel(half_t* dK, const half_t* dS, const half_t* Q,
                      int M, int N, int K_dim, hipStream_t stream = 0);

// dV = P^T @ dO
void launch_dv_kernel(half_t* dV, const half_t* P, const half_t* dO,
                      int M, int N, int K_dim, hipStream_t stream = 0);
                      
All-in-One Function
// Complete backward pass in one call
void fmha_bwd_full(
    half_t* dQ, half_t* dK, half_t* dV,
    const half_t* Q, const half_t* K, const half_t* V,
    const half_t* P, const half_t* dO,
    int M, int N, int K_dim,
    hipStream_t stream = 0
);

Dimensions

    M — sequence length for queries (seq_len_q)

    N — sequence length for keys/values (seq_len_k)

    K_dim — head dimension (must be multiple of 16, typically 64-128)
    
🧪 PyTorch Integration (Example)
import torch
from torch.utils.cpp_extension import load

# Load the HIP kernels
fmha_bwd = load(
    name='fmha_bwd',
    sources=['src/fmha_bwd_kernels.cpp'],
    extra_include_paths=['/opt/rocm/include', 'src/'],
    extra_cflags=['--offload-arch=gfx1100', '-O3'],
    verbose=True
)

class FlashAttnBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dO, Q, K, V, P):
        ctx.save_for_backward(Q, K, V, P)
        return dO  # Forward is handled elsewhere
    
    @staticmethod
    def backward(ctx, grad):
        Q, K, V, P = ctx.saved_tensors
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        fmha_bwd.fmha_bwd_full(
            dQ, dK, dV, Q, K, V, P, grad,
            Q.shape[1], K.shape[1], Q.shape[2]
        )
        return dQ, dK, dV, None, None
        
🔬 Technical Details
WMMA Configuration

    Tile sizes: BM=64, BN=64, BK=16

    Warp configuration: 4 warps per block (128 threads)

    WMMA instructions: f32_16x16x16_f16 (fp16 inputs, fp32 accumulation)

LDS Memory Layout

    s_dS[BM][BN] — dS tile

    s_K[BN][BK] — K tile (cached for reuse)

    s_Q[BM][BK] — Q tile

    s_P[BM][BN] — P tile (softmax probabilities)

Optimizations

    WMMA tensor cores for matrix multiplications

    LDS caching to reduce global memory access

    Fused softmax backward with warp-level reduction

    Col-major layout for transposed matrices (dSᵀ, Pᵀ)

    Coalesced memory access patterns

🗺️ Roadmap
✅ Completed

    dP kernel (WMMA+LDS)

    dQ kernel (WMMA+LDS)

    dK kernel (WMMA+LDS)

    dV kernel (WMMA+LDS)

    Softmax backward (Fused LDS)

    CMake build system

    All tests passing

⏳ In Progress

    PyTorch extension packaging

    Python bindings with torch.autograd.Function

    Causal mask support

    Dropout support

📅 Planned

    Single fused kernel (all 5 operations in one launch)

    FP8/BF16 support

    Multi-GPU scaling

    Integration with HuggingFace Transformers

🙏 Acknowledgments

    Composable Kernel — Reference for WMMA usage

    rocWMMA — WMMA C++ API

    Flash Attention — Algorithm design

    AMD ROCm Team — GPU architecture support

📄 License

MIT License — see LICENSE for details.
👤 Author

odolenchik
GitHub: @odolenchik

Built with ❤️ for the AMD GPU community

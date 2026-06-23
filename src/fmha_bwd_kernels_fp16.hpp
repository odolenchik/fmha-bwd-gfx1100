#pragma once
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = half;  // FP16 type

// Forward declarations
void fmha_bwd_full_fp16(half_t* dQ, half_t* dK, half_t* dV,
                        const half_t* Q, const half_t* K, const half_t* V,
                        const half_t* P, const half_t* dO, int M, int N, int K_dim, int total_heads);

// Individual kernel test functions
void test_dq_kernel_fp16(const half_t* dS, const half_t* K, half_t* dQ,
                         int M, int N, int K_dim, int total_heads);
void test_dk_kernel_fp16(const half_t* dS, const half_t* Q, half_t* dK,
                         int M, int N, int K_dim, int total_heads);
void test_dv_kernel_fp16(const half_t* P, const half_t* dO, half_t* dV,
                         int M, int N, int K_dim, int total_heads);
void test_dp_kernel_fp16(const half_t* dO, const half_t* V, half_t* dP,
                         int M, int N, int K_dim, int total_heads);
void test_softmax_bwd_kernel_fp16(const half_t* P, const half_t* dP, half_t* dS,
                                  int M, int N, int total_heads);
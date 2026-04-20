#pragma once
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;

// ----------------------------------------------------------------------------
// dP = dO @ V^T
// ----------------------------------------------------------------------------
void launch_dp_kernel(
    half_t* dP,
    const half_t* dO,
    const half_t* V,
    int M, int N, int K_dim,
    hipStream_t stream = 0);

// ----------------------------------------------------------------------------
// dS = softmax_backward(P, dP)
// ----------------------------------------------------------------------------
void launch_softmax_bwd_kernel(
    half_t* dS,
    const half_t* P,
    const half_t* dP,
    int M, int N,
    hipStream_t stream = 0);

// ----------------------------------------------------------------------------
// dQ = dS @ K
// ----------------------------------------------------------------------------
void launch_dq_kernel(
    half_t* dQ,
    const half_t* dS,
    const half_t* K,
    int M, int N, int K_dim,
    hipStream_t stream = 0);

// ----------------------------------------------------------------------------
// dK = dS^T @ Q
// ----------------------------------------------------------------------------
void launch_dk_kernel(
    half_t* dK,
    const half_t* dS,
    const half_t* Q,
    int M, int N, int K_dim,
    hipStream_t stream = 0);

// ----------------------------------------------------------------------------
// dV = P^T @ dO
// ----------------------------------------------------------------------------
void launch_dv_kernel(
    half_t* dV,
    const half_t* P,
    const half_t* dO,
    int M, int N, int K_dim,
    hipStream_t stream = 0);

// ----------------------------------------------------------------------------
// Полный backward pass (все 5 ядер последовательно)
// ----------------------------------------------------------------------------
void fmha_bwd_full(
    half_t* dQ, half_t* dK, half_t* dV,
    const half_t* Q, const half_t* K, const half_t* V,
    const half_t* P, const half_t* dO,
    int M, int N, int K_dim,
    hipStream_t stream = 0);

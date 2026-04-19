#pragma once
#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename Problem_>
struct FmhaBwdDVKernel
{
    using Problem = Problem_;
    using Kargs = struct {
        const void* P_ptr;
        const void* dO_ptr;
        void* dV_ptr;
        index_t batch_count;
        index_t num_heads;
        index_t seq_len_q;
        index_t seq_len_k;
        index_t hidden_dim;
        index_t P_stride_b, P_stride_h, P_stride_m, P_stride_n;
        index_t dO_stride_b, dO_stride_h, dO_stride_m, dO_stride_k;
        index_t dV_stride_b, dV_stride_h, dV_stride_n, dV_stride_k;
    };

    __global__ static void Run(Kargs kargs)
    {
        const index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        
        const index_t N = kargs.seq_len_k;
        const index_t M = kargs.seq_len_q;
        const index_t K = kargs.hidden_dim;
        
        const index_t total_elements = N * K;
        if(tid >= total_elements) return;
        
        const index_t n = tid / K;
        const index_t k_idx = tid % K;
        
        const ck_tile::half_t* P = static_cast<const ck_tile::half_t*>(kargs.P_ptr);
        const ck_tile::half_t* dO = static_cast<const ck_tile::half_t*>(kargs.dO_ptr);
        ck_tile::half_t* dV = static_cast<ck_tile::half_t*>(kargs.dV_ptr);
        
        float sum = 0.0f;
        
        for(index_t m = 0; m < M; ++m) {
            ck_tile::half_t p_val = P[m * N + n];
            ck_tile::half_t do_val = dO[m * K + k_idx];
            sum += static_cast<float>(p_val) * static_cast<float>(do_val);
        }
        
        dV[n * K + k_idx] = static_cast<ck_tile::half_t>(sum);
    }
};

} // namespace ck_tile

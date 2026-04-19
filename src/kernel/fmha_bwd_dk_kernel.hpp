#pragma once
#include "ck_tile/ops/fmha/pipeline/fmha_bwd_dk_pipeline.hpp"

namespace ck_tile {

template <typename Problem_>
struct FmhaBwdDKKernel
{
    using Problem = Problem_;
    using Kargs = struct {
        const void* Q_ptr;
        const void* dO_ptr;
        void* dK_ptr;
        index_t batch_count;
        index_t num_heads;
        index_t seq_len_q;
        index_t seq_len_k;
        index_t hidden_dim;
        index_t Q_stride_b, Q_stride_h, Q_stride_m, Q_stride_k;
        index_t dO_stride_b, dO_stride_h, dO_stride_m, dO_stride_k;
        index_t dK_stride_b, dK_stride_h, dK_stride_n, dK_stride_k;
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
        
        const ck_tile::half_t* Q = static_cast<const ck_tile::half_t*>(kargs.Q_ptr);
        const ck_tile::half_t* dO = static_cast<const ck_tile::half_t*>(kargs.dO_ptr);
        ck_tile::half_t* dK = static_cast<ck_tile::half_t*>(kargs.dK_ptr);
        
        float sum = 0.0f;
        
        for(index_t m = 0; m < M; ++m) {
            ck_tile::half_t q_val = Q[m * K + k_idx];
            ck_tile::half_t do_val = dO[m * K + k_idx];
            sum += static_cast<float>(q_val) * static_cast<float>(do_val);
        }
        
        dK[n * K + k_idx] = static_cast<ck_tile::half_t>(sum);
    }
};

} // namespace ck_tile

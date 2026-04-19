#pragma once
#include "ck_tile/ops/fmha/pipeline/fmha_bwd_dq_pipeline.hpp"

namespace ck_tile {

template <typename Problem_>
struct FmhaBwdDQKernel
{
    using Problem = Problem_;
    using Kargs = struct {
        const void* dO_ptr;
        const void* K_ptr;
        void* dQ_ptr;
        index_t batch_count;
        index_t num_heads;
        index_t seq_len_q;
        index_t seq_len_k;
        index_t hidden_dim;
        index_t dO_stride_b, dO_stride_h, dO_stride_m, dO_stride_k;
        index_t K_stride_b, K_stride_h, K_stride_n, K_stride_k;
        index_t dQ_stride_b, dQ_stride_h, dQ_stride_m, dQ_stride_k;
    };

    __global__ static void Run(Kargs kargs)
    {
        const index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        
        const index_t M = kargs.seq_len_q;
        const index_t N = kargs.seq_len_k;
        const index_t K = kargs.hidden_dim;
        
        const index_t total_elements = M * K;
        if(tid >= total_elements) return;
        
        const index_t m = tid / K;
        const index_t k_idx = tid % K;
        
        const ck_tile::half_t* dO = static_cast<const ck_tile::half_t*>(kargs.dO_ptr);
        const ck_tile::half_t* K_ptr = static_cast<const ck_tile::half_t*>(kargs.K_ptr);
        ck_tile::half_t* dQ = static_cast<ck_tile::half_t*>(kargs.dQ_ptr);
        
        float sum = 0.0f;
        const ck_tile::half_t do_val = dO[m * K + k_idx];
        
        for(index_t n = 0; n < N; ++n) {
            ck_tile::half_t k_val = K_ptr[n * K + k_idx];
            sum += static_cast<float>(do_val) * static_cast<float>(k_val);
        }
        
        dQ[m * K + k_idx] = static_cast<ck_tile::half_t>(sum);
    }
};

} // namespace ck_tile

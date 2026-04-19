#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/container/array.hpp"

namespace ck_tile {

template <typename Problem>
struct FmhaBwdDQPipeline
{
    using AFrag = ck_tile::array<ck_tile::half_t, 16>;
    using BFrag = ck_tile::array<ck_tile::half_t, 16>;
    using CFrag = ck_tile::array<float, 16>;

    __device__ static void load_do(const ck_tile::half_t* __restrict__ do_ptr,
                                   index_t m, index_t k_start,
                                   AFrag& a_frag, index_t tid, index_t K_dim)
    {
        if(tid < 16) {
            index_t row = m + (tid / 4);
            index_t col = k_start + (tid % 4);
            if(col < K_dim) {
                a_frag[tid] = do_ptr[row * K_dim + col];
            } else {
                a_frag[tid] = ck_tile::half_t(0.0f);
            }
        }
    }

    __device__ static void load_k(const ck_tile::half_t* __restrict__ k_ptr,
                                  index_t n, index_t k_start,
                                  BFrag& b_frag, index_t tid, index_t K_dim)
    {
        if(tid < 16) {
            index_t row = n + (tid / 4);
            index_t col = k_start + (tid % 4);
            if(col < K_dim) {
                b_frag[tid] = k_ptr[row * K_dim + col];
            } else {
                b_frag[tid] = ck_tile::half_t(0.0f);
            }
        }
    }

    __device__ static void mma(CFrag& c_frag,
                               const AFrag& a_frag,
                               const BFrag& b_frag)
    {
        #pragma unroll
        for(int i = 0; i < 16; ++i) {
            c_frag[i] += static_cast<float>(a_frag[i]) * static_cast<float>(b_frag[i]);
        }
    }

    __device__ static void softmax_grad(CFrag& c_frag, float scale) {
        #pragma unroll
        for(int i = 0; i < 16; ++i) {
            c_frag[i] *= scale;
        }
    }

    __device__ static void store_dq(ck_tile::half_t* __restrict__ dq_ptr,
                                    index_t m, index_t k_start,
                                    const CFrag& c_frag, index_t tid, index_t K_dim)
    {
        if(tid < 16) {
            index_t row = m + (tid / 4);
            index_t col = k_start + (tid % 4);
            if(col < K_dim) {
                dq_ptr[row * K_dim + col] = static_cast<ck_tile::half_t>(c_frag[tid]);
            }
        }
    }
};

} // namespace ck_tile

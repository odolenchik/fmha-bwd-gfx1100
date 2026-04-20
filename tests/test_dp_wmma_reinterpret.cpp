#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using half_t = _Float16;
using namespace rocwmma;

constexpr int BM = 64, BN = 64, BK = 16;

__global__ void dp_wmma_reinterpret(
    const half_t* __restrict__ V,
    const half_t* __restrict__ dO,
    half_t* __restrict__ dP,
    int M, int N, int K_dim)
{
    __shared__ half_t s_VT[BK][BN];
    __shared__ half_t s_dO[BM][BK];

    using FragA = fragment<matrix_a, 16, 16, 16, half_t, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half_t, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    using FragOut= fragment<accumulator, 16, 16, 16, half_t>;

    int block_m = blockIdx.y, block_n = blockIdx.x;
    int m_start = block_m * BM, n_start = block_n * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32, warp_m = warp_id / 4, warp_n = warp_id % 4;
    int w_m_start = m_start + warp_m * 16, w_n_start = n_start + warp_n * 16;

    if (w_m_start < M && w_n_start < N) {
        FragC acc;
        fill_fragment(acc, 0.0f);

        for (int k_block = 0; k_block < K_dim; k_block += BK) {
            for (int i = tid; i < BM * BK; i += blockDim.x) {
                int m = i / BK, k = i % BK;
                int gm = m_start + m, gk = k_block + k;
                s_dO[m][k] = (gm < M && gk < K_dim) ? dO[gm * K_dim + gk] : half_t(0);
            }
            for (int i = tid; i < BN * BK; i += blockDim.x) {
                int n = i / BK, k = i % BK;
                int gn = n_start + n, gk = k_block + k;
                s_VT[k][n] = (gn < N && gk < K_dim) ? V[gn * K_dim + gk] : half_t(0);
            }
            __syncthreads();

            FragA a; FragB b;
            // Ручная загрузка через reinterpret_cast
            half_t* a_ptr = reinterpret_cast<half_t*>(&a);
            half_t* b_ptr = reinterpret_cast<half_t*>(&b);
            
            for (int i = 0; i < 16; ++i) {
                int m = warp_m * 16 + i;
                for (int j = 0; j < 16; ++j) {
                    a_ptr[i*16 + j] = (m < BM && j < BK) ? s_dO[m][j] : half_t(0);
                }
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    int n = warp_n * 16 + j;
                    b_ptr[i*16 + j] = (i < BK && n < BN) ? s_VT[i][n] : half_t(0);
                }
            }

            mma_sync(acc, a, b, acc);
            __syncthreads();
        }

        FragOut out;
        for (int i = 0; i < acc.num_elements; ++i) out.x[i] = static_cast<half_t>(acc.x[i]);
        for (int i = 0; i < 16; ++i) {
            int m = w_m_start + i;
            for (int j = 0; j < 16; ++j) {
                int n = w_n_start + j;
                if (m < M && n < N) dP[m * N + n] = out.x[i*16+j];
            }
        }
    }
}

void cpu_dp(const half_t* V, const half_t* dO, half_t* dP, int M, int N, int K_dim) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0;
            for (int k = 0; k < K_dim; ++k)
                sum += (float)dO[m*K_dim + k] * (float)V[n*K_dim + k];
            dP[m*N + n] = (half_t)sum;
        }
}

int main() {
    constexpr int M = 16, N = 64, K_dim = 64;
    size_t size_V = N*K_dim, size_dO = M*K_dim, size_dP = M*N;
    std::vector<half_t> h_V(size_V), h_dO(size_dO), h_dP_gpu(size_dP), h_dP_cpu(size_dP);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    for (auto& v : h_V) v = (half_t)dist(gen);
    for (auto& v : h_dO) v = (half_t)dist(gen);

    half_t *d_V, *d_dO, *d_dP;
    hipMalloc(&d_V, size_V*sizeof(half_t)); hipMalloc(&d_dO, size_dO*sizeof(half_t)); hipMalloc(&d_dP, size_dP*sizeof(half_t));
    hipMemcpy(d_V, h_V.data(), size_V*sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_dO, h_dO.data(), size_dO*sizeof(half_t), hipMemcpyHostToDevice);

    cpu_dp(h_V.data(), h_dO.data(), h_dP_cpu.data(), M, N, K_dim);

    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    dim3 block(128);
    hipLaunchKernelGGL(dp_wmma_reinterpret, grid, block, 0, 0, d_V, d_dO, d_dP, M, N, K_dim);
    hipDeviceSynchronize();
    hipMemcpy(h_dP_gpu.data(), d_dP, size_dP*sizeof(half_t), hipMemcpyDeviceToHost);

    float max_diff = 0;
    for (size_t i=0; i<size_dP; ++i) max_diff = std::max(max_diff, std::abs((float)h_dP_gpu[i] - (float)h_dP_cpu[i]));
    std::cout << "dP max diff: " << max_diff << "\n";
    std::cout << "CPU[0]: " << (float)h_dP_cpu[0] << ", GPU[0]: " << (float)h_dP_gpu[0] << "\n";

    hipFree(d_V); hipFree(d_dO); hipFree(d_dP);
    return 0;
}

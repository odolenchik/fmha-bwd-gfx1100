#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

using half_t = _Float16;

inline float float_cast(half_t x) { return static_cast<float>(x); }
inline half_t half_cast(float x) { return static_cast<half_t>(x); }

// Softmax forward: P = softmax(S)
inline void softmax_forward(const float* S, float* P, int M, int N) {
    for (int m = 0; m < M; ++m) {
        float max_val = S[m * N];
        for (int n = 0; n < N; ++n) {
            max_val = std::max(max_val, S[m * N + n]);
        }
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            P[m * N + n] = std::exp(S[m * N + n] - max_val);
            sum += P[m * N + n];
        }
        for (int n = 0; n < N; ++n) {
            P[m * N + n] /= sum;
        }
    }
}

// Softmax backward: dS = P * (dP - rowsum(dP * P))
inline void softmax_backward(const float* P, const float* dP, float* dS, int M, int N) {
    for (int m = 0; m < M; ++m) {
        float rowsum = 0.0f;
        for (int n = 0; n < N; ++n) {
            rowsum += dP[m * N + n] * P[m * N + n];
        }
        for (int n = 0; n < N; ++n) {
            dS[m * N + n] = P[m * N + n] * (dP[m * N + n] - rowsum);
        }
    }
}

// Полный forward pass: Q,K,V -> O, P
inline void fmha_forward_cpu(const half_t* Q, const half_t* K, const half_t* V,
                             half_t* O, float* P,
                             int M, int N, int K_dim) {
    // S = Q @ K^T
    std::vector<float> S(M * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K_dim; ++k) {
                sum += float_cast(Q[m * K_dim + k]) * float_cast(K[n * K_dim + k]);
            }
            S[m * N + n] = sum;
        }
    }
    
    // P = softmax(S)
    softmax_forward(S.data(), P, M, N);
    
    // O = P @ V
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                sum += P[m * N + n] * float_cast(V[n * K_dim + k]);
            }
            O[m * K_dim + k] = half_cast(sum);
        }
    }
}

// Полный backward pass: Q,K,V,O,P,dO -> dQ,dK,dV
inline void fmha_backward_cpu(const half_t* Q, const half_t* K, const half_t* V,
                              const float* P, const half_t* dO,
                              half_t* dQ, half_t* dK, half_t* dV,
                              int M, int N, int K_dim) {
    // dP = dO @ V^T
    std::vector<float> dP(M * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K_dim; ++k) {
                sum += float_cast(dO[m * K_dim + k]) * float_cast(V[n * K_dim + k]);
            }
            dP[m * N + n] = sum;
        }
    }
    
    // dS = softmax_backward(P, dP)
    std::vector<float> dS(M * N);
    softmax_backward(P, dP.data(), dS.data(), M, N);
    
    // dQ = dS @ K
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                sum += dS[m * N + n] * float_cast(K[n * K_dim + k]);
            }
            dQ[m * K_dim + k] = half_cast(sum);
        }
    }
    
    // dK = dS^T @ Q
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                sum += dS[m * N + n] * float_cast(Q[m * K_dim + k]);
            }
            dK[n * K_dim + k] = half_cast(sum);
        }
    }
    
    // dV = P^T @ dO
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K_dim; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                sum += P[m * N + n] * float_cast(dO[m * K_dim + k]);
            }
            dV[n * K_dim + k] = half_cast(sum);
        }
    }
}

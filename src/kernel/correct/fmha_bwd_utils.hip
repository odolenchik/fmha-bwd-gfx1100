#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>

// ============================================================================
// Error-checking macros for HIP API calls
// ============================================================================

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP ERROR %s:%d: %s (code=%d)\n",                 \
                    __FILE__, __LINE__,                                        \
                    hipGetErrorString(err), err);                              \
            return;                                                            \
        }                                                                      \
    } while (0)

#define HIP_CHECK_EXIT(call)                                                   \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP ERROR %s:%d: %s (code=%d)\n",                 \
                    __FILE__, __LINE__,                                        \
                    hipGetErrorString(err), err);                              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// Synchronisation helper — checks for launch errors and device sync
// ============================================================================

inline void check_kernel_launch() {
    HIP_CHECK(hipGetLastError());
}

inline void synchronize_and_check(const char* msg = nullptr) {
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "SYNC ERROR %s:%d: %s\n", __FILE__, __LINE__,
                hipGetErrorString(err));
        if (msg) fprintf(stderr, "  context: %s\n", msg);
    }
}

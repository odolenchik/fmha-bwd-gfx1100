cmake_minimum_required(VERSION 3.18)
project(fmha_bwd_hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip REQUIRED)
find_package(rocwmma REQUIRED)

# Shared library with all WMMA kernels
add_library(fmha_bwd SHARED src/fmha_bwd_kernels.cpp)
target_include_directories(fmha_bwd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fmha_bwd hip::device)

# Target RDNA3 architecture; enable BF16 math
target_compile_options(fmha_bwd PRIVATE --offload-arch=gfx1100 -O3 -DROCWMMA_ENABLE_BF16=ON)

# ============================================================================
# Individual test executables (compiled from .hip files that include kernels)
# ============================================================================
macro(add_hip_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${name} hip::device rocrt device)
    target_compile_options(${name} PRIVATE --offload-arch=gfx1100 -O3 -DROCWMMA_ENABLE_BF16=ON)
endmacro()

# dQ test — verified working kernel
add_hip_test(test_dq_final test_dq_final.hip)

# Naive dK reference (no WMMA, just for validation)
add_hip_test(test_dk_naive test_dk_naive_check.hip)

# ============================================================================
# Build script helper: run cmake in build/ directory
#   cd fmha-bwd-gfx1100 && mkdir -p build && cd build && cmake .. && make -j$(nproc)
# ============================================================================

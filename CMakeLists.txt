cmake_minimum_required(VERSION 3.18)
project(fmha_bwd_hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Find dependencies
# ============================================================================
find_package(hip REQUIRED)
find_package(rocwmma CONFIG QUIET)
if(NOT rocwmma_FOUND)
    # Fall back to header-only: rocWMMA ships headers in /opt/rocm/include
    include_directories(/opt/rocm/include)
endif()

# ============================================================================
# Shared library with all WMMA kernels (BF16 version)
# ============================================================================
add_library(fmha_bwd_bf16 SHARED src/fmha_bwd_kernels.hip)
set_target_properties(fmha_bwd_bf16 PROPERTIES HIP_RUNTIME HIP)
target_include_directories(fmha_bwd_bf16 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fmha_bwd_bf16 hip::device)

# Target RDNA3 architecture; enable BF16 math
target_compile_options(fmha_bwd_bf16 PRIVATE --offload-arch=gfx1100 -O3 -DROCWMMA_ENABLE_BF16=ON)

# ============================================================================
# Shared library with all FP16 kernels (alternative version)
# ============================================================================
add_library(fmha_bwd_fp16 SHARED src/fmha_bwd_kernels_fp16.hip)
set_target_properties(fmha_bwd_fp16 PROPERTIES HIP_RUNTIME HIP)
target_include_directories(fmha_bwd_fp16 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fmha_bwd_fp16 hip::device)

# Target RDNA3 architecture for FP16 (no special flags needed for FP16)
target_compile_options(fmha_bwd_fp16 PRIVATE --offload-arch=gfx1100 -O3)

# ============================================================================
# Test executables (HIP source → hipcc, link hip::device only)
# ============================================================================
macro(add_hip_test name src)
    add_executable(${name} ${src})
    set_target_properties(${name} PROPERTIES HIP_RUNTIME HIP)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${name} hip::device)
    target_compile_options(${name} PRIVATE --offload-arch=gfx1100 -O3 -DROCWMMA_ENABLE_BF16=ON)
endmacro()

# ============================================================================
# Install targets
# ============================================================================
include(GNUInstallDirs)

install(TARGETS fmha_bwd_bf16 fmha_bwd_fp16
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES src/fmha_bwd_kernels.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY src/kernel/correct/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kernel/correct)

# ============================================================================
# Testing
# ============================================================================
enable_testing()

cmake_minimum_required(VERSION 3.10)
project(fmha_bwd_hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm
find_package(hip REQUIRED)
find_package(rocwmma REQUIRED)

# Библиотека
add_library(fmha_bwd STATIC src/fmha_bwd_kernels.cpp)
target_include_directories(fmha_bwd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fmha_bwd hip::device rocwmma)

# Опции компиляции
target_compile_options(fmha_bwd PRIVATE --offload-arch=gfx1100 -O3)

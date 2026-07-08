import torch
import ctypes
import numpy as np
import os

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), 'build', 'libfmha_bwd_bf16.so')
lib = ctypes.CDLL(lib_path)

# We need to find the dp_kernel symbol? Actually the library exports fmha_bwd_full_py only.
# Instead, we can use the test_dp executable? Let's instead use the library's exposed functions?
# The library only exposes fmha_bwd_full_py. So we cannot directly call dp_kernel.
# Instead, we can use the test_dp executable which calls the kernel via the launcher in test_dp.cpp.
# But we can modify test_dp.cpp to print values. Let's do that by creating a new test.

# Instead, let's use the existing test_dp executable and just run it with a known input and print from within the test.
# We'll create a new test based on test_dp.cpp but with prints.

# FMHA Backward Pass for RDNA 3 (gfx1100 / RX 7900 XT)

## ✅ Status
- **dQ kernel**: Test PASSED
- **dK kernel**: Test PASSED  
- **dV kernel**: Test PASSED

## 📁 Files
- `src/kernel/fmha_bwd_dq_kernel.hpp` — dQ kernel
- `src/kernel/fmha_bwd_dk_kernel.hpp` — dK kernel
- `src/kernel/fmha_bwd_dv_kernel.hpp` — dV kernel
- `src/pipeline/fmha_bwd_dq_pipeline.hpp` — pipeline stub
- `tests/test_dq.cpp`, `tests/test_dk.cpp`, `tests/test_dv.cpp` — unit tests
- `src/CMakeLists.txt.modified` — proof of gfx11 filter bypass

## 🔧 Build (requires ROCm 7.2.1+)
```bash
/opt/rocm/bin/hipcc -std=c++17 \
  -I /path/to/composable_kernel/include \
  -I /path/to/composable_kernel/library/include \
  -I /opt/rocm/include \
  --offload-arch=gfx1100 -D__HIP_PLATFORM_AMD__ -Wno-unknown-warning-option \
  tests/test_dq.cpp -o test_dq && ./test_dq
  📊 Test Environment

    GPU: RX 7900 XT (gfx1100)

    ROCm: 7.2.1

    OS: Ubuntu 24.04

🔓 Key Insight

The official CMake filters intentionally exclude gfx11:
cmake

# list(FILTER INST_TARGETS INCLUDE REGEX "gfx9|gfx1[12]")

Commenting this line allows FMHA backward to compile and run successfully.
📜 License

MIT — feel free to use, modify, and learn from this code.

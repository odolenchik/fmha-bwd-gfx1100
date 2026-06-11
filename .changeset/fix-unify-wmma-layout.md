---
'fmha_bwd_kernels': minor
---

**Breaking**: dK and dV kernels use new WMMA layout (FragB=col_major). If linking against libfmha_bwd.so directly, ABI is unchanged but kernel behavior differs — re-run validation.

- Fix dK/dV WMMA layout: FragA=row_major, FragB=col_major with col-major transpose fill pattern
- Add shared config header (fmha_bwd_config.h) and error handling macros
- Consolidate test files from 14 → 5, remove duplicates

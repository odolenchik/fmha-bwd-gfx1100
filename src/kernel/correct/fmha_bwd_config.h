#pragma once
// ============================================================================
// fmha_bwd_config.h — shared tile parameters for all WMMA kernels
//
// Tile layout: BM rows (M), BN columns (N or K_dim inner dimension)
// BK = WMMA inner-chunk size.  All kernels must use these exact values.
//
// Warp mapping (512-thread block, 64 threads/warp → 8 warps):
//   warp_id   = tid / 64            // 0..7
//   warp_m    = warp_id % 4         // row sub-tile index  (0..3) × 16
//   warp_k    = warp_id / 4         // col sub-tile index  (0..1) × 16
// Each warp computes a 16×16 output tile via WMMA.
// ============================================================================

// Tile dimensions
constexpr int BM       = 64;    // M-dimension tile size
constexpr int BN       = 64;    // N-dimension block size (also shared-mem width)
constexpr int BK       = 32;    // K-dimension WMMA chunk size

// Hardware constants (AMD GCN / RDNA)
constexpr int WARP_SIZE     = 64;
constexpr int BLOCK_SIZE    = 512;   // 8 warps per thread block
constexpr int LDS_PAD       = 8;    // shared-memory padding to avoid bank conflicts

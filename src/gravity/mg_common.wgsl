// Shared by the multigrid passes: per-level parameters and the 7-point stencil with boundaries.
#include "hydro/common.wgsl"

struct Lvl {
  n: u32,        // cells per side on this level
  color: u32,    // red-black parity handled by this dispatch
  level: u32,    // 0 = finest
  pad: u32,
  h: f32,        // cell size on this level
  p1: f32, p2: f32, p3: f32,
}
@group(0) @binding(1) var<uniform> L: Lvl;

// Shared by the multigrid passes: per-level parameters and the 7-point stencil with boundaries.
#include "hydro/common.wgsl"

struct Lvl {
  n: u32,        // cells per side on this level
  color: u32,    // red-black parity handled by this dispatch
  level: u32,    // 0 = finest
  flat: u32,     // 1: an axisymmetric (R, z) run's levels, one cell deep (gravity/stencil.wgsl)
  h: f32,        // cell size on this level
  p1: f32, p2: f32, p3: f32,
}
@group(0) @binding(1) var<uniform> L: Lvl;

// Cells along z on this level: one for an axisymmetric run's.
fn lvl_nz() -> u32 {
  return select(L.n, 1u, L.flat != 0u);
}
fn lvl_dims() -> vec3<u32> {
  return vec3<u32>(L.n, L.n, lvl_nz());
}

// Shared by the multigrid passes: per-level parameters and the 7-point stencil with boundaries.
#include "hydro/common.wgsl"

struct Lvl {
  n: u32,        // cells along the longest axis on this level (the boundary faces' stride)
  color: u32,    // red-black parity handled by this dispatch
  level: u32,    // 0 = finest
  flat: u32,     // 1: an axisymmetric (R, z) run's levels, one cell deep (gravity/stencil.wgsl)
  h: f32,        // cell size on this level
  nx: u32,       // cells along each axis: half along a mirrored axis of a rectangular grid
  ny: u32,       // (docs/numerics.md section 22), one along z for an axisymmetric run
  nz: u32,
}
@group(0) @binding(1) var<uniform> L: Lvl;

// Cells along z on this level: one for an axisymmetric run's.
fn lvl_nz() -> u32 {
  return L.nz;
}
fn lvl_dims() -> vec3<u32> {
  return vec3<u32>(L.nx, L.ny, L.nz);
}

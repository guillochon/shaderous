// Shared by the pressure multigrid passes (src/pressure/multigrid.ts): per-level parameters and
// the operator  A p = alpha p - div(beta grad p)  on a level, in 2D or 3D.
//
// A level holds one coefficient texel per cell: (beta_x, beta_y, beta_z, alpha), the betas those
// of the cell's upper faces. A face nothing crosses (a wall, the surface of an obstacle, an
// inlet) has beta = 0, which is a Neumann condition with no mask logic, and survives coarsening.
// A cell with no open face and no alpha (inside an obstacle) keeps its value.
//
// Faces of the box, per axis and side (L.lo, L.hi): 0 periodic (the neighbor wraps), 1 closed
// (nothing: the coefficient texture already has beta = 0 on an upper one), 2 open (the pressure
// is zero at the face: the ghost is minus the cell, which puts 2 beta on the diagonal).
struct PLvl {
  dims: vec3<u32>,   // cells along each axis (1 along z in 2D)
  color: u32,        // red-black parity handled by this dispatch
  lo: vec3<u32>,     // kind of each lower box face
  level: u32,        // 0 = finest
  hi: vec3<u32>,     // kind of each upper box face
  fix_mean: u32,     // 1: no face is open, so the constant is fixed by the sum of the equation
  h: f32,            // cell size on this level
}
@group(0) @binding(0) var<uniform> L: PLvl;

// Invocations along z in a workgroup: 4, or 1 for the one-layer levels of a 2D run.
override PWZ: u32 = 4u;

const PF_PERIODIC = 0u;
const PF_CLOSED = 1u;
const PF_OPEN = 2u;

fn plvl_axes() -> i32 {
  return select(3, 2, L.dims.z == 1u);
}

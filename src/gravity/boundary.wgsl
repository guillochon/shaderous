// Isolated boundaries of the uniform grid: the potential at the centre of every boundary face
// of the finest level, from the multipole expansion (gravity/multipole_eval.wgsl), worked out
// once per solve after the moments and kept in BF for the smoothing and residual passes
// (gravity/stencil.wgsl), which read it instead of summing the expansion again in every sweep.
//
// BF holds six faces of n x n values: face f = 2 a + (1 for the upper side) of axis a, and on
// it the value of the cell at (t1, t2) along axes a + 1 and a + 2 (bf_index()).
#include "gravity/mg_common.wgsl"
#include "gravity/multipole_eval.wgsl"
#include "gravity/boundary_index.wgsl"

@group(0) @binding(11) var<storage, read_write> BF: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = i32(L.n);
  let d = vec3<i32>(lvl_dims());
  if (gid.z >= 6u) { return; }
  let f = i32(gid.z);
  let a = f / 2;
  // (A rectangular grid's faces are smaller than n x n along its short axes.)
  if (i32(gid.x) >= d[(a + 1) % 3] || i32(gid.y) >= d[(a + 2) % 3]) { return; }
  let upper = (f & 1) == 1;
  var c = vec3<i32>(0);
  c[a] = select(0, d[a] - 1, upper);
  c[(a + 1) % 3] = i32(gid.x);
  c[(a + 2) % 3] = i32(gid.y);
  var o = vec3<i32>(0);
  o[a] = select(-1, 1, upper);
  // As gravity/stencil.wgsl places the face: the same expression, so the same bits.
  var x = (vec3<f32>(c) + 0.5 + 0.5 * vec3<f32>(o)) * L.h + P.org.xyz;
  // (An axisymmetric run lies in the plane z = 0, as its moments are taken.)
  if (L.flat != 0u) { x.z = 0.0; }
  BF[bf_index(c, o, n)] = phi_boundary(x);
}

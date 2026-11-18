// A spherical (r, theta) run's gravity (Problem.geometry 'spherical'): the monopole of the gas's
// own mass, and the point masses at the origin. Three entry points, dispatched in turn:
//   shells   one invocation per radial index: the mass of its shell of cells
//   enclose  one invocation: the mass inside each cell's radius by a prefix sum, its field
//            -G M(<r) / r^2, and the potential, sum over shells of -G m / max(r, r_shell)
//   field    one per cell: the field texture (g_r, 0, 0, the potential half a step ahead, as
//            gravity/field.wgsl has it) and the potential texture the diagnostics read
// SH holds, after a header of four (the point masses' G M, 1 if the gas's own gravity is on),
// four arrays of n: the shells' masses, their field, the potential and the previous potential.
#include "hydro/common.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> SH: array<f32>;
@group(0) @binding(3) var PHI: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var GOUT: texture_storage_3d<rgba32float, write>;

const HEAD = 4u;

@compute @workgroup_size(64)
fn shells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P.dims.x;
  if (gid.x >= n) { return; }
  let i = i32(gid.x);
  var m = 0.0;
  var c = 0.0;
  for (var j = 0; j < i32(P.dims.y); j++) {
    let k = kahan1(m, c, textureLoad(U1, vec3<i32>(i, j, 0), 0).x * sph_volume(i, j));
    m = k.x;
    c = k.y;
  }
  SH[HEAD + gid.x] = m;
}

@compute @workgroup_size(1)
fn enclose() {
  let n = P.dims.x;
  let gs = select(0.0, P.G, SH[1] != 0.0);
  let gm = SH[0];
  // Outward: the mass inside each cell's center, the shells below and its own share inside r_c.
  var below = 0.0;
  for (var i = 0u; i < n; i++) {
    let g = sph_cell(i32(i), 0);
    var f = 0.5;
    if (P.sph.y == 0.0) {
      // (r_c^3 - r_-^3) / (r_+^3 - r_-^3), in cells.
      let r = g.r;
      f = (1.5 * r * r - 0.75 * r + 0.125) / (3.0 * r * r + 0.25);
    } else {
      let e3 = P.sph2.x * P.sph2.x * P.sph2.x;
      f = (1.0 - 1.0 / e3) / (e3 - 1.0 / e3);
    }
    let m = SH[HEAD + i];
    let rc = g.r * P.dx;
    SH[HEAD + n + i] = -(gs * (below + f * m) + gm) / (rc * rc);
    // The previous potential, before this one replaces it.
    SH[HEAD + 3u * n + i] = SH[HEAD + 2u * n + i];
    SH[HEAD + 2u * n + i] = -(gs * (below + f * m) + gm) / rc - gs * (1.0 - f) * m / rc;
    below += m;
  }
  // Inward: the shells outside each center.
  var outside = 0.0;
  for (var k = 0u; k < n; k++) {
    let i = n - 1u - k;
    let rc = sph_cell(i32(i), 0).r * P.dx;
    SH[HEAD + 2u * n + i] -= gs * outside;
    outside += SH[HEAD + i] / rc;
  }
}

@compute @workgroup_size(8, 8, 1)
fn field(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid.xy >= P.dims.xy)) { return; }
  let n = P.dims.x;
  let c = vec3<i32>(vec2<i32>(gid.xy), 0);
  let phi = SH[HEAD + 2u * n + gid.x];
  let old = SH[HEAD + 3u * n + gid.x];
  textureStore(PHI, c, vec4<f32>(phi, 0.0, 0.0, 0.0));
  textureStore(GOUT, c, vec4<f32>(SH[HEAD + n + gid.x], 0.0, 0.0, 1.5 * phi - 0.5 * old));
}

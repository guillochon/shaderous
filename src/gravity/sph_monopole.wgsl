// A spherical (r, theta) run's gravity (Problem.geometry 'spherical'): the monopole of the gas's
// own mass, and the point masses at the origin. Three entry points, dispatched in turn:
//   shells   one invocation per radial index: the mass of its shell of cells
//   enclose  one invocation: the mass inside each cell's radius by a prefix sum, its field
//            -G M(<r) / r^2, and the potential, sum over shells of -G m / max(r, r_shell)
//   field    one per cell: the field texture (g_r, 0, 0, the potential half a step ahead, as
//            gravity/field.wgsl has it) and the potential texture the diagnostics read
// SH holds, after a header of four (the point masses' G M, 1 if the gas's own gravity is on),
// four arrays of n: the shells' masses, their field, the potential and the previous potential.
//
// A 1D run's own gravity takes the same three passes (plans/one-dimensional.md), LINE saying
// its geometry: 0 the spherical monopole above; 1 a Cartesian run, each cell a slab of mass
// per area m, its field -2 pi G (the mass below - the mass above) and its potential the 1D Green's
// function's, 2 pi G sum m |x - x'|; 2 the same in a periodic box, of the mass less its mean,
// the field less its own mean (which is what makes the potential periodic); 3 an axisymmetric
// run's R, each cell a ring of mass per length m, its field -2 G (the mass inside R) / R and its
// potential 2 G (the mass inside ln R + sum over the rings outside of m ln R_ring).
#include "hydro/common.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> SH: array<f32>;
@group(0) @binding(3) var PHI: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var GOUT: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var<storage, read> clock: Clock;

const HEAD = 4u;
override LINE: u32 = 0u;

@compute @workgroup_size(64)
fn shells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P.dims.x;
  if (gid.x >= n) { return; }
  let i = i32(gid.x);
  var m = 0.0;
  var c = 0.0;
  if (LINE == 1u || LINE == 2u) {
    m = textureLoad(U1, vec3<i32>(i, 0, 0), 0).x * P.dx;
  } else if (LINE == 3u) {
    m = textureLoad(U1, vec3<i32>(i, 0, 0), 0).x * 6.2831853 * (f32(i) + 0.5) * P.dx * P.dx;
  } else {
    for (var j = 0; j < i32(P.dims.y); j++) {
      let k = kahan1(m, c, textureLoad(U1, vec3<i32>(i, j, 0), 0).x * sph_volume(i, j));
      m = k.x;
      c = k.y;
    }
  }
  SH[HEAD + gid.x] = m;
}

// A 1D Cartesian run's field and potential (LINE 1 and 2), into SH's arrays.
fn enclose_planar(n: u32, gs: f32) {
  var tot = 0.0;
  for (var i = 0u; i < n; i++) { tot += SH[HEAD + i]; }
  let mean = select(0.0, tot / f32(n), LINE == 2u);
  // The mass and its first moment (in cells) over the whole row, then below each cell.
  var at = 0.0;
  var bt = 0.0;
  for (var i = 0u; i < n; i++) {
    let m = SH[HEAD + i] - mean;
    at += m;
    bt += f32(i) * m;
  }
  let k = 6.2831853 * gs;
  var a = 0.0;
  var b = 0.0;
  var gsum = 0.0;
  for (var i = 0u; i < n; i++) {
    let m = SH[HEAD + i] - mean;
    let fi = f32(i);
    let above = at - a - m;
    let g = -k * (a - above);
    // sum over j of m_j |i - j| cells: below, i a - b; above, (bt - b - i m) - i (at - a - m).
    let phi = k * P.dx * ((fi * a - b) + (bt - b - fi * m) - fi * above);
    SH[HEAD + 3u * n + i] = SH[HEAD + 2u * n + i];
    SH[HEAD + n + i] = g;
    SH[HEAD + 2u * n + i] = phi;
    gsum += g;
    a += m;
    b += fi * m;
  }
  if (LINE == 2u) {
    // Periodic: no mean field, and the potential that goes with it.
    let gbar = gsum / f32(n);
    for (var i = 0u; i < n; i++) {
      SH[HEAD + n + i] -= gbar;
      SH[HEAD + 2u * n + i] += gbar * (f32(i) + 0.5) * P.dx;
    }
  }
}

// A 1D axisymmetric run's field and potential (LINE 3), into SH's arrays.
fn enclose_cylinder(n: u32, gs: f32) {
  var below = 0.0;
  for (var i = 0u; i < n; i++) {
    let m = SH[HEAD + i];
    let fi = f32(i);
    let rc = (fi + 0.5) * P.dx;
    // The ring's own share inside R_c: (R_c^2 - R_-^2) / (R_+^2 - R_-^2).
    let f = (fi + 0.25) / (2.0 * fi + 1.0);
    SH[HEAD + 3u * n + i] = SH[HEAD + 2u * n + i];
    SH[HEAD + n + i] = -2.0 * gs * (below + f * m) / rc;
    SH[HEAD + 2u * n + i] = 2.0 * gs * (below + m) * log(rc);
    below += m;
  }
  var outside = 0.0;
  for (var k = 0u; k < n; k++) {
    let i = n - 1u - k;
    SH[HEAD + 2u * n + i] += 2.0 * gs * outside;
    outside += SH[HEAD + i] * log((f32(i) + 0.5) * P.dx);
  }
}

@compute @workgroup_size(1)
fn enclose() {
  let n = P.dims.x;
  let gs = select(0.0, P.G, SH[1] != 0.0);
  let gm = SH[0];
  // (Comoving coordinates: the field of the comoving density over a, as gravity/rhs.wgsl and
  // gravity/field.wgsl have it.)
  let gl = gs / select(1.0, clock.a1, cosmo_on());
  if (LINE == 1u || LINE == 2u) {
    enclose_planar(n, gl);
    return;
  }
  if (LINE == 3u) {
    enclose_cylinder(n, gl);
    return;
  }
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
  // (Comoving coordinates: the field over a once more, as gravity/field.wgsl has it.)
  let g = SH[HEAD + n + gid.x] / select(1.0, clock.a1, cosmo_on() && LINE != 0u);
  textureStore(GOUT, c, vec4<f32>(g, 0.0, 0.0, 1.5 * phi - 0.5 * old));
}

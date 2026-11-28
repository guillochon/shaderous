// A spherical (r, theta) run's gravity on blocks (gravity/sph_monopole.wgsl on the uniform grid):
// the monopole of the gas's own mass, and the point masses at the origin. The shells are those of
// the finest level, whatever the mesh: a leaf's cells hand their mass to the finest shells they
// cover by volume, so that a cell's center at any level lies on a finest face (or, at the finest
// level, at a finest center) and finds the mass inside it there. Four entry points, in turn:
//   rows     one invocation per row of a leaf (its cells at one radial index): their mass
//   shells   one invocation per finest shell: the rows' shares of it, by volume, in leaf order
//   enclose  one invocation: the mass inside each finest face and center, and the sum over the
//            shells outside of m / r, each with its last values kept for the half step
//   field    every block's cells and two ghost layers: the field texture (g_r, 0, 0, the
//            potential half a step ahead, or now with per-block timesteps) and the potential
// SH: a header of four (the point masses' G M, 1 if the gas's own gravity is on), then the
// finest shells' masses (n), the mass inside each of 2n + 1 half indices h (2i a face, 2i + 1
// a center) and the outer sum at each, the same two as they were, and the rows (BS per leaf).
//
// A 1D run's own gravity takes the same passes (plans/one-dimensional.md; LINE as in
// gravity/sph_monopole.wgsl: 1 Cartesian slabs, 2 the same periodic, 3 axisymmetric rings):
// the finest cells' masses per area or per length, and at each half index the field (in the
// array of the mass inside) and the potential (in that of the outer sum). Its blocks' rows are
// all alike (amr/grid.wgsl is1d): the first is the block's.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> SH: array<f32>;
@group(0) @binding(3) var PHI: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var GOUT: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var<storage, read> clock: Clock;

// The potential's half-step extrapolation (off with per-block timesteps, whose blocks' halves
// differ: the potential now).
override HALF: bool = true;

const HEAD = 4u;
override LINE: u32 = 0u;
fn nf() -> u32 {
  return P.n;
}
fn o_mass() -> u32 { return HEAD; }
fn o_in() -> u32 { return HEAD + nf(); }
fn o_out() -> u32 { return HEAD + 3u * nf() + 1u; }
fn o_in_old() -> u32 { return HEAD + 5u * nf() + 2u; }
fn o_out_old() -> u32 { return HEAD + 7u * nf() + 3u; }
fn o_rows() -> u32 { return HEAD + 9u * nf() + 4u; }

@compute @workgroup_size(64)
fn rows(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x / u32(BS);
  if (k >= LIST[0]) { return; }
  let slot = i32(LIST[1u + k]);
  let i = i32(gid.x % u32(BS));
  var m = 0.0;
  var c = 0.0;
  if (LINE != 0u) {
    let dx = blk_dx(slot);
    let rho = textureLoad(U1, atl(blk_origin(slot), vec3<i32>(i, 0, 0)), 0).x;
    let gx = blk_coord(slot).x * BS + i;
    m = rho * select(dx, 6.2831853 * (f32(gx) + 0.5) * dx * dx, LINE == 3u);
    SH[o_rows() + gid.x] = m;
    return;
  }
  for (var j = 0; j < select(BS, 1, is1d()); j++) {
    let l = vec3<i32>(i, j, 0);
    let s = kahan1(m, c, textureLoad(U1, atl(blk_origin(slot), l), 0).x * blk_sph_volume(slot, l));
    m = s.x;
    c = s.y;
  }
  SH[o_rows() + gid.x] = m;
}

@compute @workgroup_size(64)
fn shells(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= nf()) { return; }
  let i = i32(gid.x);
  let q0 = sph_level(0);
  let v0 = sph_v3(i, q0);
  var m = 0.0;
  var c = 0.0;
  for (var k = 0u; k < LIST[0]; k++) {
    let slot = i32(LIST[1u + k]);
    let lc = i32(P.amr.z) - bt(slot, 0u);
    let il = i >> u32(lc);
    let row = il - blk_coord(slot).x * BS;
    if (row < 0 || row >= BS) { continue; }
    // The finest shell's share of the row's: its volume over the row's cells' (in finest cells).
    // (A 1D slab's, an equal share; a ring's, its area, as R.)
    var share = v0 / (sph_v3(il, sph_level(lc)) * f32(1 << u32(3 * lc)));
    if (LINE == 1u || LINE == 2u) { share = 1.0 / f32(1 << u32(lc)); }
    if (LINE == 3u) { share = (f32(i) + 0.5) / ((f32(il) + 0.5) * f32(1 << u32(2 * lc))); }
    let s = kahan1(m, c, SH[o_rows() + k * u32(BS) + u32(row)] * share);
    m = s.x;
    c = s.y;
  }
  SH[o_mass() + gid.x] = m;
}

// A 1D Cartesian run's field and potential at every half index (LINE 1 and 2): into the
// arrays of the mass inside and of the outer sum. Masses as at the finest cells' centers.
fn enclose_planar(n: u32, dxf: f32, gs: f32) {
  var tot = 0.0;
  for (var i = 0u; i < n; i++) { tot += SH[o_mass() + i]; }
  let mean = select(0.0, tot / f32(n), LINE == 2u);
  var at = 0.0;
  var bt = 0.0;
  for (var i = 0u; i < n; i++) {
    let m = SH[o_mass() + i] - mean;
    at += m;
    bt += f32(i) * m;
  }
  let k = 6.2831853 * gs;
  var a = 0.0;   // mass below cell i
  var b = 0.0;   // and its first moment, in cells
  var gsum = 0.0;
  for (var h = 0u; h <= 2u * n; h++) {
    let i = h / 2u;
    let x = 0.5 * f32(h);
    let mi = select(0.0, SH[o_mass() + min(i, n - 1u)] - mean, i < n);
    let below = a + select(0.0, 0.5 * mi, (h & 1u) == 1u);
    let g = -k * (2.0 * below - at);
    // sum m_j |x - (j + 1/2)| cells: below cell i, x a - (b + a / 2); from it up, (bt - b) + (at - a) / 2 - x (at - a).
    let phi = k * dxf * ((x * a - b - 0.5 * a) + (bt - b) + 0.5 * (at - a) - x * (at - a));
    SH[o_in() + h] = g;
    SH[o_out() + h] = phi;
    if ((h & 1u) == 1u) {
      gsum += g;
      a += mi;
      b += f32(i) * mi;
    }
  }
  if (LINE == 2u) {
    let gbar = gsum / f32(n);
    for (var h = 0u; h <= 2u * n; h++) {
      SH[o_in() + h] -= gbar;
      SH[o_out() + h] += gbar * 0.5 * f32(h) * dxf;
    }
  }
}

// A 1D axisymmetric run's at every half index (LINE 3), as gravity/sph_monopole.wgsl has it.
fn enclose_cylinder(n: u32, dxf: f32, gs: f32) {
  var below = 0.0;
  for (var h = 0u; h <= 2u * n; h++) {
    let i = h / 2u;
    let r = 0.5 * f32(h) * dxf;
    let center = (h & 1u) == 1u;
    let mi = select(0.0, SH[o_mass() + min(i, n - 1u)], i < n);
    let f = (f32(i) + 0.25) / (2.0 * f32(i) + 1.0);
    let inside = below + select(0.0, f * mi, center);
    SH[o_in() + h] = select(-2.0 * gs * inside / r, 0.0, r == 0.0);
    // The rings inside (a center's own whole) at ln r; those outside added below.
    let held = below + select(0.0, mi, center);
    SH[o_out() + h] = select(2.0 * gs * held * log(r), 0.0, held == 0.0);
    if (center) { below += mi; }
  }
  var outside = 0.0;
  for (var k = 0u; k <= 2u * n; k++) {
    let h = 2u * n - k;
    let i = h / 2u;
    if ((h & 1u) == 0u && i < n) {
      // A face: ring i is outside it.
      outside += SH[o_mass() + i] * log((f32(i) + 0.5) * dxf);
    }
    // (A center's own ring is held inside: the outside sum is of the rings past it.)
    SH[o_out() + h] += 2.0 * gs * outside;
  }
}

@compute @workgroup_size(1)
fn enclose() {
  let n = nf();
  let q0 = sph_level(0);
  let dxf = P.dx / f32(1u << P.amr.z);
  for (var h = 0u; h <= 2u * n; h++) {
    SH[o_in_old() + h] = SH[o_in() + h];
    SH[o_out_old() + h] = SH[o_out() + h];
  }
  // (Comoving coordinates: the field of the comoving density over a, as gravity/rhs.wgsl has it.)
  let gl = select(0.0, P.G, SH[1] != 0.0) / select(1.0, clock.a1, cosmo_on());
  if (LINE == 1u || LINE == 2u) {
    enclose_planar(n, dxf, gl);
    return;
  }
  if (LINE == 3u) {
    enclose_cylinder(n, dxf, gl);
    return;
  }
  // Outward: the mass inside each face, and inside each center (its own share inside r_c, as
  // gravity/sph_monopole.wgsl has it).
  var below = 0.0;
  var f = 0.5;
  if (q0.d != 0.0) {
    let e3 = q0.k.x * q0.k.x * q0.k.x;
    f = (1.0 - 1.0 / e3) / (e3 - 1.0 / e3);
  }
  for (var i = 0u; i < n; i++) {
    let m = SH[o_mass() + i];
    if (q0.d == 0.0) {
      let r = sph_cell_g(i32(i), 0, q0).r;
      f = (1.5 * r * r - 0.75 * r + 0.125) / (3.0 * r * r + 0.25);
    }
    SH[o_in() + 2u * i] = below;
    SH[o_in() + 2u * i + 1u] = below + f * m;
    below += m;
  }
  SH[o_in() + 2u * n] = below;
  // Inward: the sum of m / r over the shells outside each face, and each center's own outer share.
  var outside = 0.0;
  SH[o_out() + 2u * n] = 0.0;
  for (var k = 0u; k < n; k++) {
    let i = n - 1u - k;
    let m = SH[o_mass() + i];
    let r = sph_cell_g(i32(i), 0, q0).r;
    var fi = f;
    if (q0.d == 0.0) { fi = (1.5 * r * r - 0.75 * r + 0.125) / (3.0 * r * r + 0.25); }
    let rc = r * dxf;
    SH[o_out() + 2u * i + 1u] = outside + (1.0 - fi) * m / rc;
    outside += m / rc;
    SH[o_out() + 2u * i] = outside;
  }
}

@compute @workgroup_size(8, 8, 4)
fn field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, 2);
  if (!inv.ok) { return; }
  let l = inv.l;
  let out = vec3<i32>(l < vec3<i32>(0)) + vec3<i32>(l >= vec3<i32>(BS));
  if (out.x + out.y + out.z > 1) { return; }
  let n = i32(nf());
  let lc = i32(P.amr.z) - bt(inv.slot, 0u);
  let gx = blk_coord(inv.slot).x * BS + l.x;
  // The cell's center as a half index of the finest shells: on a face, or a center at the finest level.
  let h = u32(clamp((2 * gx + 1) << u32(lc), 0, 2 * n));
  // (A ghost below r_min may have no radius of its own on a grid uniform in r: a quarter cell.)
  let r = max(sph_cell_g(gx, 0, sph_level(lc)).r, 0.25) * blk_dx(inv.slot);
  let gs = select(0.0, P.G, SH[1] != 0.0);
  let gm = SH[0];
  let c0 = atl(blk_origin(inv.slot), l);
  if (LINE != 0u) {
    // A 1D run: the field and potential at the half index, as enclose left them.
    // (Comoving coordinates: the field over a once more, as gravity/field.wgsl has it.)
    let g1 = SH[o_in() + h] / select(1.0, clock.a1, cosmo_on());
    let p1 = SH[o_out() + h];
    let w1 = select(p1, 1.5 * p1 - 0.5 * SH[o_out_old() + h], HALF);
    textureStore(PHI, c0, vec4<f32>(p1, 0.0, 0.0, 0.0));
    textureStore(GOUT, c0, vec4<f32>(g1, 0.0, 0.0, w1));
    return;
  }
  let inside = gs * SH[o_in() + h] + gm;
  let phi = -inside / r - gs * SH[o_out() + h];
  var w = phi;
  if (HALF) {
    let old = -(gs * SH[o_in_old() + h] + gm) / r - gs * SH[o_out_old() + h];
    w = 1.5 * phi - 0.5 * old;
  }
  let c = atl(blk_origin(inv.slot), l);
  textureStore(PHI, c, vec4<f32>(phi, 0.0, 0.0, 0.0));
  textureStore(GOUT, c, vec4<f32>(-inside / (r * r), 0.0, 0.0, w));
}

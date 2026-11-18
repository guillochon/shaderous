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
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> SH: array<f32>;
@group(0) @binding(3) var PHI: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var GOUT: texture_storage_3d<rgba32float, write>;

// The potential's half-step extrapolation (off with per-block timesteps, whose blocks' halves
// differ: the potential now).
override HALF: bool = true;

const HEAD = 4u;
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
  for (var j = 0; j < BS; j++) {
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
    let share = v0 / (sph_v3(il, sph_level(lc)) * f32(1 << u32(3 * lc)));
    let s = kahan1(m, c, SH[o_rows() + k * u32(BS) + u32(row)] * share);
    m = s.x;
    c = s.y;
  }
  SH[o_mass() + gid.x] = m;
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

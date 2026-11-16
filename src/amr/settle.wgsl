// Settling stars into the mesh's own hydrostatic balance, on the GPU (problems/settle.ts holds
// the method and the CPU version it mirrors; amr/amr_sim.ts, settleOnGpu, drives the rounds).
//
// Each star's density is made to satisfy h(rho) + phi = C, with phi the grid's potential less
// the other stars' pull and C fixed by the star's mass. The host finds C from sums over the
// star's cells, which these passes make per block; it adds the blocks' sums in double precision.
//
//   measure, MODE 0: per block, each star's mass (cells denser than the cut) and the densest cell
//   measure, MODE 1: per block, the least potential among star S's cells
//   measure, MODE 2: per block, star S's mass at 16 trial values of C
//   measure, MODE 3: per block, each star's mass and its moments m x, m y, m z (its center of
//                    mass, where the others' pull on it is taken from)
//   apply:           a round's new density, from the current state into the other parity;
//                    per block, the largest change
//   finish:          every star cell's whole state from its settled density (cold gas at the
//                    star's velocity), or the ambient gas; into the other parity
//
// One workgroup per block of the leaf list, in rows (amr/list.wgsl, flat_row). A cell belongs to
// the star it is fewest radii from, within that star's reach, as on the CPU.
#include "hydro/common.wgsl"
#include <eos>
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

struct SStar {
  c: vec4<f32>, // center (relative to the box center), radius
  r: vec4<f32>, // reach, mass, passive scalar, barotrope (0 polytrope, 1 cold degenerate gas)
  b: vec4<f32>, // barotrope: polytrope (n, K, (n + 1) K, 0); degenerate (rho_d, A, and the cold Coulomb term's c1 and rho0: 0 and 1 for none)
  v: vec4<f32>, // velocity
  k: vec4<f32>, // C as two floats (C = k.x + k.y)
  q: vec4<f32>, // center of mass, where its pull on the other stars comes from
}
struct Settle {
  h: vec4<u32>,        // number of stars, star S of measure modes 1 and 2
  f: vec4<f32>,        // density cut, ambient density, ambient pressure, over-relaxation
  g: vec4<f32>,        // G
  stars: array<SStar, 4>,
  cand: array<vec4<f32>, 8>, // the 16 trial values of C of MODE 2, as (hi, lo) pairs
}

@group(0) @binding(10) var<uniform> S: Settle;
@group(0) @binding(2) var N1: texture_3d<f32>;
@group(0) @binding(3) var N2: texture_3d<f32>;
@group(0) @binding(4) var PHI: texture_3d<f32>;
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(11) var<storage, read_write> OUT: array<f32>;

override MODE: u32 = 0u;
const WG = 64u;
const NC = 16u;

var<workgroup> red: array<array<f32, 16>, 64>;

fn owner(x: vec3<f32>) -> i32 {
  var best = -1;
  var bd = 3.0e38;
  for (var i = 0u; i < S.h.x; i++) {
    let s = S.stars[i];
    let d = length(x - s.c.xyz);
    if (d / s.c.w < bd && d <= s.r.x) {
      bd = d / s.c.w;
      best = i32(i);
    }
  }
  return best;
}

// The other stars' pull on star i, undone: outside a spherical star it is that of a point of its mass.
fn pull(i: u32, x: vec3<f32>) -> f32 {
  var p = 0.0;
  for (var j = 0u; j < S.h.x; j++) {
    if (j == i) { continue; }
    p += S.g.x * S.stars[j].r.y / length(x - S.stars[j].q.xyz);
  }
  return p;
}

fn rho_of_h(i: u32, h: f32) -> f32 {
  if (!(h > 0.0)) { return 0.0; }
  let s = S.stars[i];
  if (s.r.w == 0.0) { return pow(h / s.b.z, s.b.x); }
  // Cold degenerate gas: h = (8 A / rho_d) (sqrt(1 + x^2) - 1), x^3 = rho / rho_d.
  let u = h * s.b.x / (8.0 * s.b.y);
  let x = sqrt(u * (2.0 + u));
  var rho = s.b.x * x * x * x;
  if (s.b.z == 0.0) { return rho; }
  // With the cold Coulomb term the enthalpy has no closed inverse: Newton's method from the
  // electrons' answer, with dh/drho = c^2 / rho (problems/white_dwarf.ts, degenerateBarotrope).
  for (var it = 0; it < 8; it++) {
    let x2 = pow(rho / s.b.x, 2.0 / 3.0);
    let c = settle_coulomb(i, rho);
    let hr = 8.0 * s.b.y / s.b.x * (x2 / (sqrt(1.0 + x2) + 1.0)) + c.z;
    let c2 = 8.0 * s.b.y * x2 * x2 * sqrt(x2) / (3.0 * rho * sqrt(1.0 + x2)) + c.y;
    let next = max(rho + (h - hr) * rho / c2, 0.25 * rho);
    let done = abs(next - rho) <= 1e-6 * rho;
    rho = next;
    if (done) { break; }
  }
  return rho;
}

// The cold part of the ions' Coulomb energy at density rho (eos/helmholtz.ts): its pressure,
// dp/drho and enthalpy. Per unit mass the energy is c1 rho^(1/3) w, w = rho / (rho0 + rho).
fn settle_coulomb(i: u32, rho: f32) -> vec3<f32> {
  let s = S.stars[i];
  let w = rho / (s.b.w + rho);
  let e = s.b.z * pow(rho, 1.0 / 3.0) * w;
  return vec3<f32>(rho * e * (4.0 / 3.0 - w), e * (4.0 / 3.0 * (4.0 / 3.0 - w) + (1.0 - w) * (4.0 / 3.0 - 2.0 * w)), e * (7.0 / 3.0 - w));
}

// f(x) of the cold degenerate pressure A f(x) (eos/eos.ts, degFG), a series below x = 1/2.
fn settle_deg_f(x: f32) -> f32 {
  if (x < 0.5) {
    let x2 = x * x;
    var xp = x2 * x2 * x;
    var c = 1.0;
    var f = 0.0;
    for (var k = 0; k < 12; k++) {
      f += 8.0 * c * xp / f32(5 + 2 * k);
      c *= -f32(2 * k + 1) / f32(2 * k + 2);
      xp *= x2;
    }
    return f;
  }
  return x * (2.0 * x * x - 3.0) * sqrt(1.0 + x * x) + 3.0 * asinh(x);
}

fn baro_pressure(i: u32, rho: f32) -> f32 {
  let s = S.stars[i];
  if (s.r.w == 0.0) { return s.b.y * pow(rho, 1.0 + 1.0 / s.b.x); }
  var p = s.b.y * settle_deg_f(pow(rho / s.b.x, 1.0 / 3.0));
  if (s.b.z != 0.0 && rho > 0.0) { p += settle_coulomb(i, rho).x; }
  return p;
}

fn cells_per_block() -> u32 {
  return select(u32(BS * BS * BS), u32(BS * BS), is2d());
}

fn cell_l(i: u32) -> vec3<i32> {
  let b = u32(BS);
  return vec3<i32>(i32(i % b), i32((i / b) % b), i32(i / (b * b)));
}

// The block of this workgroup, or -1 past the end of the list.
fn block_of(wid: vec3<u32>) -> i32 {
  let row = flat_row(wid);
  if (row >= LIST[0]) { return -1; }
  return i32(LIST[1u + row]);
}

// Reduces red[*][0..n) over the workgroup into red[0][0..n). `use_max`: the largest, not the sum.
fn reduce(lix: u32, n: u32, use_max: bool) {
  for (var w = WG / 2u; w > 0u; w /= 2u) {
    workgroupBarrier();
    if (lix < w) {
      for (var k = 0u; k < n; k++) {
        let a = red[lix][k];
        let b = red[lix + w][k];
        red[lix][k] = select(a + b, max(a, b), use_max);
      }
    }
  }
  workgroupBarrier();
}

@compute @workgroup_size(64)
fn measure(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let slot = block_of(wid);
  let row = flat_row(wid);
  var acc: array<f32, 16>;
  // MODE 1 looks for a least value, the others add up from zero.
  for (var k = 0u; k < NC; k++) { acc[k] = select(0.0, 3.0e38, MODE == 1u); }
  if (slot >= 0) {
    g_dx = blk_dx(slot);
    let vol = g_dx * g_dx * g_dx;
    let o = blk_origin(slot);
    let star = S.h.y;
    for (var i = lix; i < cells_per_block(); i += WG) {
      let l = cell_l(i);
      let x = blk_pos(slot, l);
      let who = owner(x);
      if (who < 0) { continue; }
      let c = atl(o, l);
      if (MODE == 0u) {
        let rho = textureLoad(N1, c, 0).x;
        if (rho > S.f.x) { acc[u32(who)] += rho * vol; }
        acc[4] = max(acc[4], rho);
        continue;
      }
      if (MODE == 3u) {
        let rho = textureLoad(N1, c, 0).x;
        if (rho > S.f.x) {
          let m = rho * vol;
          let o4 = 4u * u32(who);
          acc[o4] += m;
          acc[o4 + 1u] += m * x.x;
          acc[o4 + 2u] += m * x.y;
          acc[o4 + 3u] += m * x.z;
        }
        continue;
      }
      if (u32(who) != star) { continue; }
      let ph = textureLoad(PHI, c, 0).x + pull(star, x);
      if (MODE == 1u) {
        acc[0] = min(acc[0], ph);
        continue;
      }
      for (var k = 0u; k < NC; k++) {
        let q = S.cand[k / 2u];
        let C = select(q.zw, q.xy, k % 2u == 0u);
        let r = rho_of_h(star, (C.x - ph) + C.y);
        if (r > S.f.x) { acc[k] += r * vol; }
      }
    }
  }
  for (var k = 0u; k < NC; k++) { red[lix][k] = acc[k]; }
  if (MODE == 1u) {
    // The least: negate, take the largest, negate back.
    red[lix][0] = -acc[0];
    reduce(lix, 1u, true);
    if (lix == 0u && slot >= 0) { OUT[row * NC] = -red[0][0]; }
    return;
  }
  if (MODE == 0u) {
    let peak = acc[4];
    red[lix][4] = 0.0;
    reduce(lix, 4u, false);
    if (lix == 0u && slot >= 0) { for (var k = 0u; k < 4u; k++) { OUT[row * NC + k] = red[0][k]; } }
    workgroupBarrier();
    red[lix][0] = peak;
    reduce(lix, 1u, true);
    if (lix == 0u && slot >= 0) { OUT[row * NC + 4u] = red[0][0]; }
    return;
  }
  reduce(lix, NC, false);
  if (lix == 0u && slot >= 0) { for (var k = 0u; k < NC; k++) { OUT[row * NC + k] = red[0][k]; } }
}

@compute @workgroup_size(64)
fn apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let slot = block_of(wid);
  let row = flat_row(wid);
  var change = 0.0;
  if (slot >= 0) {
    g_dx = blk_dx(slot);
    let o = blk_origin(slot);
    for (var i = lix; i < cells_per_block(); i += WG) {
      let l = cell_l(i);
      let c = atl(o, l);
      var u1 = textureLoad(N1, c, 0);
      let u2 = textureLoad(N2, c, 0);
      let x = blk_pos(slot, l);
      let who = owner(x);
      if (who >= 0) {
        let s = S.stars[u32(who)];
        let ph = textureLoad(PHI, c, 0).x + pull(u32(who), x);
        var r = rho_of_h(u32(who), (s.k.x - ph) + s.k.y);
        if (!(r > S.f.x)) { r = S.f.y; }
        change = max(change, abs(r - u1.x));
        // Over-relaxed: the rounds converge geometrically, so going further than asked gets there sooner.
        let next = u1.x + S.f.w * (r - u1.x);
        u1.x = select(S.f.y, next, next > S.f.x);
      }
      textureStore(O1, c, u1);
      textureStore(O2, c, u2);
    }
  }
  red[lix][0] = change;
  reduce(lix, 1u, true);
  if (lix == 0u && slot >= 0) { OUT[row * NC] = red[0][0]; }
}

@compute @workgroup_size(64)
fn finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let slot = block_of(wid);
  if (slot < 0) { return; }
  g_dx = blk_dx(slot);
  let o = blk_origin(slot);
  for (var i = lix; i < cells_per_block(); i += WG) {
    let l = cell_l(i);
    let c = atl(o, l);
    var u1 = textureLoad(N1, c, 0);
    var u2 = textureLoad(N2, c, 0);
    let x = blk_pos(slot, l);
    let who = owner(x);
    if (who >= 0) {
      comp_use(comp_tex(c));
      let r = u1.x;
      if (r > S.f.x) {
        let s = S.stars[u32(who)];
        let eint = eos_eint_from_p(r, baro_pressure(u32(who), r));
        let v = s.v.xyz;
        u1 = vec4<f32>(r, r * v);
        u2 = vec4<f32>(eint + 0.5 * r * dot(v, v), eint, r * s.r.z, u2.w);
      } else {
        // Ambient gas at rest, in rough pressure balance with the stars' outermost layers.
        let e = eos_eint_from_p(S.f.y, S.f.z);
        u1 = vec4<f32>(S.f.y, 0.0, 0.0, 0.0);
        u2 = vec4<f32>(e, e, 0.0, u2.w);
      }
    }
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
  }
}

// Copies the leaf cells' state into the other parity: apply and finish write the parity the
// state is not in, and this brings it back, so that the state stays in its parity (the
// magnetic field's textures go by it too).
@compute @workgroup_size(64)
fn copy_back(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let slot = block_of(wid);
  if (slot < 0) { return; }
  let o = blk_origin(slot);
  for (var i = lix; i < cells_per_block(); i += WG) {
    let c = atl(o, cell_l(i));
    textureStore(O1, c, textureLoad(N1, c, 0));
    textureStore(O2, c, textureLoad(N2, c, 0));
  }
}

// ANEOS materials for planetary impacts: rock, iron and water, tabulated (eos/aneos.ts, which
// says what the table holds; tools/eos/aneos_table.py builds it). On a grid uniform in
// x = log10 rho (g/cm^3) and y = log10 T (K), with slopes for bicubic Hermite interpolation:
//   psi = asinh(p / AN_PS)      the pressure
//   lam = ln(e - E_c(rho))      the energy per gram above the cold curve
// The hydro holds the energy (or the pressure), so the temperature comes first: lam (or psi)
// rises along y, so the node rows on either side of the answer are found by bisection (8 steps
// for the 173 rows), and the place between them by Newton's method on the cubic through them.
// The other quantity and the sound speed are then read at that place. Inside, everything is in
// cgs and kelvin; the code's units are converted on the way in and out.
//
// Two materials: the table then holds the mixture at five mass fractions of the second
// (eos/aneos.ts buildPair), stacked along the rows, and a cell reads the two on either side of
// its own fraction, blended texel by texel. The fraction is g_comp.y (species/mixture.ts).
//
// P.eos = (g/cm^3, erg/g, kelvin per code unit of density, specific energy, temperature; unused)
// Float64 mirror: src/eos/aneos.ts.
const EOS_HAS_ENERGY = true;
// E - KE below the cold curve is no reason to rebuild E (as eos/degenerate.wgsl).
const EOS_KEEP_TOTAL = true;

#include "eos/aneos_consts.wgsl"

@group(0) @binding(41) var AN_TAB: texture_2d<f32>;

const AN_LN10: f32 = 2.30258509;
// The least c^2, (cm/s)^2: inside the vapor dome it nearly vanishes. And the most: 100 km/s,
// several times any sound speed of the tables.
const AN_C2_MIN: f32 = 1e7;
const AN_C2_MAX: f32 = 1e14;
// The most (dp/dT) / (rho de/dT), the Grueneisen parameter, may be: it is 1 to 3 in the
// condensed phases and below 1 in the gas. Where the energy's slope along T is next to nothing
// (the lowest isotherm of a cold, pressed solid, whose heat the table does not resolve) the
// ratio of the two slopes is noise, and a sound speed made from it, of 1e3 km/s, wrecked the
// fluxes of the cells it was in.
const AN_GRUN_MAX: f32 = 5.0;
// Newton steps inside a cell of the table (ANEOS_NEWTON in eos/aneos.ts).
const AN_NEWTON: i32 = 8;

// For the few uses outside the module (the entropy the view shows).
fn eos_gamma() -> f32 { return 5.0 / 3.0; }
fn eos_gm1() -> f32 { return 2.0 / 3.0; }

// Which rows a cell reads: the tables on either side of its mass fraction, and the weight of the upper.
var<private> g_an_lo: i32 = 0;
var<private> g_an_hi: i32 = 0;
var<private> g_an_w: f32 = 0.0;
fn an_view() {
  g_an_lo = 0;
  g_an_hi = 0;
  g_an_w = 0.0;
  if (COMP) {
    let levels = i32(textureDimensions(AN_TAB).y) / AN_NR;
    if (levels > 1) {
      let f = clamp(g_comp.y, 0.0, 1.0) * f32(levels - 1);
      let l = min(i32(floor(f)), levels - 2);
      g_an_lo = l * AN_NR;
      g_an_hi = g_an_lo + AN_NR;
      g_an_w = f - f32(l);
    }
  }
}
fn an_tex(col: i32, row: i32) -> vec4<f32> {
  let a = textureLoad(AN_TAB, vec2<i32>(col, g_an_lo + row), 0);
  if (g_an_w > 0.0) { return mix(a, textureLoad(AN_TAB, vec2<i32>(col, g_an_hi + row), 0), g_an_w); }
  return a;
}

fn an_hb(t: f32) -> vec4<f32> { return vec4<f32>(2.0*t*t*t - 3.0*t*t + 1.0, t*t*t - 2.0*t*t + t, -2.0*t*t*t + 3.0*t*t, t*t*t - t*t); }
fn an_hb1(t: f32) -> vec4<f32> { return vec4<f32>(6.0*t*t - 6.0*t, 3.0*t*t - 4.0*t + 1.0, -6.0*t*t + 6.0*t, 3.0*t*t - 2.0*t); }

// Where a density (g/cm^3) lies along x: the row below, the place between rows, and rho over
// the table's least density where it is below that (1 otherwise).
struct AnX { i: i32, u: f32, thin: f32 }
fn an_x(rho: f32) -> AnX {
  let x = (log(rho) / AN_LN10 - AN_X0) / AN_D;
  var thin = 1.0;
  if (x < 0.0) { thin = exp(x * AN_D * AN_LN10); }
  let xc = clamp(x, 0.0, f32(AN_NR) - 1.0001);
  let i = i32(floor(xc));
  return AnX(i, xc - f32(i), thin);
}
// A column of the table at a place along x: (value, slope along y per grid step).
fn an_column(col: i32, a: AnX) -> vec2<f32> {
  let A = an_hb(a.u);
  let t0 = an_tex(col, a.i);
  let t1 = an_tex(col, a.i + 1);
  return vec2<f32>(A.x * t0.x + A.y * t0.y + A.z * t1.x + A.w * t1.y, A.x * t0.z + A.y * t0.w + A.z * t1.z + A.w * t1.w);
}
// The place along y (grid steps) at which the block of columns from `off` (0: psi, AN_NT: lam)
// has the value `goal`.
fn an_solve(off: i32, a: AnX, goal: f32) -> f32 {
  var lo = 0;
  var hi = AN_NT - 1;
  var glo = an_column(off, a);
  if (!(goal > glo.x)) { return 0.0; }
  var ghi = an_column(off + AN_NT - 1, a);
  if (!(goal < ghi.x)) { return f32(AN_NT - 1); }
  for (var k = 0; k < 8; k++) {
    if (hi - lo <= 1) { break; }
    let mid = (lo + hi) / 2;
    let g = an_column(off + mid, a);
    if (g.x <= goal) { lo = mid; glo = g; } else { hi = mid; ghi = g; }
  }
  // Newton's method on the cubic through the two rows, kept in its bracket by false position
  // (the Illinois rule), as solveY in eos/aneos.ts.
  var a0 = 0.0;
  var a1 = 1.0;
  var f0 = glo.x - goal;
  var f1 = ghi.x - goal;
  var side = 0;
  var t = 0.0;
  if (ghi.x > glo.x) { t = (goal - glo.x) / (ghi.x - glo.x); }
  for (var k = 0; k < AN_NEWTON; k++) {
    let B = an_hb(t);
    let B1 = an_hb1(t);
    let f = B.x * glo.x + B.y * glo.y + B.z * ghi.x + B.w * ghi.y - goal;
    if (f > 0.0) {
      a1 = t;
      f1 = f;
      if (side == 1) { f0 *= 0.5; }
      side = 1;
    } else {
      a0 = t;
      f0 = f;
      if (side == -1) { f1 *= 0.5; }
      side = -1;
    }
    let df = B1.x * glo.x + B1.y * glo.y + B1.z * ghi.x + B1.w * ghi.y;
    var next = t - f / df;
    if (!(next > a0 && next < a1)) {
      next = 0.5 * (a0 + a1);
      if (f1 > f0) { next = a0 - f0 * (a1 - a0) / (f1 - f0); }
    }
    t = next;
  }
  return f32(lo) + clamp(t, 0.0, 1.0);
}

// The cold curve at a place along x: (E_c, dE_c/dln rho), erg/g. Thinner than the table it is
// flat, as a gas's.
fn an_cold(a: AnX) -> vec2<f32> {
  let c0 = an_tex(2 * AN_NT, a.i);
  if (a.thin < 1.0) { return vec2<f32>(c0.x, 0.0); }
  let c1 = an_tex(2 * AN_NT, a.i + 1);
  let A = an_hb(a.u);
  let A1 = an_hb1(a.u);
  return vec2<f32>(A.x * c0.x + A.y * c0.y + A.z * c1.x + A.w * c1.y, (A1.x * c0.x + A1.y * c0.y + A1.z * c1.x + A1.w * c1.y) / (AN_D * AN_LN10));
}

// One block of the table at a place: (value, d/dln rho, d/dln T). Node by node, with no array
// (as helm_comb in eos/helmholtz.wgsl, and for its reason).
fn an_node(q: vec4<f32>, a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
  return a0 * b0 * q.x + a1 * b0 * q.y + a0 * b1 * q.z + a1 * b1 * q.w;
}
fn an_block(off: i32, a: AnX, j: i32, t: f32) -> vec3<f32> {
  let c00 = an_tex(off + j, a.i);
  let c01 = an_tex(off + j + 1, a.i);
  let c10 = an_tex(off + j, a.i + 1);
  let c11 = an_tex(off + j + 1, a.i + 1);
  let A = an_hb(a.u);
  let A1 = an_hb1(a.u);
  let B = an_hb(t);
  let B1 = an_hb1(t);
  let s = 1.0 / (AN_D * AN_LN10);
  return vec3<f32>(
    an_node(c00, A.x, A.y, B.x, B.y) + an_node(c01, A.x, A.y, B.z, B.w) + an_node(c10, A.z, A.w, B.x, B.y) + an_node(c11, A.z, A.w, B.z, B.w),
    (an_node(c00, A1.x, A1.y, B.x, B.y) + an_node(c01, A1.x, A1.y, B.z, B.w) + an_node(c10, A1.z, A1.w, B.x, B.y) + an_node(c11, A1.z, A1.w, B.z, B.w)) * s,
    (an_node(c00, A.x, A.y, B1.x, B1.y) + an_node(c01, A.x, A.y, B1.z, B1.w) + an_node(c10, A.z, A.w, B1.x, B1.y) + an_node(c11, A.z, A.w, B1.z, B1.w)) * s,
  );
}

// The state at a place of the table, cgs: (pressure, energy per gram, temperature, c^2).
// c^2 = (dp/drho)_T + (dp/dT)_rho^2 T / (rho^2 c_v), every term of it positive. (The speed of the
// system the hydro solves is (dp/drho)_e + (p / rho^2) (dp/de)_rho, which is the same where the
// table is consistent with a free energy; it holds the energy's slope along the density, which
// in the vapor dome is the latent heat's and large, times a ratio of two slopes along T that in
// cold gas are both next to nothing. Cold dust at 100 K came out at 26 km/s and set the timestep
// of a whole run.)
fn an_state(a: AnX, y: f32, rho: f32) -> vec4<f32> {
  let j = min(i32(floor(y)), AN_NT - 2);
  let t = y - f32(j);
  let ps = an_block(0, a, j, t);
  let lm = an_block(AN_NT, a, j, t);
  let ec = an_cold(a);
  // sinh and cosh of psi from one exponential. The tables' pressures are positive: so is what
  // is read between their nodes.
  let ex = exp(max(ps.x, 0.0));
  let sh = 0.5 * (ex - 1.0 / ex);
  let ch = 0.5 * (ex + 1.0 / ex);
  let p = AN_PS * sh * a.thin;
  let eth = exp(lm.x);
  let pT = AN_PS * ch * ps.z * a.thin;
  var pR = AN_PS * ch * ps.y;
  if (a.thin < 1.0) { pR = p; }
  let eT = eth * lm.z;
  var c2 = max(pR, 0.0) / rho;
  if (eT > 0.0 && pT > 0.0) { c2 += clamp(pT / (eT * rho), 0.0, AN_GRUN_MAX) * pT / rho; }
  c2 = clamp(c2, AN_C2_MIN, AN_C2_MAX);
  return vec4<f32>(p, ec.x + eth, exp((AN_Y0 + AN_D * y) * AN_LN10), c2);
}

// By the energy: rho and eint in code units.
fn an_from_e(rho: f32, eint: f32) -> vec4<f32> {
  an_view();
  let rc = rho * P.eos.x;
  let a = an_x(rc);
  let eth = eint / rho * P.eos.y - an_cold(a).x;
  var y = 0.0;
  if (eth > 0.0) { y = an_solve(AN_NT, a, log(eth)); }
  return an_state(a, y, rc);
}
// By the pressure.
fn an_from_p(rho: f32, p: f32) -> vec4<f32> {
  an_view();
  let rc = rho * P.eos.x;
  let a = an_x(rc);
  let y = an_solve(0, a, asinh(p * P.eos.x * P.eos.y / (AN_PS * a.thin)));
  return an_state(a, y, rc);
}
fn an_p(s: vec4<f32>) -> f32 { return s.x / (P.eos.x * P.eos.y); }
fn an_eint(s: vec4<f32>, rho: f32) -> f32 { return rho * s.y / P.eos.y; }
fn an_cs(s: vec4<f32>) -> f32 { return sqrt(s.w / P.eos.y); }

fn eos_pressure(rho: f32, eint: f32) -> f32 { return an_p(an_from_e(rho, eint)); }
fn eos_temperature(rho: f32, eint: f32) -> f32 { return an_from_e(rho, eint).z / P.eos.z; }
fn eos_sound_speed(rho: f32, p: f32) -> f32 { return an_cs(an_from_p(rho, p)); }
fn eos_eint_from_p(rho: f32, p: f32) -> f32 { return an_eint(an_from_p(rho, p), rho); }
// A face state's sound speed and internal energy density, from one solve (hydro/flux.wgsl).
fn eos_face(rho: f32, p: f32) -> vec2<f32> {
  let s = an_from_p(rho, p);
  return vec2<f32>(an_cs(s), an_eint(s, rho));
}
// A cell's pressure and sound speed from its internal energy density, from one solve (hydro/prim.wgsl).
fn eos_prim(rho: f32, eint: f32) -> vec2<f32> {
  let s = an_from_e(rho, eint);
  return vec2<f32>(an_p(s), an_cs(s));
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  an_view();
  let rc = rho * P.eos.x;
  let y = clamp((log(max(T * P.eos.z, 1e-30)) / AN_LN10 - AN_Y0) / AN_D, 0.0, f32(AN_NT - 1));
  return an_eint(an_state(an_x(rc), y, rc), rho);
}
// The least energy a state may have: that of the table's lowest isotherm. A state below it is
// read as that isotherm's, whose pressure then gives a face more energy than the cell holds:
// the cell would hand on energy it does not have, step after step (a planet's surface, where gas
// expanding into near-vacuum lands on the isotherm, lost all it had in half a dynamical time).
fn eos_eint_floor(rho: f32) -> f32 {
  an_view();
  let a = an_x(rho * P.eos.x);
  return rho * (an_cold(a).x + exp(an_column(AN_NT, a).x)) / P.eos.y;
}

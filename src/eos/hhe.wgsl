// Hydrogen and helium in chemical equilibrium among H2, H, H+, He, He+, He++ and electrons: a
// tabulated equation of state from molecular clouds to stellar envelopes. One free energy
// (src/eos/hhe_model.ts: an ideal mixture, H2 with its rotation and vibration, the dissociation
// and ionization energies counted from the molecule's ground state) on a grid in log10 rho
// (1e-20 to 1e3 g/cm^3) and log10 T (10 to 1e8 K), made by tools/eos/make_hhe_table.mjs.
// With h = F m_u / (k T) per atomic mass unit,
//   p = rho (k T / m_u) dh/dln rho + a T^4 / 3
//   e = -rho (k T / m_u) dh/dln T + a T^4
// so the gas softens as it dissociates and ionizes (Gamma_1 falls to 1.1) and the energy that
// takes is part of e, which is zero at T = 0. The radiation term is analytic and optional.
//
// The table is for one composition: the run's (solar or hydrogen), set when it is bound
// (eos/hhe.ts). A composition that varies from cell to cell (eos/comp_on) is not followed.
// The lookup and the solve for the temperature are eos/table.wgsl, shared with
// eos/helmholtz.wgsl; inside, everything is in cgs, kelvin and per unit mass.
//
// Where it holds: below about 0.1 g/cm^3 at any temperature of the table, and above that where
// the gas is hot enough to be ionized (a star's interior). There are no degenerate electrons
// and no Coulomb terms, and pressure ionization is a stand-in (hhe_model.ts): cool gas denser
// than about 1 g/cm^3 is outside it, and there the pressure need not rise with temperature.
// Beyond the table's ends in T the gas keeps the composition it has at the end; beyond them in
// density, the ionization it has there.
//
// P.eos = (g/cm^3, erg/g and kelvin per code unit of density, specific energy and temperature;
//          the multiplier of the radiation term: 1, or 0 for gas alone)
// Float64 mirror: src/eos/hhe.ts.
const EOS_HAS_ENERGY = true;
// Where E - KE cannot be trusted, rebuild E from the advected e_int (hydro/dual_energy.wgsl).
const EOS_KEEP_TOTAL = false;

#include "eos/hhe_consts.wgsl"

const TAB_DIFF = true;
const TAB_LNT_MIN: f32 = 2.3025851;    // ln 10
const TAB_LNT_MAX: f32 = 18.420681;    // ln 1e8
#include "eos/table.wgsl"

// k / m_u, erg/g/K
const HH_KMU: f32 = 8.3144626e7;
// a (1e4 K)^4, erg/cm^3
const HH_A4: f32 = 75.65723;
const HH_LN1E4: f32 = 9.2103404;
// The sound speed squared is at least this share of p / rho (outside where the model holds,
// dp/drho at fixed T can be negative).
const HH_C2_MIN: f32 = 0.01;

// The gas's adiabatic index, for the few uses outside the module (entropy shown in the view).
fn eos_gamma() -> f32 { return 5.0 / 3.0; }
fn eos_gm1() -> f32 { return 2.0 / 3.0; }

// a T^4 / rho (erg/g), times the multiplier; in factors that stay inside single precision in
// any order a compiler takes them (eos/helmholtz.wgsl, helm_rad).
fn hhe_rad(T: f32, rho: f32) -> f32 {
  let t4 = T * 1e-4;
  return P.eos.w * HH_A4 * (t4 * t4) * (t4 * t4) / rho;
}

struct HheThermo { e: f32, pr: f32, de: f32, dpr: f32, dprho: f32 }

// Per unit mass at density rho (g/cm^3) and ln T:
//   e (erg/g), p / rho, de/dln T, d(p/rho)/dln T at fixed rho, dp/drho at fixed T.
// Beyond the table in T its lookup is that of the nearest end, with the true temperature in
// front: the gas is then ideal with the composition it has there.
fn hhe_thermo(rho: f32, lnT: f32) -> HheThermo {
  let T = exp(lnT);
  let h = tab_lookup(tab_at(log(rho) / TAB_LN10, lnT / TAB_LN10));
  let sm = HH_KMU * T;
  let rm = hhe_rad(T, rho);
  var r: HheThermo;
  r.e = -sm * h.ht + rm;
  r.de = -sm * (h.ht + h.htt) + 4.0 * rm;
  r.pr = sm * h.hn + rm / 3.0;
  r.dpr = sm * (h.hn + h.hnt) + 4.0 * rm / 3.0;
  r.dprho = sm * (h.hn + h.hnn);
  return r;
}
// c^2 = (dp/drho)_T + T (dp/dT)_rho^2 / (rho^2 c_v), in code units. The ratio of the two slopes
// first, and bounded: with radiation in thin gas they reach 1e31 erg/g each, and a compiler
// that multiplies before it divides (Metal's fast math did) squares one of them past single
// precision's limit. The bound is a function it cannot fold into the product.
fn hhe_c2(t: HheThermo) -> f32 {
  let g = min(t.dpr / t.de, 1e6);
  return max((t.dprho + t.dpr * g) / P.eos.y, HH_C2_MIN * t.pr / P.eos.y);
}

// What Newton's method needs (tab_fd of eos/table.wgsl): the energy per gram (byP false) or
// p / rho (byP true) at ln T, and its derivative in ln T.
struct TabCtx { rho: f32 }
fn tab_fd(at: TabAt, lnT: f32, byP: bool, c: TabCtx) -> vec2<f32> {
  let T = exp(lnT);
  let h = tab_lookup2(at, byP);
  let sm = HH_KMU * T;
  let rm = hhe_rad(T, c.rho);
  if (byP) { return vec2<f32>(sm * h.x + rm / 3.0, sm * (h.x + h.y) + 4.0 * rm / 3.0); }
  return vec2<f32>(-sm * h.x + rm, -sm * (h.x + h.y) + 4.0 * rm);
}
// Above 1e8 K: how far, in ln T, for a gas that goes as T (hhe_thermo there) with radiation as
// T^4 to reach the goal: g e^d + r e^(4d) = goal. Newton from the smaller of the two bounds
// each term alone gives, which is within a factor of two of the goal, from above.
fn tab_above(fd: vec2<f32>, goal: f32, byP: bool, c: TabCtx) -> f32 {
  let r = hhe_rad(exp(TAB_LNT_MAX), c.rho) * select(1.0, 1.0 / 3.0, byP);
  let g = max(fd.x - r, 1e-30 * fd.x);
  var d = log(goal / g);
  if (r > 0.0) { d = min(d, 0.25 * log(goal / r)); }
  for (var it = 0; it < 5; it++) {
    let eg = g * exp(d);
    let er = r * exp(4.0 * d);
    d = max(d - (eg + er - goal) / (eg + 4.0 * er), 0.0);
  }
  return d;
}

// ln T at which the energy per gram (byP false) or p / rho (byP true) is `goal`, at density
// rho (g/cm^3). The starting point is the table of the gas's own inverse, which is what
// carries the solve through dissociation and ionization, where the energy rises steeply with
// T; with radiation, no more than the temperature whose radiation alone is the goal.
fn hhe_solve(rho: f32, goal: f32, byP: bool) -> f32 {
  if (!(goal > 0.0)) { return -1e30; }
  let lx = log(rho) / TAB_LN10;
  let gy = clamp((log(goal) / TAB_LN10 - select(HH_E0, HH_P0, byP)) / select(HH_DE, HH_DP, byP), 0.0, f32(TAB_NY) - 1.0001);
  let a = tab_bilinear(TAB_NX, tab_x(lx), gy);
  var lnT = clamp(select(a.y, a.z, byP) * TAB_LN10, TAB_LNT_MIN, TAB_LNT_MAX);
  var hi = TAB_LNT_MAX;
  if (P.eos.w > 0.0) {
    hi = clamp(HH_LN1E4 + 0.25 * (log(goal) + log(rho) + select(0.0, 1.0986123, byP) - log(HH_A4 * P.eos.w)), TAB_LNT_MIN, TAB_LNT_MAX);
    lnT = min(lnT, hi);
  }
  return tab_solve(lnT, lx, goal, byP, hi, TabCtx(rho));
}

fn eos_temperature(rho: f32, eint: f32) -> f32 {
  let lnT = hhe_solve(rho * P.eos.x, eint / rho * P.eos.y, false);
  return exp(lnT) / P.eos.z;
}
fn eos_pressure(rho: f32, eint: f32) -> f32 {
  let rc = rho * P.eos.x;
  let goal = eint / rho * P.eos.y;
  let lnT = hhe_solve(rc, goal, false);
  if (lnT < -1e29) { return 0.0; }
  let t = hhe_thermo(rc, lnT);
  return (t.pr + t.dpr * tab_fix(t.e, t.de, goal, lnT)) * rho / P.eos.y;
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  let rc = rho * P.eos.x;
  let lnT = hhe_solve(rc, p / rho * P.eos.y, true);
  if (lnT < -1e29) { return 0.0; }
  return sqrt(hhe_c2(hhe_thermo(rc, lnT)));
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  let rc = rho * P.eos.x;
  let goal = p / rho * P.eos.y;
  let lnT = hhe_solve(rc, goal, true);
  if (lnT < -1e29) { return 0.0; }
  let t = hhe_thermo(rc, lnT);
  return (t.e + t.de * tab_fix(t.pr, t.dpr, goal, lnT)) * rho / P.eos.y;
}
// A face state's sound speed and internal energy density (hydro/flux.wgsl), from one solve
// for its temperature.
fn eos_face(rho: f32, p: f32) -> vec2<f32> {
  let rc = rho * P.eos.x;
  let goal = p / rho * P.eos.y;
  let lnT = hhe_solve(rc, goal, true);
  if (lnT < -1e29) { return vec2<f32>(0.0); }
  let t = hhe_thermo(rc, lnT);
  return vec2<f32>(sqrt(hhe_c2(t)), (t.e + t.de * tab_fix(t.pr, t.dpr, goal, lnT)) * rho / P.eos.y);
}
// A cell's pressure and sound speed from its internal energy density (hydro/prim.wgsl), from
// one solve for its temperature.
fn eos_prim(rho: f32, eint: f32) -> vec2<f32> {
  let rc = rho * P.eos.x;
  let goal = eint / rho * P.eos.y;
  let lnT = hhe_solve(rc, goal, false);
  if (lnT < -1e29) { return vec2<f32>(0.0); }
  let t = hhe_thermo(rc, lnT);
  return vec2<f32>((t.pr + t.dpr * tab_fix(t.e, t.de, goal, lnT)) * rho / P.eos.y, sqrt(hhe_c2(t)));
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  if (!(T > 0.0)) { return 0.0; }
  return hhe_thermo(rho * P.eos.x, log(T * P.eos.z)).e * rho / P.eos.y;
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl): that of the pressure
// floor for cold gas, whose energy is 3/2 of its pressure (H2 below some 50 K does not turn).
fn eos_eint_floor(rho: f32) -> f32 {
  return 1.5 * P.p_floor;
}

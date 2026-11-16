// Degenerate electrons at any temperature, ions and radiation: a tabulated equation of state in
// the manner of Timmes & Swesty's Helmholtz (eos/helmholtz.ts), for white dwarf matter that
// burns. Pressure and energy are
//   p = P_deg(rho) + P_e,th(n, T) + n_i k T + a T^4 / 3
//   e = E_deg(rho) + E_e,th(n, T) + 3/2 n_i k T + a T^4
// with P_deg, E_deg the zero-temperature electrons (as eos/degenerate.wgsl) and the electron-
// positron gas's thermal part from its free energy, F_th = -P_pair(T) + n k T h(n, T): h
// tabulated in log10(rho Ye) and log10 T with its first derivatives (bicubic Hermite between
// nodes), P_pair the pair gas at zero net charge as a Chebyshev series in log10 T. Electrons
// are counted once: unlike eos/degenerate.wgsl, whose ideal-gas part gives every electron
// 3/2 k T and so reads shocked degenerate fuel 2-5 times too cold.
//
// The hydro holds the energy (or the pressure), so the temperature comes from Newton's method
// in ln T, started from a table of the inverse at three Abar (4, 14, 56; within a few percent,
// so two or three iterations converge it). Inside, everything is in cgs, kelvin and per unit
// mass, which keeps single precision clear of its 3e38 limit; the code's units are converted
// on the way in and out.
//
// The ions' Coulomb energy is added to both, from one free energy (the one-component plasma's,
// as Helmholtz: Yakovlev & Shalybkov 1989), so the two stay consistent. Its part that does not
// depend on T joins the zero-temperature electrons (helm_cold); the rest is thermal
// (helm_coul). Both are faded out in thin matter, where they do not apply. Derivation and
// tests: eos/helmholtz.ts.
//
// P.eos  = (g/cm^3, erg/g, kelvin per code unit of density, specific energy, temperature; mu
//           of the reference composition)
// P.eos2 = (rho_d, A of the zero-temperature electrons in code units (eos/degenerate.wgsl), Ye
//           of the reference composition, and the multiplier of the Coulomb terms: 1, or 0
//           for none)
// The ions per atomic mass unit are 1 / mu - Ye. Where the composition varies (eos/comp_on),
// Ye is the reference's times g_comp.y, and the ions per atomic mass unit are
// g_comp.x / mu - Ye (g_comp.x counts ions and electrons).
// Float64 mirror: src/eos/helmholtz.ts.
const EOS_HAS_ENERGY = true;
// As eos/degenerate.wgsl: E - KE below the zero-temperature energy is no reason to rebuild E.
const EOS_KEEP_TOTAL = true;

#include "eos/helmholtz_consts.wgsl"

@group(0) @binding(22) var HELM_TAB: texture_2d<f32>;

const HK: f32 = 1.380649e-16;
const HMU: f32 = 1.6605390666e-24;
const HMEC2: f32 = 8.1871057769e-7;
const HARAD: f32 = 7.565723e-15;
const HLN10: f32 = 2.30258509;
// 2 pi m_e k / h^2, cgs
const HNQ: f32 = 1.79988e10;
const HLNT_MIN: f32 = 9.2103404;     // ln 1e4
const HLNT_MAX: f32 = 25.328436;     // ln 1e11

// The gas's adiabatic index, for the few uses outside the module (entropy shown in the view).
fn eos_gamma() -> f32 { return 5.0 / 3.0; }
fn eos_gm1() -> f32 { return 2.0 / 3.0; }

fn helm_ye() -> f32 {
  if (COMP) { return P.eos2.z * g_comp.y; }
  return P.eos2.z;
}
// Ions per atomic mass unit, 1 / Abar.
fn helm_ions() -> f32 {
  if (COMP) { return max(g_comp.x / P.eos.w - helm_ye(), 1e-3); }
  return max(1.0 / P.eos.w - P.eos2.z, 1e-3);
}

// ---- The zero-temperature electrons (as eos/degenerate.wgsl) ----

fn deg_rho_d() -> f32 {
  if (COMP) { return P.eos2.x / g_comp.y; }
  return P.eos2.x;
}
fn deg_fg(x: f32) -> vec2<f32> {
  if (x < 0.5) {
    let x2 = x * x;
    var xp = x2 * x2 * x;
    var c = 1.0;
    var b = 0.5;
    var f = 0.0;
    var g = 0.0;
    for (var k = 0; k < 12; k++) {
      let n = f32(5 + 2 * k);
      f += 8.0 * c * xp / n;
      g += (8.0 * b - 8.0 * c / n) * xp;
      let kf = f32(k);
      c *= -(2.0 * kf + 1.0) / (2.0 * kf + 2.0);
      b *= (0.5 - (kf + 1.0)) / (kf + 2.0);
      xp *= x2;
    }
    return vec2<f32>(f, g);
  }
  let s = sqrt(1.0 + x * x);
  let f = x * (2.0 * x * x - 3.0) * s + 3.0 * asinh(x);
  return vec2<f32>(f, 8.0 * x * x * x * (s - 1.0) - f);
}
fn deg_x(rho: f32) -> f32 { return pow(rho / deg_rho_d(), 1.0 / 3.0); }
// (P_deg, E_deg) at density rho, code units.
fn deg_pe(rho: f32) -> vec2<f32> { return P.eos2.y * deg_fg(deg_x(rho)); }
// dP_deg/drho, code units.
fn deg_c2(rho: f32) -> f32 {
  let x = deg_x(rho);
  let x2 = x * x;
  return P.eos2.y * 8.0 * x2 * x2 * x / (3.0 * rho * sqrt(1.0 + x2));
}

// ---- The table ----

struct HelmH { hn: f32, ht: f32, hnn: f32, hnt: f32, htt: f32 }

fn helm_hb(t: f32) -> vec4<f32> { return vec4<f32>(2.0*t*t*t - 3.0*t*t + 1.0, t*t*t - 2.0*t*t + t, -2.0*t*t*t + 3.0*t*t, t*t*t - t*t); }
fn helm_hb1(t: f32) -> vec4<f32> { return vec4<f32>(6.0*t*t - 6.0*t, 3.0*t*t - 4.0*t + 1.0, -6.0*t*t + 6.0*t, 3.0*t*t - 2.0*t); }
fn helm_hb2(t: f32) -> vec4<f32> { return vec4<f32>(12.0*t - 6.0, 6.0*t - 4.0, -12.0*t + 6.0, 6.0*t - 2.0); }
// sum over the four nodes of A_a(u) B_b(v) (h, h_x, h_y, h_xy): basis entries 0 and 2 weigh the
// value at nodes 0 and 1, entries 1 and 3 the slope there. Written out node by node, with no
// array: arrays in a function live in each thread's local memory, and on Firefox (Direct3D,
// NVIDIA) the adaptive mesh's update ran out of it (SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE) and lost the device.
fn helm_node(q: vec4<f32>, a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
  return a0 * b0 * q.x + a1 * b0 * q.y + a0 * b1 * q.z + a1 * b1 * q.w;
}
// The nodes (u, v) = (0, 0), (0, 1), (1, 0), (1, 1).
struct HelmCell { c00: vec4<f32>, c01: vec4<f32>, c10: vec4<f32>, c11: vec4<f32> }
fn helm_comb(c: HelmCell, A: vec4<f32>, B: vec4<f32>) -> f32 {
  return helm_node(c.c00, A.x, A.y, B.x, B.y) + helm_node(c.c01, A.x, A.y, B.z, B.w)
       + helm_node(c.c10, A.z, A.w, B.x, B.y) + helm_node(c.c11, A.z, A.w, B.z, B.w);
}
// The four nodes of the table's cell (i, j): i along log10(rho Ye), j along log10 T.
fn helm_cell(i: i32, j: i32) -> HelmCell {
  return HelmCell(
    textureLoad(HELM_TAB, vec2<i32>(j, i), 0),
    textureLoad(HELM_TAB, vec2<i32>(j + 1, i), 0),
    textureLoad(HELM_TAB, vec2<i32>(j, i + 1), 0),
    textureLoad(HELM_TAB, vec2<i32>(j + 1, i + 1), 0),
  );
}
// The table's cell at log10(rho Ye) and log10 T, and where in it the point lies.
struct HelmAt { c: HelmCell, u: f32, v: f32 }
fn helm_at(lrye: f32, lt: f32) -> HelmAt {
  let x = clamp((lrye - HT_R0) / HT_D, 0.0, f32(HT_NR) - 1.0001);
  let y = clamp((lt - HT_T0) / HT_D, 0.0, f32(HT_NT) - 1.0001);
  let i = i32(floor(x));
  let j = i32(floor(y));
  return HelmAt(helm_cell(i, j), x - f32(i), y - f32(j));
}
// h's first and second derivatives per ln n and ln T there (h itself is never needed).
fn helm_lookup(lrye: f32, lt: f32) -> HelmH {
  let a = helm_at(lrye, lt);
  let s = 1.0 / (HT_D * HLN10);
  let A0 = helm_hb(a.u);
  let A1 = helm_hb1(a.u);
  let B0 = helm_hb(a.v);
  let B1 = helm_hb1(a.v);
  var r: HelmH;
  r.hn = helm_comb(a.c, A1, B0) * s;
  r.ht = helm_comb(a.c, A0, B1) * s;
  r.hnn = helm_comb(a.c, helm_hb2(a.u), B0) * s * s;
  r.hnt = helm_comb(a.c, A1, B1) * s * s;
  r.htt = helm_comb(a.c, A0, helm_hb2(a.v)) * s * s;
  return r;
}
// The two of them Newton's method needs: (h_T, h_TT) to solve by the energy, (h_n, h_nT) by
// the pressure. A third of the work of the whole set.
fn helm_lookup2(a: HelmAt, byP: bool) -> vec2<f32> {
  let s = 1.0 / (HT_D * HLN10);
  let B1 = helm_hb1(a.v);
  if (byP) {
    let A1 = helm_hb1(a.u);
    return vec2<f32>(helm_comb(a.c, A1, helm_hb(a.v)) * s, helm_comb(a.c, A1, B1) * s * s);
  }
  let A0 = helm_hb(a.u);
  return vec2<f32>(helm_comb(a.c, A0, B1) * s, helm_comb(a.c, A0, helm_hb2(a.v)) * s * s);
}

// The three Chebyshev series of the pair gas (HP_Q, HP_DQ, HP_D2Q) at x, together, by
// Clenshaw's recurrence. Unrolled, so every coefficient is a constant: a loop indexes the series
// at run time, which puts them in each thread's local memory (see helm_comb).
fn helm_clen(b: mat2x3<f32>, x: f32, c: vec3<f32>) -> mat2x3<f32> {
  return mat2x3<f32>(2.0 * x * b[0] - b[1] + c, b[0]);
}
fn helm_cheb(x: f32) -> vec3<f32> {
  var b = mat2x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0));
  b = helm_clen(b, x, vec3<f32>(HP_Q[16], HP_DQ[16], HP_D2Q[16]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[15], HP_DQ[15], HP_D2Q[15]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[14], HP_DQ[14], HP_D2Q[14]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[13], HP_DQ[13], HP_D2Q[13]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[12], HP_DQ[12], HP_D2Q[12]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[11], HP_DQ[11], HP_D2Q[11]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[10], HP_DQ[10], HP_D2Q[10]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[9], HP_DQ[9], HP_D2Q[9]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[8], HP_DQ[8], HP_D2Q[8]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[7], HP_DQ[7], HP_D2Q[7]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[6], HP_DQ[6], HP_D2Q[6]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[5], HP_DQ[5], HP_D2Q[5]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[4], HP_DQ[4], HP_D2Q[4]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[3], HP_DQ[3], HP_D2Q[3]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[2], HP_DQ[2], HP_D2Q[2]));
  b = helm_clen(b, x, vec3<f32>(HP_Q[1], HP_DQ[1], HP_D2Q[1]));
  return x * b[0] - b[1] + vec3<f32>(HP_Q[0], HP_DQ[0], HP_D2Q[0]);
}
// The pair gas at zero charge, per unit mass at density rho: (P_p / rho, dln P_p/dln T,
// d^2 ln P_p/dln T^2). Below 10^8.5 K, the nonrelativistic Boltzmann pairs, already negligible;
// below 10^8.3 K (HP_NONE), none: there they are under 1e-8 of the radiation, at any density,
// which single precision cannot hold beside it, and the series is not summed.
const HP_NONE: f32 = 8.3;
fn helm_pairs(lt: f32, rho: f32) -> vec3<f32> {
  if (lt < HP_NONE) { return vec3<f32>(0.0); }
  let lnT = lt * HLN10;
  let T = exp(lnT);
  let beta = HK * T / HMEC2;
  let x = clamp((lt - HP_MID) / HP_HALF, -1.0, 1.0);
  let q = helm_cheb(x);
  let s = 1.0 / (HP_HALF * HLN10);
  let lnP = log(4.0 * HK / rho) + lnT + 1.5 * log(HNQ * T) - 1.0 / beta + q.x;
  return vec3<f32>(exp(lnP), 2.5 + 1.0 / beta + q.y * s, -1.0 / beta + q.z * s * s);
}

struct HelmThermo { e: f32, pr: f32, de: f32, dpr: f32, dprho: f32 }

// ---- The ions' Coulomb energy (eos/helmholtz.ts has the derivation) ----

const HC_A1: f32 = -0.898004;
const HC_B1: f32 = 0.96786;
const HC_C1: f32 = 0.220703;
const HC_D1: f32 = -0.86097;
const HC_A2: f32 = 0.29561;
const HC_B2: f32 = 1.9885;
const HC_C2: f32 = 0.288675;
// f_th continuous at Gamma = 1.
const HC_K: f32 = -2.5193142;
// Gamma T = HC_GT Zbar^2 (rho / Abar)^(1/3), cgs and kelvin: e^2 (4 pi / 3 m_u)^(1/3) / k.
const HC_GT: f32 = 227470.67;
// The fade's density scale is HC_FADE Zbar^2 / Ye g/cm^3 (HELM_FADE in eos/helmholtz.ts).
const HC_FADE: f32 = 11.488235;

// The thermal Coulomb free energy per ion over k T, f_th, at coupling G: (f_th, g = G f_th',
// G g'). All positive, as are g - G g' and f_th - g.
fn helm_cfns(G: f32) -> vec3<f32> {
  if (G >= 1.0) {
    let q = sqrt(sqrt(G));
    return vec3<f32>(4.0 * HC_B1 * q - 4.0 * HC_C1 / q + HC_D1 * log(G) + HC_K, HC_B1 * q + HC_C1 / q + HC_D1, 0.25 * (HC_B1 * q - HC_C1 / q));
  }
  let s = G * sqrt(G);
  let y = pow(G, HC_B2);
  return vec3<f32>(-HC_A1 * G - 2.0 * HC_C2 * s + HC_A2 / HC_B2 * y, -HC_A1 * G - 3.0 * HC_C2 * s + HC_A2 * y, -HC_A1 * G - 4.5 * HC_C2 * s + HC_A2 * HC_B2 * y);
}
// At density rho (g/cm^3): the fade w, the cold Coulomb energy per gram before it (erg/g,
// negative), and Gamma T.
fn helm_cscale(rho: f32, ye: f32, ia: f32) -> vec3<f32> {
  let z = ye / ia;
  let r = rho * ye / (HC_FADE * z * z);
  let gT = HC_GT * z * z * pow(rho * ia, 1.0 / 3.0);
  return vec3<f32>(r / (1.0 + r), HC_A1 * ia * HK / HMU * gT, gT);
}
// The thermal part, in the terms of HelmThermo, given helm_cscale's answer at the density
// (which does not change as Newton's method looks for the temperature).
fn helm_coul(cs: vec3<f32>, lnT: f32, ia: f32) -> HelmThermo {
  let w = cs.x;
  let T = exp(lnT);
  let f = helm_cfns(cs.z / T);
  let s = P.eos2.w * w * ia * HK * T / HMU;
  var r: HelmThermo;
  r.e = s * f.y;
  r.de = s * (f.y - f.z);
  r.pr = s * (f.y / 3.0 + (1.0 - w) * f.x);
  r.dpr = s * ((f.y - f.z) / 3.0 + (1.0 - w) * (f.x - f.y));
  r.dprho = r.pr + s * (f.z / 9.0 + 2.0 / 3.0 * (1.0 - w) * f.y + (1.0 - w) * (1.0 - 2.0 * w) * f.x);
  return r;
}
// The zero-temperature matter at density rho, code units: (pressure, energy density, dp/drho)
// of the electrons and of the cold part of the Coulomb energy.
fn helm_cold(rho: f32) -> vec3<f32> {
  let d = deg_pe(rho);
  var r = vec3<f32>(d.x, d.y, deg_c2(rho));
  if (P.eos2.w > 0.0) {
    let cs = helm_cscale(rho * P.eos.x, helm_ye(), helm_ions());
    let w = cs.x;
    let s = P.eos2.w * w * cs.y / P.eos.y;
    r += vec3<f32>(s * rho * (4.0 / 3.0 - w), s * rho, s * (4.0 / 3.0 * (4.0 / 3.0 - w) + (1.0 - w) * (4.0 / 3.0 - 2.0 * w)));
  }
  return r;
}

// Everything above the zero-temperature matter, per unit mass, at density rho (g/cm^3) and
// ln T, with Ye and ions per atomic mass unit ia:
//   e (erg/g), p / rho, de/dln T, d(p/rho)/dln T at fixed rho, dp/drho at fixed T.
fn helm_thermo(rho: f32, lnT: f32, ye: f32, ia: f32) -> HelmThermo {
  let lt = lnT / HLN10;
  let T = exp(lnT);
  let hh = helm_lookup(log(rho * ye) / HLN10, lt);
  let pp = helm_pairs(lt, rho);
  let sm = ye * HK * T / HMU;           // n k T / rho
  let im = ia * HK * T / HMU;           // n_i k T / rho
  let rm = HARAD * T * T * T * T / rho; // a T^4 / rho
  var r: HelmThermo;
  r.e = pp.x * (pp.y - 1.0) - sm * hh.ht + 1.5 * im + rm;
  r.de = pp.x * (pp.y * pp.y - pp.y + pp.z) - sm * (hh.ht + hh.htt) + 1.5 * im + 4.0 * rm;
  r.pr = pp.x + sm * hh.hn + im + rm / 3.0;
  r.dpr = pp.x * pp.y + sm * (hh.hn + hh.hnt) + im + 4.0 * rm / 3.0;
  r.dprho = sm * (hh.hn + hh.hnn) + im;
  if (P.eos2.w > 0.0) {
    let c = helm_coul(helm_cscale(rho, ye, ia), lnT, ia);
    r.e += c.e;
    r.de += c.de;
    r.pr += c.pr;
    r.dpr += c.dpr;
    r.dprho += c.dprho;
  }
  return r;
}

// Newton's starting point: log10 T from the inverse table (byP: from p / rho, else from e) by
// bilinear interpolation, and between the three Abar in log Abar.
fn helm_guess(rho: f32, x: f32, ia: f32, byP: bool) -> f32 {
  let gx = clamp((log(rho) / HLN10 - HG_RHO0) / HG_DRHO, 0.0, f32(HG_NRHO) - 1.0001);
  let gy = clamp((log(x) / HLN10 - HG_X0) / HG_DX, 0.0, f32(HG_NX) - 1.0001);
  let i = i32(floor(gx));
  let j = i32(floor(gy));
  let u = gx - f32(i);
  let v = gy - f32(j);
  let row = select(HG_ROW_E, HG_ROW_P, byP);
  let a = mix(mix(textureLoad(HELM_TAB, vec2<i32>(j, row + i), 0), textureLoad(HELM_TAB, vec2<i32>(j + 1, row + i), 0), v),
              mix(textureLoad(HELM_TAB, vec2<i32>(j, row + i + 1), 0), textureLoad(HELM_TAB, vec2<i32>(j + 1, row + i + 1), 0), v), u);
  // Abar = 4, 14, 56 in r, g, b.
  let la = clamp(-log(ia), 1.3862944, 4.0253517);
  let w = (la - 1.3862944) / (2.6390573 - 1.3862944);
  let w2 = (la - 2.6390573) / (4.0253517 - 2.6390573);
  return select(mix(a.x, a.y, w), mix(a.y, a.z, w2), la > 2.6390573);
}

// What Newton's method needs of helm_thermo: the thermal energy per gram (byP false) or
// p_th / rho (byP true) at ln T, and its derivative in ln T.
fn helm_fd(rho: f32, at: HelmAt, lnT: f32, ye: f32, ia: f32, byP: bool, cs: vec3<f32>) -> vec2<f32> {
  let lt = lnT / HLN10;
  let T = exp(lnT);
  let h = helm_lookup2(at, byP);
  let pp = helm_pairs(lt, rho);
  let sm = ye * HK * T / HMU;
  let im = ia * HK * T / HMU;
  let rm = HARAD * T * T * T * T / rho;
  var r: vec2<f32>;
  if (byP) {
    r = vec2<f32>(pp.x + sm * h.x + im + rm / 3.0, pp.x * pp.y + sm * (h.x + h.y) + im + 4.0 * rm / 3.0);
  } else {
    r = vec2<f32>(pp.x * (pp.y - 1.0) - sm * h.x + 1.5 * im + rm, pp.x * (pp.y * pp.y - pp.y + pp.z) - sm * (h.x + h.y) + 1.5 * im + 4.0 * rm);
  }
  if (P.eos2.w > 0.0) {
    let c = helm_coul(cs, lnT, ia);
    r += select(vec2<f32>(c.e, c.de), vec2<f32>(c.pr, c.dpr), byP);
  }
  return r;
}

// Newton's method converges quadratically, so a step this small in ln T leaves an error of its
// square times half the curvature (at most a few): under the 1.5e-6 single precision holds
// ln T to. The step is taken and not checked by another evaluation; what the callers return is
// then corrected from the evaluation they make anyway (helm_fix).
const HELM_LAST_STEP: f32 = 3e-4;

// ln T at which the thermal energy per gram (byP false) or p_th / rho (byP true) is `goal`.
// Outside the table: below 1e4 K the gas's thermal energy is nearly that of its ions, linear in
// T; above 1e11 K, radiation's, as T^4.
fn helm_solve(rho: f32, goal: f32, ye: f32, ia: f32, byP: bool) -> f32 {
  if (!(goal > 0.0)) { return -1e30; }
  var lnT = clamp(helm_guess(rho, goal, ia, byP) * HLN10, HLNT_MIN, HLNT_MAX);
  // The inverse tables know nothing of the Coulomb heat, several times the ideal ions' in cold,
  // strongly coupled matter: allow for it at their guess, taking both parts as linear in T.
  var cs = vec3<f32>(0.0);
  if (P.eos2.w > 0.0) {
    cs = helm_cscale(rho, ye, ia);
    let c = helm_coul(cs, lnT, ia);
    lnT = max(lnT - log(1.0 + select(c.e, c.pr, byP) / goal), HLNT_MIN);
  }
  // The density's place in the table does not change, and after the first step or two neither
  // does the temperature's cell: its four nodes are fetched only when it does.
  let x = clamp((log(rho * ye) / HLN10 - HT_R0) / HT_D, 0.0, f32(HT_NR) - 1.0001);
  let i = i32(floor(x));
  var j = -1;
  var cell: HelmCell;
  for (var it = 0; it < 10; it++) {
    let y = clamp((lnT / HLN10 - HT_T0) / HT_D, 0.0, f32(HT_NT) - 1.0001);
    let jn = i32(floor(y));
    if (jn != j) {
      j = jn;
      cell = helm_cell(i, j);
    }
    let fd = helm_fd(rho, HelmAt(cell, x - f32(i), y - f32(j)), lnT, ye, ia, byP, cs);
    if (lnT <= HLNT_MIN && fd.x > goal) { return HLNT_MIN + log(goal / fd.x); }
    if (lnT >= HLNT_MAX && fd.x < goal) { return HLNT_MAX + 0.25 * log(goal / fd.x); }
    let step = clamp(-(fd.x - goal) / fd.y, -1.0, 1.0);
    lnT = clamp(lnT + step, HLNT_MIN, HLNT_MAX);
    // At an end of the table the next pass decides whether the answer lies beyond it.
    if (abs(step) < HELM_LAST_STEP && lnT > HLNT_MIN && lnT < HLNT_MAX) { break; }
  }
  return lnT;
}

// The evaluation at the temperature found gives what is returned, and with it one more
// Newton step for nothing: the step in ln T that would take the solved quantity (the energy,
// or p / rho if byP) from its value there onto `goal`. Callers move what they return by its
// own slope times this, so the pair (energy, pressure) they hand back is consistent to second
// order in what the iteration left, with no evaluation the iteration did not already owe.
// Inside the table only: beyond it the temperature is an extrapolation.
fn helm_fix(t: HelmThermo, goal: f32, byP: bool, lnT: f32) -> f32 {
  if (lnT <= HLNT_MIN || lnT >= HLNT_MAX) { return 0.0; }
  return clamp((goal - select(t.e, t.pr, byP)) / select(t.de, t.dpr, byP), -1e-3, 1e-3);
}

// Thermal energy per gram and p_th / rho in cgs, from the code's internal energy density.
fn helm_eth(rho: f32, eint: f32, ecold: f32) -> f32 {
  return max(eint - ecold, 0.0) / rho * P.eos.y;
}

fn eos_temperature(rho: f32, eint: f32) -> f32 {
  let lnT = helm_solve(rho * P.eos.x, helm_eth(rho, eint, helm_cold(rho).y), helm_ye(), helm_ions(), false);
  return exp(lnT) / P.eos.z;
}
fn eos_pressure(rho: f32, eint: f32) -> f32 {
  let rc = rho * P.eos.x;
  let ye = helm_ye();
  let ia = helm_ions();
  let d = helm_cold(rho);
  let goal = helm_eth(rho, eint, d.y);
  let lnT = helm_solve(rc, goal, ye, ia, false);
  var pth = 0.0;
  if (lnT > -1e29) {
    let t = helm_thermo(rc, lnT, ye, ia);
    pth = (t.pr + t.dpr * helm_fix(t, goal, false, lnT)) * rho / P.eos.y;
  }
  return d.x + pth;
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  let rc = rho * P.eos.x;
  let ye = helm_ye();
  let ia = helm_ions();
  let d = helm_cold(rho);
  let prth = max(p - d.x, 0.0) / rho * P.eos.y;
  var c2 = d.z;
  let lnT = helm_solve(rc, prth, ye, ia, true);
  if (lnT > -1e29) {
    let t = helm_thermo(rc, lnT, ye, ia);
    // c^2 = (dp/drho)_T + T (dp/dT)_rho^2 / (rho^2 c_v), all per unit mass.
    c2 += (t.dprho + t.dpr * (t.dpr / t.de)) / P.eos.y;
  }
  return sqrt(c2);
}
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  let rc = rho * P.eos.x;
  let ye = helm_ye();
  let ia = helm_ions();
  let d = helm_cold(rho);
  let goal = max(p - d.x, 0.0) / rho * P.eos.y;
  let lnT = helm_solve(rc, goal, ye, ia, true);
  var eth = 0.0;
  if (lnT > -1e29) {
    let t = helm_thermo(rc, lnT, ye, ia);
    eth = (t.e + t.de * helm_fix(t, goal, true, lnT)) * rho / P.eos.y;
  }
  return d.y + eth;
}
// A face state's sound speed and internal energy density (hydro/flux.wgsl), from one Newton
// solve for its temperature: eos_sound_speed and eos_eint_from_p each solve for the same one,
// and at every face of every cell those solves were nearly all of the update's time.
fn eos_face(rho: f32, p: f32) -> vec2<f32> {
  let rc = rho * P.eos.x;
  let ye = helm_ye();
  let ia = helm_ions();
  let d = helm_cold(rho);
  var c2 = d.z;
  var eth = 0.0;
  let goal = max(p - d.x, 0.0) / rho * P.eos.y;
  let lnT = helm_solve(rc, goal, ye, ia, true);
  if (lnT > -1e29) {
    let t = helm_thermo(rc, lnT, ye, ia);
    c2 += (t.dprho + t.dpr * (t.dpr / t.de)) / P.eos.y;
    eth = (t.e + t.de * helm_fix(t, goal, true, lnT)) * rho / P.eos.y;
  }
  return vec2<f32>(sqrt(c2), d.y + eth);
}
// A cell's pressure and sound speed from its internal energy density (hydro/prim.wgsl), from
// one solve for its temperature: eos_pressure and then eos_sound_speed would solve twice, by
// the energy and then by the pressure, for every cell of every stencil.
fn eos_prim(rho: f32, eint: f32) -> vec2<f32> {
  let rc = rho * P.eos.x;
  let ye = helm_ye();
  let ia = helm_ions();
  let d = helm_cold(rho);
  var c2 = d.z;
  var pth = 0.0;
  let goal = helm_eth(rho, eint, d.y);
  let lnT = helm_solve(rc, goal, ye, ia, false);
  if (lnT > -1e29) {
    let t = helm_thermo(rc, lnT, ye, ia);
    c2 += (t.dprho + t.dpr * (t.dpr / t.de)) / P.eos.y;
    pth = (t.pr + t.dpr * helm_fix(t, goal, false, lnT)) * rho / P.eos.y;
  }
  return vec2<f32>(d.x + pth, sqrt(c2));
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  let d = helm_cold(rho);
  if (!(T > 0.0)) { return d.y; }
  let lnT = log(T * P.eos.z);
  let rc = rho * P.eos.x;
  var eth: f32;
  if (lnT < HLNT_MIN) {
    eth = helm_thermo(rc, HLNT_MIN, helm_ye(), helm_ions()).e * exp(lnT - HLNT_MIN);
  } else {
    eth = helm_thermo(rc, min(lnT, HLNT_MAX), helm_ye(), helm_ions()).e;
  }
  return d.y + eth * rho / P.eos.y;
}
// As eos/degenerate.wgsl: no floor above the zero-temperature energy.
fn eos_eint_floor(rho: f32) -> f32 {
  return 0.0;
}

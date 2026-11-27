// Primitive state as the reconstruction sees it, and its limited reconstruction. Requires the
// limiter.
//   a = (rho, vx, vy, vz)    b = (p, scalar, c_s, 1 if the cell is solid)
struct Q { a: vec4<f32>, b: vec4<f32> }
#include <refdev>

fn q_solid(q: Q) -> bool {
  return q.b.w > 0.5;
}
// The image of a gas cell in a wall normal to `axis`: the same state with the normal velocity
// reversed. Standing in for the solid cells beyond the wall, it makes the wall reflecting.
fn q_mirror(q: Q, axis: i32) -> Q {
  var a = q.a;
  a[axis + 1] = -a[axis + 1];
  return Q(a, vec4<f32>(q.b.xyz, 0.0));
}
// The same for a half-step state, whose b holds flags that the image keeps.
fn pred_mirror(p: Q, axis: i32) -> Q {
  var a = p.a;
  a[axis + 1] = -a[axis + 1];
  return Q(a, p.b);
}

// Twice the increment from a cell's center to its upper face (edge_up) and from its lower face
// to its center (edge_dn), given the cell and its two neighbors along the axis. A slope
// limiter gives the same for both; a third-order one does not.
//
// The limiter is also given the natural scale of each quantity at the cell, to judge
// smoothness by: density and pressure themselves, the sound speed for velocities. The passive
// scalar has none (zero: always limited).
//
// Well-balanced reconstruction of the pressure. `dl` and `dr` are the differences of the
// gravitational potential from the lower neighbor to the cell and from the cell to the upper
// neighbor (zero where there is no gravity, or the option is off: then this is the plain
// reconstruction, bit for bit). In hydrostatic balance the pressure follows dp = -rho dphi,
// and it is that balance, not the pressure, that is smooth across a stratified cell. So what
// is limited and reconstructed is each neighbor's deviation from balance with the cell,
//   (p_r - p_c) + (rho_c + rho_r) / 2 * dr      and the same below,
// and the balanced part, -rho_c * dphi to the face, is added back unlimited. Where the gas
// obeys the trapezoid rule p_r - p_c = -(rho_c + rho_r) / 2 * dr exactly, the deviations
// vanish, the two cells sharing a face agree on its pressure, and the pressures of a cell's
// two faces differ by -rho_c (phi_r - phi_l) / 2: what the gravity kick adds to its momentum,
// with the acceleration a centered difference of the potential. The static state then holds
// to round-off, however steep. (After Kappeli & Mishra 2014, with the balance defined by the
// potential differences this code's kicks use.)
//
// `df` is the potential difference toward the face the increment is for: `dr` or `dl`,
// except next to a wall. The cell beyond a wall is the image of this one and has its
// potential, so there is no deviation across the wall (dl or dr is zero), but the wall face
// still lies half a cell down the true potential: df is the true difference there, and its
// negative for the image, so that both arrive at the same pressure on the wall.
// The scale a pressure difference is judged smooth against: the pressure itself, or rho c^2
// where the pressure takes both signs (a liquid's, eos/liquid.wgsl).
fn p_scale(c: Q) -> f32 {
  if (p_signed()) { return c.a.x * c.b.z * c.b.z; }
  return c.b.x + g_qs.y;
}
// A gas stored as its deviation from a reference (hydro/ref_on.wgsl): the reference density and
// pressure of the middle cell of the edge increment being taken, which its callers set, so that
// smoothness is judged against the whole density and pressure, not their deviations. Zero
// otherwise.
var<private> g_qs: vec2<f32> = vec2<f32>(0.0);
fn wb_below(l: Q, c: Q, dl: f32) -> vec4<f32> {
  var d = c.b - l.b;
  d.x += 0.5 * (l.a.x + c.a.x) * dl;
  return d;
}
fn wb_above(c: Q, r: Q, dr: f32) -> vec4<f32> {
  var d = r.b - c.b;
  d.x += 0.5 * (c.a.x + r.a.x) * dr;
  return d;
}
// Density reconstructed along the adiabat (P.flags bit 7, SimConfig.isentropicRecon; the ideal
// gas): what is limited is each neighbor's density brought to the cell's pressure adiabatically,
// rho_n (p_c / p_n)^(1/gamma), the analog for the entropy of what the balanced reconstruction
// does for the pressure, and the face density is that reconstruction carried to the face's
// pressure along the adiabat, kept between the densities of the cell and its neighbor beyond the
// face. In a layer at the adiabat the limited quantity is flat, and the face states keep the
// cell's entropy. Limited as it is, the density changes by about a part in a hundred per cell in a
// layer a hundred cells per scale height deep, where the limiter's switch lies, and the faces of a
// layer at the adiabat came out off it by (dx / H)^2: a layer 1e-5 of the adiabatic gradient on
// the stable side convected at Mach 0.008 (docs/numerics.md section 7).
// It works on the whole densities and pressures. For a gas stored as its deviation from a
// reference, the references of the increment's middle cell (g_qs) and of its lower and upper
// neighbors (g_ql, g_qr), which the callers set, make them; zero otherwise.
//
// Where the gas is not on one smooth adiabat it does harm: across a shock, in a reaction zone,
// in cold gas falling at several times its sound speed (where its pressure, and so the density
// tied to it, is the least certain part of the state). Unchecked it made a Sedov blast 1.7 cells
// out of round, drifted the Evrard collapse's energy 4.8% and slowed a detonation front by 1.2%.
// It is not switched off there: what it adds to the plain density's increment is bounded instead,
// by ISEN_CAP of the smaller density step to a neighbor, or by ISEN_ROOM of the density where the
// gas is smooth (the pressure changing by less than ISEN_SMOOTH from cell to cell, the neighbors'
// densities at the cell's pressure within ISEN_ADIABAT of the cell's, and the velocity changing
// by less than ISEN_SHEAR of the sound speed), that room falling to none at twice each of those
// bounds, continuously. In a stratified layer or slow convection the room is far more than the
// correction, (dx / H)^2 of the density, and the reconstruction acts in full; across a shock it
// moves the face by a quarter of a step at most.
// (Switches, and weights blending the two reconstructions, both failed. A switch turned a
// difference of rounding into one of method in the cells where it flipped: a Sedov blast in an
// octant differed from the whole box by 1.5e-5 in density, against 2e-6. A weight, however
// smooth, fed back through the large difference between the two reconstructions at a shock: in
// the converging flow onto a sink the whole box's own mirror asymmetry grew to 5e-3 by step 64
// with switches and 7.5e-3 with weights, against 1.4e-6 with the plain density. Bounding the
// correction instead, its gain is that of the bound, ISEN_ROOM of the density over the guards'
// bounds at most: 1.8e-4 by step 218, against 3.8e-5 with the plain density.)
const ISEN_SMOOTH: f32 = 0.2;
// A departure from one adiabat, per cell, as a fraction of the density: the limiter's own
// measure of smoothness (limiters/koren.wgsl SMOOTH). A layer 1e-2 off the adiabat, 50 cells per
// scale height deep, departs by 1e-4 per cell. A detonation's reaction zone is smooth in pressure
// but heated.
const ISEN_ADIABAT: f32 = 0.01;
// A change of velocity from cell to cell, as a fraction of the sound speed: gas falling onto a
// sink at Mach 3 changes by 0.2 to 0.3 per cell; convection at Mach 0.01 by far less.
const ISEN_SHEAR: f32 = 0.1;
// The bounds on the correction: a fraction of the smaller density step to a neighbor, and of the
// density where the gas is smooth. With the first at 0.5, the Evrard collapse's energy drifted
// 0.63% (0.23% with the plain density); with the second everywhere at 1% (no guards), a
// detonation's reaction zone was left in part at a coarser level of the mesh; at 5%, the converging
// flow onto a sink was unstable, or with the velocity guard a Sedov blast in an octant differed
// from the whole box by 3.2e-5.
const ISEN_CAP: f32 = 0.25;
const ISEN_ROOM: f32 = 0.01;
var<private> g_ql: vec2<f32> = vec2<f32>(0.0);
var<private> g_qr: vec2<f32> = vec2<f32>(0.0);
fn isen_on() -> bool { return (P.flags & 128u) != 0u && EOS_HAS_ENERGY && !p_signed(); }
// Twice the increment of the cell's stored density toward a face (sgn 1, its upper face) or from
// one (sgn -1, its lower face), given the pressure increment bx the cell returns there and the
// potential step df it was given, and the bound on its difference from the plain increment (0
// where a density or pressure is not positive).
fn isen_inc(l: Q, c: Q, r: Q, bx: f32, df: f32, sgn: f32) -> vec2<f32> {
  let gi = 1.0 / P.eos.x;
  let wl = vec2<f32>(l.a.x, l.b.x) + g_ql;
  let wc = vec2<f32>(c.a.x, c.b.x) + g_qs;
  let wr = vec2<f32>(r.a.x, r.b.x) + g_qr;
  if (!(all(wl > vec2<f32>(0.0)) && all(wc > vec2<f32>(0.0)) && all(wr > vec2<f32>(0.0)))) { return vec2<f32>(0.0); }
  // The neighbors' densities at the cell's pressure, less the cell's.
  let tl = wl.x * pow(wc.y / wl.y, gi) - wc.x;
  let tr = wr.x * pow(wc.y / wr.y, gi) - wc.x;
  let far = select(tr, -tl, sgn > 0.0);
  let near = select(-tl, tr, sgn > 0.0);
  let d = edge4(vec4<f32>(far, 0.0, 0.0, 0.0), vec4<f32>(near, 0.0, 0.0, 0.0), vec4<f32>(wc.x, 1.0, 1.0, 1.0)).x;
  // The whole pressure at the face: the deviation's increment and the reference's along its balance.
  let pf = wc.y + 0.5 * sgn * (bx - g_qs.x * df);
  let wn = select(wl.x, wr.x, sgn > 0.0);
  let rf = clamp((wc.x + 0.5 * sgn * d) * pow(max(pf, 1e-30) / wc.y, gi), min(wc.x, wn), max(wc.x, wn));
  // Less the reference's own increment, which the face adds (hydro/ref_on.wgsl ref_face).
  let a = g_qr.x - g_qs.x;
  let b = g_qs.x - g_ql.x;
  let rs = select(0.0, select(b, a, abs(a) < abs(b)), sign(a) * sign(b) > 0.0);
  // The bound (ISEN_CAP and ISEN_ROOM above).
  let ks = clamp(2.0 - max(abs(wl.y - wc.y), abs(wr.y - wc.y)) / (ISEN_SMOOTH * wc.y), 0.0, 1.0);
  let ka = clamp(2.0 - max(abs(tl), abs(tr)) / (ISEN_ADIABAT * wc.x), 0.0, 1.0);
  let kv = clamp(2.0 - max(length(r.a.yzw - c.a.yzw), length(c.a.yzw - l.a.yzw)) / (ISEN_SHEAR * max(c.b.z, 1e-30)), 0.0, 1.0);
  let bound = max(ISEN_CAP * min(abs(wc.x - wl.x), abs(wr.x - wc.x)), ISEN_ROOM * wc.x * ks * ka * kv);
  return vec2<f32>(2.0 * sgn * (rf - wc.x) - rs, bound);
}
// The plain density increment x moved toward the one along the adiabat, by its bound at most.
fn isen_blend(x: f32, i: vec2<f32>) -> f32 {
  return x + clamp(i.x - x, -i.y, i.y);
}
fn edge_up(l: Q, c: Q, r: Q, dl: f32, dr: f32, df: f32) -> Q {
  var b = edge4(wb_below(l, c, dl), wb_above(c, r, dr), vec4<f32>(p_scale(c), 0.0, c.b.z, 0.0));
  b.x -= c.a.x * df;
  var a = edge4(c.a - l.a, r.a - c.a, vec4<f32>(c.a.x + g_qs.x, c.b.zzz));
  if (isen_on()) { a.x = isen_blend(a.x, isen_inc(l, c, r, b.x, df, 1.0)); }
  return Q(a, b);
}
fn edge_dn(l: Q, c: Q, r: Q, dl: f32, dr: f32, df: f32) -> Q {
  var b: vec4<f32>;
  if (LIMITER_SYMMETRIC) {
    b = edge4(wb_below(l, c, dl), wb_above(c, r, dr), vec4<f32>(p_scale(c), 0.0, c.b.z, 0.0));
  } else {
    b = edge4(wb_above(c, r, dr), wb_below(l, c, dl), vec4<f32>(p_scale(c), 0.0, c.b.z, 0.0));
  }
  b.x -= c.a.x * df;
  var a: vec4<f32>;
  if (LIMITER_SYMMETRIC) {
    a = edge4(c.a - l.a, r.a - c.a, vec4<f32>(c.a.x + g_qs.x, c.b.zzz));
  } else {
    a = edge4(r.a - c.a, c.a - l.a, vec4<f32>(c.a.x + g_qs.x, c.b.zzz));
  }
  // (With either kind of limiter: a symmetric one's lower face took the plain density while its
  // upper face took this, one side of every cell reconstructed one way and the other another.)
  if (isen_on()) { a.x = isen_blend(a.x, isen_inc(l, c, r, b.x, df, -1.0)); }
  return Q(a, b);
}

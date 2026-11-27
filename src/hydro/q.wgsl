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
// pressure along the adiabat. In a layer at the adiabat the limited quantity is flat, and the
// face states keep the cell's entropy. Limited as it is, the density changes by about a part in a
// hundred per cell in a layer a hundred cells per scale height deep, where the limiter's switch
// lies, and the faces of a layer at the adiabat came out off it by (dx / H)^2: a layer 1e-5 of
// the adiabatic gradient on the stable side convected at Mach 0.008 (docs/numerics.md section 7).
// It works on the whole densities and pressures. For a gas stored as its deviation from a
// reference, the references of the increment's middle cell (g_qs) and of its lower and upper
// neighbors (g_ql, g_qr), which the callers set, make them; zero otherwise.
const ISEN_SMOOTH: f32 = 0.2;
var<private> g_ql: vec2<f32> = vec2<f32>(0.0);
var<private> g_qr: vec2<f32> = vec2<f32>(0.0);
// Only with the balanced reconstruction (a potential to balance against): it is stratification
// that it is for, and without one it only changed blasts (a Sedov blast 1.7 cells out of round
// against 1.5 with the plain density).
fn isen_on() -> bool { return (P.flags & 128u) != 0u && well_balanced() && EOS_HAS_ENERGY && !p_signed(); }
// Twice the increment of the cell's stored density toward a face (sgn 1, its upper face) or from
// one (sgn -1, its lower face), given the pressure increment bx the cell returns there and the
// potential step df it was given; -1e30 where a density or pressure is not positive.
fn isen_inc(l: Q, c: Q, r: Q, bx: f32, df: f32, sgn: f32) -> f32 {
  let gi = 1.0 / P.eos.x;
  let wl = vec2<f32>(l.a.x, l.b.x) + g_ql;
  let wc = vec2<f32>(c.a.x, c.b.x) + g_qs;
  let wr = vec2<f32>(r.a.x, r.b.x) + g_qr;
  if (!(all(wl > vec2<f32>(0.0)) && all(wc > vec2<f32>(0.0)) && all(wr > vec2<f32>(0.0)))) { return -1e30; }
  // Only where the pressure is smooth (within 20% from cell to cell; a layer 50 cells per scale
  // height changes by 2%): across a shock the gas is not on one adiabat, and there it drifted
  // the Evrard collapse's energy by 4.8% (0.23% with the plain density).
  if (abs(wl.y - wc.y) > ISEN_SMOOTH * wc.y || abs(wr.y - wc.y) > ISEN_SMOOTH * wc.y) { return -1e30; }
  // The neighbors' densities at the cell's pressure, less the cell's.
  let tl = wl.x * pow(wc.y / wl.y, gi) - wc.x;
  let tr = wr.x * pow(wc.y / wr.y, gi) - wc.x;
  let far = select(tr, -tl, sgn > 0.0);
  let near = select(-tl, tr, sgn > 0.0);
  let d = edge4(vec4<f32>(far, 0.0, 0.0, 0.0), vec4<f32>(near, 0.0, 0.0, 0.0), vec4<f32>(wc.x, 1.0, 1.0, 1.0)).x;
  // The whole pressure at the face: the deviation's increment and the reference's along its balance.
  let pf = wc.y + 0.5 * sgn * (bx - g_qs.x * df);
  let rf = (wc.x + 0.5 * sgn * d) * pow(max(pf, 1e-30) / wc.y, gi);
  // Less the reference's own increment, which the face adds (hydro/ref_on.wgsl ref_face).
  let a = g_qr.x - g_qs.x;
  let b = g_qs.x - g_ql.x;
  let rs = select(0.0, select(b, a, abs(a) < abs(b)), a * b > 0.0);
  return 2.0 * sgn * (rf - wc.x) - rs;
}
fn edge_up(l: Q, c: Q, r: Q, dl: f32, dr: f32, df: f32) -> Q {
  var b = edge4(wb_below(l, c, dl), wb_above(c, r, dr), vec4<f32>(p_scale(c), 0.0, c.b.z, 0.0));
  b.x -= c.a.x * df;
  var a = edge4(c.a - l.a, r.a - c.a, vec4<f32>(c.a.x + g_qs.x, c.b.zzz));
  if (isen_on()) {
    let i = isen_inc(l, c, r, b.x, df, 1.0);
    if (i > -1e29) { a.x = i; }
  }
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
  if (LIMITER_SYMMETRIC) { return Q(edge4(c.a - l.a, r.a - c.a, vec4<f32>(c.a.x + g_qs.x, c.b.zzz)), b); }
  var a = edge4(r.a - c.a, c.a - l.a, vec4<f32>(c.a.x + g_qs.x, c.b.zzz));
  if (isen_on()) {
    let i = isen_inc(l, c, r, b.x, df, -1.0);
    if (i > -1e29) { a.x = i; }
  }
  return Q(a, b);
}

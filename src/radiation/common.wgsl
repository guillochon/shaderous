// Shared by the radiation passes (radiation/radiation.ts): the constants of P.rad, the M1
// closure, the reduced speed of light per cell and the monotonized-central slope.
// Requires P (hydro/common.wgsl).

fn rad_c() -> f32 { return P.rad.x; }
fn rad_chat() -> f32 { return P.rad.y; }
fn rad_a() -> f32 { return P.rad.z; }
fn rad_flags() -> u32 { return u32(P.rad.w); }
// The reduced speed of light is chosen per cell (section 3.4 of the plan), not one everywhere.
fn rad_local_c() -> bool { return (rad_flags() & 2u) != 0u; }
// The face flux is global Lax-Friedrichs (speed c) rather than HLL with the M1 wave speeds.
fn rad_glf() -> bool { return (rad_flags() & 4u) != 0u; }

// Levermore's Eddington factor chi(f), f = |F| / (c E) clamped to [0, 1].
fn rad_chi(f: f32) -> f32 {
  let g = clamp(f, 0.0, 1.0);
  return (3.0 + 4.0 * g * g) / (5.0 + 2.0 * sqrt(max(4.0 - 3.0 * g * g, 0.0)));
}

// The reduced flux f = F / (c E) of a state (zero where there is no energy).
fn rad_reduced(E: f32, F: vec3<f32>) -> vec3<f32> {
  return select(vec3<f32>(0.0), F / (rad_c() * E), E > 0.0);
}

// Row `a` of the Eddington tensor D for the reduced flux f: D_ab = (1 - chi)/2 delta_ab +
// (3 chi - 1)/2 n_a n_b, n = f / |f|. The pressure tensor is P = D E.
fn rad_eddington_row(f: vec3<f32>, a: i32) -> vec3<f32> {
  let m = length(f);
  let chi = rad_chi(m);
  let n = select(vec3<f32>(0.0), f / m, m > 0.0);
  var row = 0.5 * (3.0 * chi - 1.0) * n[a] * n;
  row[a] += 0.5 * (1.0 - chi);
  return row;
}

// The reduction factor r of the speed of light in a cell of optical depth tau_c = kappa_R rho dx:
// min(1, (chat / c) max(1, tau_c / tau_0)), or chat / c everywhere without the per-cell rule.
// As reduction() in radiation/radiation.ts.
fn rad_reduction(tau: f32) -> f32 {
  let base = min(1.0, rad_chat() / rad_c());
  if (!rad_local_c()) { return base; }
  return min(1.0, base * max(1.0, tau / P.rad2.z));
}

// The sweep's slopes of (E, f): the central difference, at most 1.2 times either one-sided one
// for E and twice for f (as energySlope and mcSlope in reference/radiation.ts, where the 1.2 is
// explained: with 2 an opaque body's surface sends out its light in pulses).
const RAD_THETA = vec4<f32>(1.2, 2.0, 2.0, 2.0);
fn rad_slope(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let c = 0.5 * (a + b);
  let s = sign(c) * min(min(RAD_THETA * abs(a), RAD_THETA * abs(b)), abs(c));
  return select(vec4<f32>(0.0), s, a * b > vec4<f32>(0.0));
}

// Monotonized-central slope, componentwise (as mcSlope in reference/radiation.ts).
fn rad_mc(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let c = 0.5 * (a + b);
  let s = sign(c) * min(min(2.0 * abs(a), 2.0 * abs(b)), abs(c));
  return select(vec4<f32>(0.0), s, a * b > vec4<f32>(0.0));
}

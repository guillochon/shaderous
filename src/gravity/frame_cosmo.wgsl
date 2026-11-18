// The frame slot for comoving coordinates (src/cosmo.ts; docs/numerics.md section 22): the
// expansion drag, -(adot / a) rho v on the momentum and -3 (adot / a) p on the internal energy,
// each solved exactly over its interval and split around the step: from the scale factor at
// the start to that half way through before the fluxes (frame_open), and on to the end after
// the second kick (frame_close). The peculiar velocity goes as 1/a, so the momentum (whose
// comoving density does not change) does too; an ideal gas's rho e goes as a^(-3 (gamma - 1)).
// E changes by exactly what the kinetic and internal energies changed by, so the dual-energy
// pair stays in step. The interval is the clock's (a, ah, a1), whatever tau is passed.
// Requires P and the clock.
const FRAME_ON = true;

// (1 + e)^y - 1 for |e| below a percent (the scale factor changes by at most 1% a step), from
// the series of log(1 + e) and of exp(z) - 1. pow() would do, but its rounding is biased on
// some GPUs: over the few hundred steps from a = 0.01 to 0.1 it moved the pressure by 1e-4.
fn near_pow_m1(e: f32, y: f32) -> f32 {
  let l = e * (1.0 - e * (0.5 - e * (1.0 / 3.0 - e * 0.25)));
  let z = y * l;
  return z * (1.0 + z * (0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0))));
}

fn cosmo_drag(u1: vec4<f32>, u2: vec4<f32>, a_from: f32, a_to: f32) -> FrameState {
  // r = a_from / a_to = 1 + e
  let e = (a_from - a_to) / a_to;
  let rho = max(u1.x, P.rho_floor);
  let dke = 0.5 * (e * (2.0 + e)) * dot(u1.yzw, u1.yzw) / rho;
  let dei = u2.y * near_pow_m1(e, 3.0 * (P.eos.x - 1.0));
  return FrameState(vec4<f32>(u1.x, u1.yzw + u1.yzw * e), vec4<f32>(u2.x + dke + dei, u2.y + dei, u2.zw));
}

fn frame_open(u1: vec4<f32>, u2: vec4<f32>, tau: f32) -> FrameState {
  return cosmo_drag(u1, u2, clock.a, clock.ah);
}
fn frame_close(u1: vec4<f32>, u2: vec4<f32>, tau: f32) -> FrameState {
  return cosmo_drag(u1, u2, clock.ah, clock.a1);
}
// For the predictor's half step: the drag's rates of change of v and p.
fn frame_predict(v: vec3<f32>, p: f32) -> vec4<f32> {
  return vec4<f32>(-clock.hub * v, -3.0 * (P.eos.x - 1.0) * clock.hub * p);
}

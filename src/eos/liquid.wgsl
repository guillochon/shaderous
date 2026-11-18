// Weakly compressible liquid: p = c0^2 (rho - rho0), with rho0 = P.eos.x the density at which
// the pressure is zero and c0 = P.eos.y the sound speed, a constant. The sound speed is chosen,
// not physical: about ten times the fastest flow keeps the density within 1% of rho0, and the
// flow is incompressible to that accuracy (plans/incompressible.md, stage 1).
//
// Only the pressure's departure from zero is ever formed, as a difference of two nearby
// densities, which single precision gives exactly. An isothermal gas with the same sound speed
// would carry p = c0^2 rho, a hundred times the differences that drive the flow.
//
// The pressure is negative where rho < rho0. Runs with this module have P.p_floor negative
// (src/sim.ts packParams), which is how the shared passes know not to floor a pressure, divide
// by one or take one as a scale (p_signed, hydro/common.wgsl).
const EOS_HAS_ENERGY = false;
const EOS_KEEP_TOTAL = false;

// The display's entropy function divides by rho^gamma; a liquid has none of its own.
fn eos_gamma() -> f32 {
  return 1.0;
}

fn eos_pressure(rho: f32, eint: f32) -> f32 {
  return P.eos.y * P.eos.y * (rho - P.eos.x);
}
fn eos_sound_speed(rho: f32, p: f32) -> f32 {
  return P.eos.y;
}
// The elastic energy of compression per unit volume, int h drho with h = c0^2 ln(rho / rho0):
// rho0 c0^2 ((1 + x) ln(1 + x) - x) with x = p / (rho0 c0^2), to third order in x (x is a
// percent). With it the total energy is conserved as far as the scheme conserves it.
fn eos_eint_from_p(rho: f32, p: f32) -> f32 {
  let k = P.eos.x * P.eos.y * P.eos.y;
  let x = p / k;
  return 0.5 * k * x * x * (1.0 - x / 3.0);
}
// A liquid here has no temperature of its own: shown as c0^2, as an isothermal gas of that
// sound speed has.
fn eos_temperature(rho: f32, eint: f32) -> f32 {
  return P.eos.y * P.eos.y;
}
fn eos_eint_from_T(rho: f32, T: f32) -> f32 {
  return eos_eint_from_p(rho, eos_pressure(rho, 0.0));
}

// Least e_int the dual-energy switch accepts (hydro/dual_energy.wgsl): none.
fn eos_eint_floor(rho: f32) -> f32 {
  return 0.0;
}

// A face state's sound speed and internal energy density (hydro/flux.wgsl).
fn eos_face(rho: f32, p: f32) -> vec2<f32> {
  return vec2<f32>(P.eos.y, eos_eint_from_p(rho, p));
}
// A cell's pressure and sound speed from its internal energy density (hydro/prim.wgsl).
fn eos_prim(rho: f32, eint: f32) -> vec2<f32> {
  return vec2<f32>(eos_pressure(rho, eint), P.eos.y);
}

// Time-centering of the implicit step on blocks (pressure/theta.wgsl has the scheme). Requires
// clock, P and the equation of state.
#include "pressure/theta.wgsl"

// A gas's cell: theta from its switch chi (written by amr/pressure/source_gas.wgsl), its sound
// speed and its block's cell size. The source, the coefficients and the correction all form it
// with the same bits.
fn ap_theta(chi: f32, cs: f32, h: f32) -> f32 {
  return max(chi, select(1.0, ps_theta(cs, clock.dt, h), cs > 0.0));
}

// The liquid: one theta for the whole mesh, that of the finest cell the mesh may have. (theta
// varying from level to level would put it in the faces' coefficients, which the coarse levels'
// plain operator of a liquid does not hold: they would solve another equation.)
fn ap_theta_liquid() -> f32 {
  return ps_theta(eos_sound_speed(P.eos.x, 0.0), clock.dt, P.dx / f32(1u << P.amr.z));
}

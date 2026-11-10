// Perturber orbit, second half: from the reduced pull on the gas at the new positions, set the
// frame acceleration and the perturber's own acceleration, and apply the closing half kick.
//
// In an inertial frame the gas (mass m) accelerates at abar, the mass-weighted mean of the
// perturber's pull on it, and by Newton's third law the perturber accelerates at -(m / M) abar.
// The simulation frame accelerates with the gas, at abar. In that frame the perturber exerts no
// net force on the gas, so its centre of mass stays where it started, and the perturber
// accelerates at -(1 + m / M) abar.
//
// Nothing here assumes the star is a point mass, so the orbit responds to the tides it raises.
// Once debris leaves the box, "the gas" means what is still inside it.
//
// (Two alternatives were tried and dropped. A restoring spring on the centre of mass visibly
// shoves the remaining gas each time debris exits. Following a density-squared-weighted
// centroid instead keeps the densest gas centred, but lets the true centre of mass wander by
// several percent of the stellar radius during the disruption itself.)
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read> RED: array<f32>;
@group(0) @binding(3) var<storage, read_write> BH: Perturber;

@compute @workgroup_size(1)
fn main() {
  let m = RED[0];
  var abar = vec3<f32>(0.0);
  if (m > 0.0) { abar = vec3<f32>(RED[1], RED[2], RED[3]) / m; }
  let acc = -abar * (1.0 + m * P.G / P.bh.x);
  BH.frame = vec4<f32>(abar, 0.0);
  BH.acc = vec4<f32>(acc, 0.0);
  BH.vel = vec4<f32>(BH.vel.xyz + 0.5 * clock.dt * acc, 0.0);
}

// Explicit diffusion in the timestep. Requires P, the equation of state and g_dx set.
#include <diffusion>

// What a cell's diffusion adds to its signal speed. The diffusive fluxes (hydro/diffusion.wgsl)
// are taken from the state at the start of the step, which is stable for a step of up to
// dx^2 / (2 d D) in d dimensions. Counted as a speed of 2.5 d D / dx beside the gas's own, the
// step keeps a fifth in hand at a CFL number of 1 and falls as dx^2 where diffusion dominates.
//
// D is the larger of the two diffusivities. Viscosity: 4/3 nu, what the stress along a face's
// normal carries. Conduction: the flux is -kappa rho grad T, so the temperature diffuses at
// kappa rho / (de/dT). e / T stands for de/dT: exact for an ideal gas, kappa (gamma - 1), and
// an overestimate of the diffusivity where radiation adds to the heat capacity.
fn diff_speed(rho: f32, eint: f32) -> f32 {
  var D = 0.0;
  if (VISCOSITY) { D = 4.0 / 3.0 * P.diff.x; }
  if (CONDUCTION) { D = max(D, P.diff.y * rho * eos_temperature(rho, eint) / max(eint, 1.0e-30)); }
  return 2.5 * P.geom.y * D / g_dx;
}

// Start of a step's Poisson solve: keep the potential as the previous one (PHI_OLD, for the
// half step and the energy's correction) and set the solver's guess to its linear
// extrapolation in time, phi^n + (dt_n / dt_{n-1}) (phi^n - phi^{n-1}), instead of phi^n itself.
// The guess is then off by O(dt^2) rather than O(dt), and one V-cycle from it leaves the solve
// closer to converged than two from phi^n did (docs/numerics.md section 13): the solve's
// departure from a symmetric response, what it costs the total energy and momentum, is smaller.
// DTP holds the last step's dt (gravity/field.wgsl writes it); after a step of dt = 0 (past the
// end time) the two potentials are the same, and the guess is phi^n.
#include "gravity/mg_common.wgsl"

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var PHI_OLD: texture_storage_3d<r32float, read_write>;
@group(0) @binding(4) var<storage, read> clock: Clock;
@group(0) @binding(5) var<storage, read> DTP: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= lvl_dims())) { return; }
  let c = vec3<i32>(gid);
  let now = textureLoad(PHI, c).x;
  let old = textureLoad(PHI_OLD, c).x;
  // (At most twice the last step ahead: a step that lands on an output time can be far shorter
  // than the one after it.)
  let r = select(0.0, min(clock.dt / DTP[0], 2.0), DTP[0] > 0.0 && clock.dt > 0.0);
  textureStore(PHI_OLD, c, vec4<f32>(now, 0.0, 0.0, 0.0));
  textureStore(PHI, c, vec4<f32>(now + r * (now - old), 0.0, 0.0, 0.0));
}

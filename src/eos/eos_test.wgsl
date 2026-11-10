// Evaluates the active equation of state on a list of (rho, p) pairs, for validation.
#include "hydro/common.wgsl"
#include <eos>

@group(0) @binding(1) var<storage, read> IN: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> OUT: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  g_dx = P.dx;
  if (gid.x >= arrayLength(&IN)) { return; }
  let rho = IN[gid.x].x;
  let p = IN[gid.x].y;
  let eint = eos_eint_from_p(rho, p);
  OUT[gid.x] = vec4<f32>(eint, eos_pressure(rho, eint), eos_sound_speed(rho, p), eos_temperature(rho, eint));
}

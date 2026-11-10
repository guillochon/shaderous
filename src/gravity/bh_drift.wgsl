// Perturber orbit, first half of a kick-drift-kick step: half kick with the acceleration
// stored at the end of the previous step, then drift.
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read_write> BH: Perturber;

@compute @workgroup_size(1)
fn main() {
  let v = BH.vel.xyz + 0.5 * clock.dt * BH.acc.xyz;
  BH.vel = vec4<f32>(v, 0.0);
  BH.pos = vec4<f32>(BH.pos.xyz + clock.dt * v, 0.0);
}

// Point-mass orbits, first half of a kick-drift-kick step: half kick with the accelerations
// stored at the end of the previous step, then drift.
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read_write> BH: Masses;

@compute @workgroup_size(1)
fn main() {
  for (var i = 0u; i < mass_count(); i++) {
    let m = BH.list[i];
    let v = m.vel.xyz + 0.5 * clock.dt * m.acc.xyz;
    BH.list[i].vel = vec4<f32>(v, m.vel.w);
    BH.list[i].pos = vec4<f32>(m.pos.xyz + clock.dt * v, m.pos.w);
    // The sink's rate for this step, from the mass as it is now.
    let rs = m.acc.w;
    var rate = 0.0;
    if (rs > 0.0) { rate = m.sink.x * sqrt(m.pos.w / (rs * rs * rs)); }
    BH.list[i].sink = vec4<f32>(m.sink.xy, rate, 0.0);
  }
}

// Point-mass orbits, first half of a kick-drift-kick step: half kick with the accelerations
// stored at the end of the previous step, then drift.
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"
#include "gravity/rot.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var<storage, read_write> BH: Masses;

@compute @workgroup_size(1)
fn main() {
  for (var i = 0u; i < mass_count(); i++) {
    let m = BH.list[i];
    var v = m.vel.xyz + 0.5 * clock.dt * m.acc.xyz;
    // In a rotating frame the Coriolis force turns the velocity, by half a step's share here
    // and the other half before the closing kick (gravity/bh_kick.wgsl).
    if (rot_on()) { v = rot_turn(v, 0.5 * clock.dt); }
    // A fixed mass stays where it is.
    if (mass_fixed(m)) { v = vec3<f32>(0.0); }
    BH.list[i].vel = vec4<f32>(v, m.vel.w);
    BH.list[i].pos = vec4<f32>(m.pos.xyz + clock.dt * v, m.pos.w);
    // The sink's rate for this step, from the mass as it is now.
    let rs = m.acc.w;
    var rate = 0.0;
    if (rs > 0.0) { rate = m.sink.x * sqrt(m.pos.w / (rs * rs * rs)); }
    // A massless particle has no orbital rate of its own: its rate is given per unit time.
    if (rs > 0.0 && mass_massless(m)) { rate = m.sink.x; }
    BH.list[i].sink = vec4<f32>(m.sink.xy, rate, m.sink.w);
  }
}

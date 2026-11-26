// The damping zone's target at one point (hydro/damping.wgsl damping_target), for the query
// readout's equation terms (ui/terms.ts): the zone's rate and target state exist only as the
// problem's WGSL, so they are evaluated here, as the update does.
#include "hydro/common.wgsl"

struct DampTarget {
  rate: f32,
  rho: f32,
  vel: vec3<f32>,
  p: f32,
}

#include <damping>

// (x relative to the box center, cell size)
@group(0) @binding(1) var<storage, read> AT: array<vec4<f32>>;
// the run's clock: the target at its time
@group(0) @binding(3) var<storage, read> clock: Clock;
// (rate, rho, p, 0), (vel, 0)
@group(0) @binding(2) var<storage, read_write> OUT: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main() {
  g_dx = AT[0].w;
  var x = AT[0].xyz;
  if (P.dims.z == 1u) { x.z = 0.0; }
  let t = damping_target(x, clock.t);
  OUT[0] = vec4<f32>(select(0.0, t.rate, DAMPING), t.rho, t.p, 0.0);
  OUT[1] = vec4<f32>(t.vel, 0.0);
}

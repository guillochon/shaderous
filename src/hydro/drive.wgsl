// Turbulence driving as the update passes see it (hydro/drive_common.wgsl): the acceleration at
// a position, which they add to the momentum as a kick, its energy the kinetic energy it adds,
// k . (m + k/2) / rho. The buffer is the device's shared one (hydro/drive.ts), bound wherever a
// pass declares it.
#include "hydro/drive_common.wgsl"

@group(0) @binding(33) var<uniform> DRIVE: Drive;

fn drive_on() -> bool {
  return DRIVE.n.x > 0u;
}

// The driving acceleration at x (relative to the box center).
fn drive_accel(x: vec3<f32>) -> vec3<f32> {
  var a = vec3<f32>(0.0);
  for (var m = 0u; m < DRIVE.n.x; m++) {
    let ph = dot(DRIVE.mode[3u * m].xyz, x);
    a += DRIVE.mode[3u * m + 1u].xyz * cos(ph) - DRIVE.mode[3u * m + 2u].xyz * sin(ph);
  }
  return a;
}

// A kick of dt rho a(x) added to (kick, its energy) for gas of momentum m and density rho.
fn drive_kick(kick: vec3<f32>, ekick: f32, x: vec3<f32>, dt: f32, rho: f32, m: vec3<f32>) -> vec4<f32> {
  let kd = dt * rho * drive_accel(x);
  let e = dot(kd, m + kick + 0.5 * kd) / max(rho, P.rho_floor);
  return vec4<f32>(kick + kd, ekick + e);
}

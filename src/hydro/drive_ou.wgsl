// Turbulence driving: the Ornstein-Uhlenbeck step of the forcing amplitudes, run by the clock
// pass (hydro/clock.wgsl) over the time it has just set, so that every grid and every way of
// stepping advances them once per step (once per cycle with per-block timesteps, whose blocks
// then all see the same amplitudes). Over a time h each amplitude becomes
//   a f + sigma sqrt(1 - f^2) P xi,   f = exp(-h / tau),
// with xi a vector of unit normal deviates (the in-plane two in 2D) and P the projection
// zeta P_perp + (1 - zeta) P_par onto the parts of xi across and along k: solenoidal for
// zeta = 1, compressive for 0. Stationary, each component of a has variance sigma^2 before the
// projection. The deviates come from a counter-based hash of (seed, clock pass, mode, draw),
// so a run is the same from the same seed.
#include "hydro/drive_common.wgsl"

@group(0) @binding(33) var<storage, read_write> DRIVE: Drive;

fn drive_hash(x: u32) -> u32 {
  // PCG output permutation (Jarzynski & Olano 2020).
  let s = x * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
// A uniform deviate in (0, 1].
fn drive_uniform(a: u32, b: u32, c: u32) -> f32 {
  let h = drive_hash(drive_hash(drive_hash(a) ^ b) ^ c);
  return (f32(h >> 8u) + 1.0) * (1.0 / 16777216.0);
}
// Two unit normal deviates (Box-Muller), the stream's draw j.
fn drive_normal2(m: u32, j: u32) -> vec2<f32> {
  let n = DRIVE.n;
  let u1 = drive_uniform(n.z, n.y, 16u * m + 2u * j);
  let u2 = drive_uniform(n.z, n.y, 16u * m + 2u * j + 1u);
  let r = sqrt(-2.0 * log(u1));
  return r * vec2<f32>(cos(6.28318531 * u2), sin(6.28318531 * u2));
}

fn drive_project(xi: vec3<f32>, k: vec3<f32>) -> vec3<f32> {
  let kh = k / length(k);
  let par = kh * dot(kh, xi);
  let zeta = DRIVE.par.z;
  return zeta * (xi - par) + (1.0 - zeta) * par;
}

fn drive_advance(h: f32) {
  let n = DRIVE.n.x;
  if (n == 0u || h <= 0.0) { return; }
  let f = exp(-h / DRIVE.par.x);
  let s = DRIVE.par.y * sqrt(max(1.0 - f * f, 0.0));
  let flat = DRIVE.n.w == 2u;
  for (var m = 0u; m < n; m++) {
    let k = DRIVE.mode[3u * m].xyz;
    let g0 = drive_normal2(m, 0u);
    let g1 = drive_normal2(m, 1u);
    let g2 = drive_normal2(m, 2u);
    var xr = vec3<f32>(g0.x, g0.y, g1.x);
    var xi = vec3<f32>(g1.y, g2.x, g2.y);
    if (flat) {
      xr.z = 0.0;
      xi.z = 0.0;
    }
    DRIVE.mode[3u * m + 1u] = vec4<f32>(f * DRIVE.mode[3u * m + 1u].xyz + s * drive_project(xr, k), 0.0);
    DRIVE.mode[3u * m + 2u] = vec4<f32>(f * DRIVE.mode[3u * m + 2u].xyz + s * drive_project(xi, k), 0.0);
  }
  DRIVE.n.y += 1u;
}

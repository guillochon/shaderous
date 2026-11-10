// Shared parameter block and boundary-condition index mapping.
// Layout must match packParams() in src/sim.ts.

struct Params {
  bc_lo: vec4<u32>,      // per-axis lower-face boundary: 0 periodic, 1 outflow, 2 reflecting
  bc_hi: vec4<u32>,      // per-axis upper-face boundary
  eos: vec4<f32>,        // (gamma, K, gas constant R, radiation constant a)
  n: u32,                // cells per side
  grav_on: u32,
  grav_bc: u32,          // 0 periodic, 1 isolated
  flags: u32,
  dx: f32,
  cfl: f32,
  G: f32,
  rho_mean: f32,
  rho_floor: f32,
  p_floor: f32,
  eta_dual: f32,         // dual-energy switch: use E - KE when it exceeds eta * E
  t_end: f32,
  dt_max: f32,
  shock_thresh: f32,     // pressure-jump threshold of the shock detector
  jeans_floor: f32,      // 1 enforces the Truelove pressure floor, 0 disables it
  box: f32,              // box side length
}

struct Clock {
  dt: f32,     // timestep of the step being taken
  t: f32,      // time at the start of that step
  tc: f32,     // Kahan compensation for t
  steps: f32,
}

@group(0) @binding(0) var<uniform> P: Params;

fn bc_axis(i: i32, n: i32, lo: u32, hi: u32) -> vec2<i32> {
  if (i < 0) {
    if (lo == 0u) { return vec2<i32>(i + n, 0); }
    if (lo == 1u) { return vec2<i32>(0, 0); }
    return vec2<i32>(-1 - i, 1);
  }
  if (i >= n) {
    if (hi == 0u) { return vec2<i32>(i - n, 0); }
    if (hi == 1u) { return vec2<i32>(n - 1, 0); }
    return vec2<i32>(2 * n - 1 - i, 1);
  }
  return vec2<i32>(i, 0);
}

struct Mapped {
  c: vec3<i32>,       // in-range cell to load
  flip: vec3<f32>,    // -1 on velocity components mirrored by a reflecting face
}

// Every state load goes through this, so there are no ghost cells to fill.
fn bc_map(c: vec3<i32>) -> Mapped {
  let n = i32(P.n);
  let x = bc_axis(c.x, n, P.bc_lo.x, P.bc_hi.x);
  let y = bc_axis(c.y, n, P.bc_lo.y, P.bc_hi.y);
  let z = bc_axis(c.z, n, P.bc_lo.z, P.bc_hi.z);
  return Mapped(
    vec3<i32>(x.x, y.x, z.x),
    vec3<f32>(1.0 - 2.0 * f32(x.y), 1.0 - 2.0 * f32(y.y), 1.0 - 2.0 * f32(z.y)),
  );
}

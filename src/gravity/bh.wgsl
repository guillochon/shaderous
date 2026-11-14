// Point masses: softened gravity on the gas, and a sink around each that takes gas in.
// Requires P. P.bh = (number of masses, 1 if the frame accelerates with the gas or 2 if with
// the masses (gravity/bh_kick.wgsl), 1 if there are any, 1 if there is no Poisson potential to
// add them to).
// Layout matches packMasses() in src/sim.ts.
struct Mass {
  pos: vec4<f32>,     // position relative to the box center; G M in w
  vel: vec4<f32>,     // velocity; softening length in w
  acc: vec4<f32>,     // its acceleration in this frame; sink radius in w
  sink: vec4<f32>,    // x: removal rate at the sink's center, in units of sqrt(G M / r_sink^3)
                      // y: gas mass accreted so far
                      // z: the rate in use this step, per unit time (set as the mass drifts, so
                      //    that it does not change when the mass grows in mid-step)
  accp: vec4<f32>,    // momentum of the gas accreted so far (xyz)
  accl: vec4<f32>,    // angular momentum of that gas about the mass, in its rest frame (xyz)
  // Kahan compensations (gravity/bh_kick.wgsl): what each step's accretion adds is far below
  // float32 resolution of the totals it goes into (a binary's sinks take some 1e-9 of their mass
  // a step), so the rounding is carried here. The true value is the stored one minus these.
  cmass: vec4<f32>,   // x: for G M, y: for the accreted mass
  cvel: vec4<f32>,    // xyz: for the velocity
  cp: vec4<f32>,      // xyz: for accp
  cl: vec4<f32>,      // xyz: for accl
}
struct Masses {
  frame: vec4<f32>,   // acceleration of the frame (gravity/bh_kick.wgsl), or zero
  list: array<Mass, 4>,
}

// (Kahan summation, kahan1 and kahan3, is in hydro/common.wgsl.)

fn mass_count() -> u32 {
  return u32(P.bh.x);
}

// Center of a uniform-grid cell. A 2D run lies in the plane z = 0, where the masses are.
fn cell_pos(c: vec3<i32>) -> vec3<f32> {
  let x = (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
  return vec3<f32>(x.xy, select(x.z, 0.0, P.dims.z == 1u));
}

// Plummer-softened, so a mass may pass through the grid.
fn bh_accel(x: vec3<f32>, m: Mass) -> vec3<f32> {
  let d = x - m.pos.xyz;
  let r2 = dot(d, d) + m.vel.w * m.vel.w;
  return -m.pos.w * d / (r2 * sqrt(r2));
}

// Potential of a mass that has moved on by `ahead` (the half-step extrapolation).
fn bh_phi(x: vec3<f32>, m: Mass, ahead: f32) -> f32 {
  let d = x - (m.pos.xyz + ahead * m.vel.xyz);
  return -m.pos.w / sqrt(dot(d, d) + m.vel.w * m.vel.w);
}

// Fraction of a cell's gas (density rho, at x) that a mass's sink takes in a step of length dt.
// The rate falls smoothly from the center to nothing at the sink radius, and the gas is never
// taken below the density floor. The pass that removes the gas and the reduction that hands it
// to the mass both call this with the same arguments, so nothing is lost between them.
fn sink_fraction(m: Mass, x: vec3<f32>, rho: f32, dt: f32) -> f32 {
  let rs = m.acc.w;
  if (rs <= 0.0 || m.sink.z <= 0.0) { return 0.0; }
  let d = x - m.pos.xyz;
  let q = dot(d, d) / (rs * rs);
  if (q >= 1.0) { return 0.0; }
  let w = (1.0 - q) * (1.0 - q);
  // 1 - exp(-k), which in float32 is 0 for k under 3e-8 and coarsely rounded well above that
  // (a slow sink, or the edge of any sink): the series is used there instead.
  let k = w * m.sink.z * dt;
  let f = select(1.0 - exp(-k), k * (1.0 - k * (0.5 - k / 6.0)), k < 1.0e-3);
  return clamp(min(f, 1.0 - P.rho_floor / rho), 0.0, 1.0);
}

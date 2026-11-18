// Turns the timestep reduction into dt and advances the clock, entirely on the GPU.
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"
#include "gravity/rot.wgsl"
#include "hydro/drive_ou.wgsl"

@group(0) @binding(1) var<storage, read> RED: array<f32>;
@group(0) @binding(2) var<storage, read_write> clock: Clock;
@group(0) @binding(3) var<storage, read> BH: Masses;

// Block grids reduce rate / dx and |g| / dx per cell, since dx varies between blocks.
override PER_DX: bool = false;
// Per-block timesteps (amr/bins_on.wgsl): the step set here is repeated for a cycle of
// substeps, and the step just taken is banked by amr/bins_tick.wgsl instead.
override BANK: bool = true;
override CYCLE: f32 = 1.0;
// Off for the preview of a step not yet taken (previewDt in sim.ts, amr/amr_sim.ts): it works
// out a dt for the readout, and must leave the driving's amplitudes, which are shared and not
// copied as the clock is, where they are. (Advanced by each preview, a run's driving depended on
// how often the page had read its diagnostics before its first step.)
override ADVANCE_DRIVE: bool = true;

// Comoving coordinates: one RK4 step of da/dt = a H(a) (src/cosmo.ts rk4).
fn cosmo_rk4(a: f32, dt: f32) -> f32 {
  let k1 = a * cosmo_hubble(a);
  let a2 = a + 0.5 * dt * k1;
  let k2 = a2 * cosmo_hubble(a2);
  let a3 = a + 0.5 * dt * k2;
  let k3 = a3 * cosmo_hubble(a3);
  let a4 = a + dt * k3;
  let k4 = a4 * cosmo_hubble(a4);
  return a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

@compute @workgroup_size(1)
fn main() {
  // Bank the step that has just been taken (Kahan summation keeps t accurate in float32).
  if (BANK) {
    let prev = clock.dt;
    let k = kahan1(clock.t, clock.tc, prev);
    clock.t = k.x;
    clock.tc = k.y;
    if (prev > 0.0) { clock.steps += 1.0; }
    // The scale factor where that step ended.
    if (cosmo_on()) { clock.a = clock.a1; }
    // A shearing box: its x faces' offset along y moves on by q Omega L dt (Kahan summation, as
    // for t), kept in [0, L).
    if (shear_on() && prev > 0.0) {
      let ks = kahan1(clock.shear, clock.shear_c, prev * shear_jump());
      var s = ks.x;
      var sc = ks.y;
      if (s >= P.box) { s -= P.box; }
      clock.shear = s;
      clock.shear_c = sc;
    }
  }

  // With comoving coordinates the stable step is set by the proper cell size, a dx.
  let dx = select(P.dx, 1.0, PER_DX) * select(1.0, clock.a, cosmo_on());
  var dt = P.cfl * dx / max(RED[0], 1.0e-30);
  // (A spherical run's geometric acceleration is in RED[1] too: hydro/common.wgsl sph_accel. This
  // pass is built without the geometry slot: P.sph's theta step says the run is spherical.)
  if ((P.grav_on != 0u || PER_DX || P.rot.w != 0.0 || P.sph.z > 0.0 || cylindrical()) && RED[1] > 0.0) {
    dt = min(dt, P.cfl * sqrt(dx / RED[1]));
  }
  // Comoving coordinates: the scale factor changes by at most 1% a step.
  if (cosmo_on()) { dt = min(dt, 0.01 / cosmo_hubble(clock.a)); }
  // A rotating frame: at most a tenth of a radian of frame turn a step (63 steps a turn). The
  // Coriolis turn is exact, so this bounds the splitting error, not stability.
  if (rot_on()) { dt = min(dt, 0.1 / length(P.rot.xyz)); }
  // Point masses: the gas knows nothing of two masses passing close to one another. Each pair
  // limits the step to a fraction of the time in which it turns about its common center,
  // sqrt(r^3 / G M), and of the time it takes to cross its own separation, r / v, with r
  // softened as the pull between the two is (gravity/bh_kick.wgsl). At the usual CFL number of
  // 0.8 the fraction is a twentieth: some 125 steps for a circular orbit, and an orbit of
  // eccentricity 0.9 keeps its energy to half a percent over two periods. A plain sink (no
  // mass) neither pulls nor is pulled.
  let nm = mass_count();
  for (var i = 0u; i + 1u < nm; i++) {
    for (var j = i + 1u; j < nm; j++) {
      let a = BH.list[i];
      let b = BH.list[j];
      if (min(a.pos.w, b.pos.w) <= 0.0) { continue; }
      // Two massless particles do not feel one another, however close they pass.
      if (mass_massless(a) && mass_massless(b)) { continue; }
      let d = a.pos.xyz - b.pos.xyz;
      let eps = max(a.vel.w, b.vel.w);
      let r2 = dot(d, d) + eps * eps;
      let r = sqrt(r2);
      let eta = 0.0625 * P.cfl;
      dt = min(dt, eta * sqrt(r2 * r / (a.pos.w + b.pos.w)));
      let v = length(a.vel.xyz - b.vel.xyz);
      if (v > 0.0) { dt = min(dt, eta * r / v); }
    }
  }
  dt = min(dt, P.dt_max);
  let remaining = P.t_end - clock.t;
  if (remaining <= 1.0e-6 * abs(P.t_end)) {
    dt = 0.0;
  } else {
    dt = min(dt, remaining / CYCLE);
  }
  // A movie's next frame time (P.refine.z, 1e30 for none) is a soft stop: the clock holds at the
  // step boundary nearest to it, which is this one once less than half of the step (or cycle)
  // about to be taken lies before it. No step is ever shortened for it, so a recorded run is the
  // unrecorded one. The recorder sees the hold, takes its frame and moves the stop on.
  if (P.refine.z - clock.t < 0.5 * CYCLE * dt) {
    dt = 0.0;
  }
  clock.dt = dt;
  // The scale factor half way through the step and at its end, and the expansion rate half way.
  if (cosmo_on()) {
    clock.ah = cosmo_rk4(clock.a, 0.5 * dt);
    clock.a1 = cosmo_rk4(clock.a, dt);
    clock.hub = cosmo_hubble(clock.ah);
  }
  // Turbulence driving: its amplitudes over the step (or cycle) just set.
  if (ADVANCE_DRIVE) { drive_advance(CYCLE * dt); }
}

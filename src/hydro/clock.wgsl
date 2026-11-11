// Turns the timestep reduction into dt and advances the clock, entirely on the GPU.
#include "hydro/common.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var<storage, read> RED: array<f32>;
@group(0) @binding(2) var<storage, read_write> clock: Clock;
@group(0) @binding(3) var<storage, read> BH: Masses;

// Block grids reduce rate / dx and |g| / dx per cell, since dx varies between blocks.
override PER_DX: bool = false;
// Per-block timesteps (amr/bins_on.wgsl): the step set here is repeated for a cycle of
// substeps, and the step just taken is banked by amr/bins_tick.wgsl instead.
override BANK: bool = true;
override CYCLE: f32 = 1.0;

@compute @workgroup_size(1)
fn main() {
  // Bank the step that has just been taken (Kahan summation keeps t accurate in float32).
  if (BANK) {
    let prev = clock.dt;
    let y = prev - clock.tc;
    let tt = clock.t + y;
    clock.tc = (tt - clock.t) - y;
    clock.t = tt;
    if (prev > 0.0) { clock.steps += 1.0; }
  }

  let dx = select(P.dx, 1.0, PER_DX);
  var dt = P.cfl * dx / max(RED[0], 1.0e-30);
  if ((P.grav_on != 0u || PER_DX) && RED[1] > 0.0) {
    dt = min(dt, P.cfl * sqrt(dx / RED[1]));
  }
  // Point masses: the gas knows nothing of two masses passing close to one another. Each pair
  // limits the step to a fraction of the time in which it turns about its common centre,
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
  clock.dt = dt;
}

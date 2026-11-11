// Turns the timestep reduction into dt and advances the clock, entirely on the GPU.
#include "hydro/common.wgsl"

@group(0) @binding(1) var<storage, read> RED: array<f32>;
@group(0) @binding(2) var<storage, read_write> clock: Clock;

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
  dt = min(dt, P.dt_max);
  let remaining = P.t_end - clock.t;
  if (remaining <= 1.0e-6 * abs(P.t_end)) {
    dt = 0.0;
  } else {
    dt = min(dt, remaining / CYCLE);
  }
  clock.dt = dt;
}

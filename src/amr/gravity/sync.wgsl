// Per-block timesteps, for the blocks whose potential has just been solved at the end of a
// step: the rate at which it changed over that step (see amr/gravity/half_on.wgsl), and the
// potential itself as the reference for the next one. Cells and both ghost layers, which
// the field pass also reads.
//
// FORGET starts the history afresh instead, on a new mesh or a new run: no rate, and the
// potential as it is for reference.
// SEED starts it from the single timestep just taken, when the run goes over to cycles: the
// reference then holds the potential of the step before (AmrGravity.copyPhiToOld), so the rate
// is their difference over the clock's dt, the step just taken.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/bins_on.wgsl"
#include "amr/blk_dt.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var PHI: texture_3d<f32>;
@group(0) @binding(3) var PSOLVED: texture_storage_3d<r32float, read_write>;   // potential at the block's last solve
@group(0) @binding(4) var DPHI: texture_storage_3d<r32float, read_write>;
// The rate over the step before (amr/gravity/half_curve.wgsl); NONE while there is none.
@group(0) @binding(5) var DPHI0: texture_storage_3d<r32float, read_write>;

override FORGET: bool = false;
override SEED: bool = false;
// No rate before: a history that has just begun (amr/gravity/half_curve.wgsl).
const NONE = 3.0e38;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, 2);
  if (!inv.ok) { return; }
  let c = blk_origin(inv.slot) + inv.l;
  let now = textureLoad(PHI, c, 0).x;
  var rate = 0.0;
  var before = NONE;
  if (SEED) {
    if (clock.dt > 0.0) { rate = (now - textureLoad(PSOLVED, c).x) / clock.dt; }
    before = rate;
  } else if (!FORGET) {
    let dt = blk_dt(inv.slot);
    if (dt > 0.0) { rate = (now - textureLoad(PSOLVED, c).x) / dt; }
    // The rate this one replaces; or this one again where it is the first (the one before
    // it marks a history that has just begun), which leaves the half step linear.
    let b = textureLoad(DPHI0, c).x;
    before = select(textureLoad(DPHI, c).x, rate, b >= NONE);
  }
  textureStore(DPHI0, c, vec4<f32>(before, 0.0, 0.0, 0.0));
  textureStore(DPHI, c, vec4<f32>(rate, 0.0, 0.0, 0.0));
  textureStore(PSOLVED, c, vec4<f32>(now, 0.0, 0.0, 0.0));
}

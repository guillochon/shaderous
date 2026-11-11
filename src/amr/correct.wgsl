// Second half of the gravity kick on the block grid, for leaf cells, with the acceleration of
// the updated state (see gravity/correct.wgsl). With per-block timesteps it closes the step a
// block has just finished, whatever its length.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "hydro/fluff.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>
#include "amr/blk_dt.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var N1: texture_3d<f32>;       // state after the hydro update
@group(0) @binding(3) var N2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;        // gravity field of that state
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> BH: Masses;

#include "gravity/bh.wgsl"
#include "gravity/sink.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  g_dx = blk_dx(inv.slot);
  g_dt = blk_dt(inv.slot);
  let c = atl(blk_origin(inv.slot), inv.l);
  var u1 = textureLoad(N1, c, 0);
  var u2 = textureLoad(N2, c, 0);
  if (abs(u2.w) > 0.5) {
    // Solid and fixed cells keep their state.
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    return;
  }
  if (P.bh.z != 0.0) {
    // The point masses' sinks take their gas before the kick, from the state the reduction saw.
    let left = apply_sinks(u1, u2, blk_pos(inv.slot, inv.l), g_dt);
    u1 = left.u1;
    u2 = left.u2;
  }

  let kick = 0.5 * g_dt * u1.x * (P.gext.xyz + textureLoad(G, c, 0).xyz);
  var mom = u1.yzw + kick;
  var E = u2.x;
  var eint = u2.y;
  if ((P.flags & 1u) == 0u) {
    E += dot(kick, u1.yzw + 0.5 * kick) / u1.x;
  } else {
    let en = dual_energy(u1.x, 0.5 * dot(mom, mom) / u1.x, E, eint);
    E = en.x;
    eint = en.y;
  }
  let tame = tame_fluff(u1.x, mom, eint);
  if (tame.changed) {
    mom = tame.mom;
    eint = tame.eint;
    E = 0.5 * dot(mom, mom) / u1.x + eint;
  }
  textureStore(O1, c, vec4<f32>(u1.x, mom));
  textureStore(O2, c, vec4<f32>(E, eint, u2.z, u2.w));
}

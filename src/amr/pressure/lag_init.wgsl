// Implicit pressure for a gas with an energy on blocks: the pressure a run starts from, and
// starts again from after every regrid: the state's own, less the reference (P.ohm.w), for the
// leaves' cells (amr/pressure/lag_fill.wgsl then spreads it to their ghosts). The solved
// pressure is carried from step to step beside the state, not in it, so a regrid loses it; the
// state's stands in for the one step after.
#include "hydro/common.wgsl"
#include <eos>
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(14) var U1N: texture_3d<f32>;
@group(0) @binding(16) var U2N: texture_3d<f32>;
@group(0) @binding(29) var PN: texture_storage_3d<r32float, write>;
// The solve's change of the pressure, which it starts each step from: none yet.
@group(0) @binding(2) var PRS: texture_storage_3d<r32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  g_dx = blk_dx(inv.slot);
  let p = eos_pressure(max(textureLoad(U1N, c, 0).x, P.rho_floor), textureLoad(U2N, c, 0).y);
  textureStore(PN, c, vec4<f32>(p - P.ohm.w, 0.0, 0.0, 0.0));
  textureStore(PRS, c, vec4<f32>(0.0));
}

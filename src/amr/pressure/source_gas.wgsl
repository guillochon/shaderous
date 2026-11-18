// The pressure equation of the implicit step for a gas with an energy on the block grid, for
// the leaves: the uniform grid's (pressure/source_gas.wgsl), for the change d of the pressure
// from the last step's solved pressure q (PBT, carried: the state's own pressure would do in
// exact arithmetic, but its gradient in the predictor is unstable at low Mach number, as the
// prototype found, plans/incompressible.md 4.2.1),
//   alpha d - div(beta grad d) = alpha (p^a - q) - div(v*) / dt + div(beta grad q),
// with the face values of amr/pressure/faces_gas.wgsl, so that the last two terms are the
// divergence of the faces' velocities as they would be with no change: what the solve must
// undo. The solve starts from the last step's change, which in a flow that varies smoothly in
// time is most of this one's (zero after a regrid: amr/pressure/lag_init.wgsl): the cycle over
// the tree's levels cuts the residual by only 0.7 to 0.85 each time on a refined mesh, so where
// it starts matters.
#include "amr/pressure/common.wgsl"
#include <ap_coef>

@group(0) @binding(3) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var TAU: texture_storage_3d<r32float, write>;
@group(0) @binding(25) var PBT: texture_3d<f32>;
@group(0) @binding(26) var PDT: texture_3d<f32>;
@group(0) @binding(14) var U1N: texture_3d<f32>;
@group(0) @binding(16) var U2N: texture_3d<f32>;
@group(0) @binding(27) var FLO: texture_3d<f32>;
@group(0) @binding(28) var FHI: texture_3d<f32>;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return 0.0;
}
#include "amr/pressure/faces_gas.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  textureStore(TAU, c, vec4<f32>(0.0));
  let dt = clock.dt;
  if (!(dt > 0.0) || ap_held(inv.slot, inv.l)) {
    textureStore(RHS, c, vec4<f32>(0.0));
    return;
  }
  let h = blk_dx(inv.slot);
  var div = 0.0;
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    div += ap_gface(inv.slot, inv.l, a, true, 0.0, dt / h, false).pv.y - ap_gface(inv.slot, inv.l, a, false, 0.0, dt / h, false).pv.y;
  }
  // p^a - q: the state's pressure, advected, less the carried one.
  let ps = eos_pressure(max(textureLoad(U1N, c, 0).x, P.rho_floor), textureLoad(U2N, c, 0).y) - P.ohm.w;
  let pa = ps + textureLoad(PDT, c, 0).x - textureLoad(PBT, c, 0).x;
  textureStore(RHS, c, vec4<f32>(ap_alpha_at(inv.slot, inv.l) * pa - div / (h * dt), 0.0, 0.0, 0.0));
}

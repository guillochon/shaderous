// The pressure equation of the implicit step on the block grid, for the leaves (the uniform
// grid's is pressure/source.wgsl): b = alpha p(rho) - div(m*) / dt, with the divergence of the
// face momenta of amr/pressure/faces.wgsl. Also the pressure the solve starts from: the last
// step's, which the state carries in its internal-energy slot (amr/pressure/correct.wgsl), so
// that a regrid moves it with the rest of the state. And the old density, for the correction.
#include "amr/pressure/common.wgsl"

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var TAU: texture_storage_3d<r32float, write>;
@group(0) @binding(13) var RHO0: texture_storage_3d<r32float, write>;
@group(0) @binding(14) var U1N: texture_3d<f32>;   // the state at the start of the step
@group(0) @binding(15) var W1: texture_3d<f32>;    // primitives of the advected state, ghosts filled
@group(0) @binding(16) var U2N: texture_3d<f32>;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return 0.0;
}
// The density at the start of the step, of a leaf's cell.
fn rho_old(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(U1N, atl(blk_origin(slot), l), 0).x;
}
#include "amr/pressure/faces.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  let rho = textureLoad(U1N, c, 0).x;
  let p = eos_pressure(max(rho, P.rho_floor), 0.0);
  textureStore(RHO0, c, vec4<f32>(rho, 0.0, 0.0, 0.0));
  textureStore(PRS, c, vec4<f32>(textureLoad(U2N, c, 0).y, 0.0, 0.0, 0.0));
  textureStore(TAU, c, vec4<f32>(0.0));
  let dt = clock.dt;
  if (!(dt > 0.0) || ap_held(inv.slot, inv.l)) {
    textureStore(RHS, c, vec4<f32>(0.0));
    return;
  }
  var div = 0.0;
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    div += ap_face(inv.slot, inv.l, a, true, 0.0, rho, 0.0, false).y - ap_face(inv.slot, inv.l, a, false, 0.0, rho, 0.0, false).y;
  }
  textureStore(RHS, c, vec4<f32>(ap_alpha() * p - div / (blk_dx(inv.slot) * dt), 0.0, 0.0, 0.0));
}

// The root level and the uniform grid's multigrid (pressure/multigrid.ts), which solves it.
// TO_ROOT: each root block's cells hand their pressure, right-hand side (with TAU) and
// coefficients (beta = 1, alpha) to the uniform textures. Otherwise: they take the solved
// pressure back.
#include "amr/pressure/common.wgsl"
#include <ap_coef>

override TO_ROOT: bool = true;

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var RHS: texture_3d<f32>;
@group(0) @binding(4) var TAU: texture_3d<f32>;
@group(0) @binding(10) var UP: texture_storage_3d<r32float, read_write>;
@group(0) @binding(11) var UB: texture_storage_3d<r32float, write>;
@group(0) @binding(12) var UK: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok || !(clock.dt > 0.0)) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  var g = blk_coord(inv.slot) * BS + inv.l;
  if (is2d()) { g.z = 0; }
  if (TO_ROOT) {
    textureStore(UP, g, vec4<f32>(textureLoad(PRS, c).x, 0.0, 0.0, 0.0));
    textureStore(UB, g, vec4<f32>(textureLoad(RHS, c, 0).x + textureLoad(TAU, c, 0).x, 0.0, 0.0, 0.0));
    // Coefficients: a root leaf's face on an obstacle or an inlet takes no pressure flux, and a
    // cell inside one is no part of the solve.
    var k = vec4<f32>(ap_beta(inv.slot, inv.l, 0, true), ap_beta(inv.slot, inv.l, 1, true), 1.0, ap_alpha_at(inv.slot, inv.l));
    if (!is2d()) { k.z = ap_beta(inv.slot, inv.l, 2, true); }
    if (ap_held(inv.slot, inv.l)) {
      k = vec4<f32>(0.0);
    } else {
      let axes = ap_axes();
      for (var a = 0; a < axes; a++) {
        let kind = ap_nb(inv.slot, inv.l, a, true).kind;
        if (kind == 2 || kind == 4) { k[a] = 0.0; }
      }
    }
    textureStore(UK, g, k);
  } else {
    textureStore(PRS, c, vec4<f32>(textureLoad(UP, g).x, 0.0, 0.0, 0.0));
  }
}

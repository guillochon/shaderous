// After the radiation sweeps on the block grid: radiation/commit.wgsl's, for every leaf cell.
// The gas's side of the exchange, G = (rho e, momentum gained), goes into the state, written to
// the other parity, with the dual-energy synchronization.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(5) var GI: texture_3d<f32>;
@group(0) @binding(7) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var O2: texture_storage_3d<rgba32float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  if (has_solids() && abs(u2.w) > 0.5) {
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    return;
  }
  g_dx = blk_dx(inv.slot);
  comp_use(comp_tex(c));
  let g = textureLoad(GI, c, 0);
  let rho = max(u1.x, P.rho_floor);
  let m1 = u1.yzw + g.yzw;
  let ke0 = 0.5 * dot(u1.yzw, u1.yzw) / rho;
  let ke1 = 0.5 * dot(m1, m1) / rho;
  let eint = max(g.x, eos_eint_floor(rho));
  let E = u2.x + (eint - u2.y) + (ke1 - ke0);
  g_kphi = swirl_ke(rho, m1);
  let s = dual_energy(rho, ke1, E, eint);
  textureStore(O1, c, vec4<f32>(u1.x, m1));
  textureStore(O2, c, vec4<f32>(s.x, s.y, u2.zw));
}

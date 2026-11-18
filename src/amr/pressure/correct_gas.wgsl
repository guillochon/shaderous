// The correction of the implicit step for a gas with an energy on the block grid, for the
// leaves (the uniform grid's is pressure/correct_gas.wgsl): the momentum takes the difference
// of the face pressures, the energy that of p v^ and the internal energy p div(v^), and the
// mass, momentum and energy what the corrected face velocities carry beyond the advective
// fluxes, all with the face values of amr/pressure/faces_gas.wgsl, which a coarse cell beside a
// refined block takes from the fine cells themselves.
#include "amr/pressure/common.wgsl"
#include "hydro/dual_energy.wgsl"
#include <ap_coef>

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read>;
@group(0) @binding(16) var N1: texture_3d<f32>;
@group(0) @binding(17) var N2: texture_3d<f32>;
@group(0) @binding(18) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(19) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(25) var PBT: texture_3d<f32>;
@group(0) @binding(27) var FLO: texture_3d<f32>;
@group(0) @binding(28) var FHI: texture_3d<f32>;
// The new pressure (less the reference), for the next step (amr/pressure/lag_fill.wgsl).
@group(0) @binding(29) var PN: texture_storage_3d<r32float, write>;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PRS, atl(blk_origin(slot), l)).x;
}
#include "amr/pressure/faces_gas.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  g_dx = blk_dx(inv.slot);
  let u1 = textureLoad(N1, c, 0);
  let u2 = textureLoad(N2, c, 0);
  textureStore(PN, c, vec4<f32>(textureLoad(PBT, c, 0).x + textureLoad(PRS, c).x, 0.0, 0.0, 0.0));
  if (abs(u2.w) > 0.5) {
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    return;
  }
  let dtdx = clock.dt / g_dx;
  let dc = textureLoad(PRS, c).x;
  let pc = P.ohm.w + textureLoad(PBT, c, 0).x + dc;
  var mom = u1.yzw;
  var de = 0.0;
  var dvh = 0.0;
  // The field's work, in flux form: its kick times the mass the cell's two faces carry, halved
  // (pressure/correct_gas.wgsl has why).
  var ek = 0.0;
  var c1 = vec4<f32>(0.0);
  var c2 = vec3<f32>(0.0);
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    let fh = ap_gface(inv.slot, inv.l, a, true, dc, dtdx, true);
    let fl = ap_gface(inv.slot, inv.l, a, false, dc, dtdx, true);
    // (The cell's kick is the mean of its two faces'.)
    mom[a] += 0.5 * (fh.k + fl.k) * u1.x - dtdx * (fh.pv.x - fl.pv.x);
    ek += 0.5 * (fh.k * fh.m + fl.k * fl.m);
    de += (P.ohm.w + fh.pv.x) * fh.pv.y - (P.ohm.w + fl.pv.x) * fl.pv.y;
    dvh += fh.pv.y - fl.pv.y;
    c1 += fh.c1 - fl.c1;
    c2 += fh.c2 - fl.c2;
  }
  let rho = max(u1.x - dtdx * c1.x, P.rho_floor);
  mom -= dtdx * c1.yzw;
  let E = u2.x + ek - dtdx * (de + c2.x);
  let eint = u2.y - dtdx * (pc * dvh + c2.y);
  g_kphi = 0.0;
  let en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);
  textureStore(O1, c, vec4<f32>(rho, mom));
  textureStore(O2, c, vec4<f32>(en.x, en.y, u2.z - dtdx * c2.z, u2.w));
}

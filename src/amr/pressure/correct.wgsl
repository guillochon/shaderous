// The correction of the implicit step on the block grid, for the leaves (the uniform grid's is
// pressure/correct.wgsl): with the solved pressure,
//   m' = m* - dt grad p',    rho' = rho - dt div(m*_f - dt d_n p'),
// as differences of the face values of amr/pressure/faces.wgsl. A coarse cell beside a refined
// block takes the fine faces' own values, so mass and momentum are conserved across levels to
// round-off. Reads the advected state (N1, N2: the cell's own momentum and energies), its
// primitives with ghosts (W1), the pressure and the old density; writes the other parity.
//
// Time-centered (amr/pressure/source.wgsl): PRS holds x = theta p' + (1 - theta) p^n, the
// pressure the step's force is from; the faces' mass fluxes mix in the face momenta the step
// started from (OLD); and the state carries p' = (x - (1 - theta) p^n) / theta to the next step.
#include "amr/pressure/common.wgsl"
#include "amr/pressure/theta.wgsl"
@group(0) @binding(34) var OLD: texture_3d<f32>;
#include <ap_mf>
#include <ap_walls>

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read>;
@group(0) @binding(13) var PNL: texture_3d<f32>;
@group(0) @binding(15) var W1: texture_3d<f32>;
@group(0) @binding(16) var N1: texture_3d<f32>;
@group(0) @binding(17) var N2: texture_3d<f32>;
@group(0) @binding(18) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(19) var O2: texture_storage_3d<rgba32float, write>;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PRS, atl(blk_origin(slot), l)).x;
}
// The density at the start of the step, of a leaf's cell (the source pass kept it).
fn rho_old(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(OLD, atl(blk_origin(slot), l), 0).x;
}
fn ap_mix() -> f32 {
  return ap_theta_liquid();
}
fn ap_mn(slo: i32, llo: vec3<i32>, shi: i32, lhi: vec3<i32>, a: i32) -> f32 {
  return 0.5 * (textureLoad(OLD, atl(blk_origin(slo), llo), 0)[a + 1] + textureLoad(OLD, atl(blk_origin(shi), lhi), 0)[a + 1]);
}
#include "amr/pressure/faces.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  let u1 = textureLoad(N1, c, 0);
  let u2 = textureLoad(N2, c, 0);
  // Solid and held cells keep their state.
  if (abs(u2.w) > 0.5) {
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    apwall_store(c);
    return;
  }
  let dtdx = clock.dt / blk_dx(inv.slot);
  let pc = textureLoad(PRS, c).x;
  let rc = rho_old(inv.slot, inv.l);
  let th = ap_theta_liquid();
  let pnew = (pc - (1.0 - th) * textureLoad(PNL, c, 0).x) / th;
  var mom = u1.yzw;
  var dm = 0.0;
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    let fh = ap_face(inv.slot, inv.l, a, true, pc, rc, dtdx, true);
    let fl = ap_face(inv.slot, inv.l, a, false, pc, rc, dtdx, true);
    // (The field's kick on the cell: the mean of its two faces'.)
    mom[a] += clock.dt * P.gext[a] * 0.5 * (fh.z + fl.z) - dtdx * (fh.x - fl.x);
    // Forces on bodies (amr/pressure/pwalls_on.wgsl).
    apwall_face(inv.slot, inv.l, a, true, fh.x);
    apwall_face(inv.slot, inv.l, a, false, fl.x);
    dm += fh.y - fl.y;
    ap_mf_face(inv.slot, inv.l, a, false, fl.y);
    ap_mf_face(inv.slot, inv.l, a, true, fh.y);
  }
  let rho = max(rc - dtdx * dm, P.rho_floor);
  let eint = eos_eint_from_p(rho, eos_pressure(rho, 0.0));
  // The passive scalar rode on the advective mass flux: it keeps its value per unit mass.
  let sc = u2.z * rho / max(u1.x, P.rho_floor);
  textureStore(O1, c, vec4<f32>(rho, mom));
  // The internal-energy slot, which a liquid does not use, carries the solved pressure to the
  // next step: its predictor's gradient and its solve's start (it is regridded and its ghosts
  // filled with the rest of the state).
  textureStore(O2, c, vec4<f32>(0.5 * dot(mom, mom) / rho + eint, pnew, sc, u2.w));
  apwall_store(c);
}

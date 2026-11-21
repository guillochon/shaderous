// The correction of the implicit step for a gas with an energy on the block grid, for the
// leaves (the uniform grid's is pressure/correct_gas.wgsl): the momentum takes the difference
// of the face pressures, the energy that of p v^ and the internal energy p div(v^), and the
// mass, momentum and energy what the corrected face velocities carry beyond the advective
// fluxes, all with the face values of amr/pressure/faces_gas.wgsl, which a coarse cell beside a
// refined block takes from the fine cells themselves.
//
// Time-centered (amr/pressure/theta.wgsl, amr/pressure/source_gas.wgsl): PRS holds e = theta d,
// so the faces' pressures are those of q + e; the faces carry on v^n + chi_f (v^ - v^n), with v^n
// from the state the step started from (OLD) and chi_f the larger of the two cells' switches
// (CHI, bound with <ap_coef>); and the next step's carried pressure is q + e / theta.
#include "amr/pressure/common.wgsl"
#include "hydro/dual_energy.wgsl"
#include <ap_coef>

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, read>;
@group(0) @binding(16) var N1: texture_3d<f32>;
@group(0) @binding(17) var N2: texture_3d<f32>;
@group(0) @binding(18) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(19) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(25) var PBT: texture_3d<f32>;
@group(0) @binding(48) var FLO: texture_3d<f32>;   // (27 is the composition's, hydro/common.wgsl)
@group(0) @binding(49) var FHI: texture_3d<f32>;
// The new pressure (less the reference), for the next step (amr/pressure/lag_fill.wgsl).
@group(0) @binding(29) var PN: texture_storage_3d<r32float, write>;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return textureLoad(PRS, atl(blk_origin(slot), l)).x;
}
// The state the step started from (amr/pressure/source_gas.wgsl kept it).
@group(0) @binding(34) var OLD: texture_3d<f32>;
#include <ap_mf>
#include <ap_walls>
// (v^n, chi_f) of the face between two leaves' cells, lower and upper: as the source formed v^n.
fn ap_vc(slo: i32, llo: vec3<i32>, shi: i32, lhi: vec3<i32>, a: i32) -> vec2<f32> {
  let clo = atl(blk_origin(slo), llo);
  let chi_ = atl(blk_origin(shi), lhi);
  let ulo = textureLoad(OLD, clo, 0);
  let uhi = textureLoad(OLD, chi_, 0);
  let vn = (ulo[a + 1] + uhi[a + 1]) / (max(ulo.x, P.rho_floor) + max(uhi.x, P.rho_floor));
  let chi = max(textureLoad(CHI, clo, 0).x, textureLoad(CHI, chi_, 0).x);
  // A face between two levels carries with the larger of its two cells' theta: on the corrected
  // velocity alone where the step does not resolve sound. There v^n, from the two cells' centers,
  // is not formed as the face's velocity is, from the fine cell and its ghost, and the
  // difference is sound the step must damp: carried on the mean, a gas vortex at Mach 0.01
  // across refinement boundaries kept a density error of 6e-3 (backward Euler: 3e-3). (On the
  // corrected velocity alone everywhere at those faces, the Sod tube's waves, which the step
  // resolves, were smoothed enough there that the mesh stopped following them.)
  if (bt(slo, 0u) != bt(shi, 0u)) {
    let tlo = ap_theta(chi, textureLoad(W2, clo, 0).y, blk_dx(slo));
    let thi = ap_theta(chi, textureLoad(W2, chi_, 0).y, blk_dx(shi));
    return vec2<f32>(vn, max(tlo, thi));
  }
  return vec2<f32>(vn, chi);
}
#include "amr/pressure/faces_gas.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  g_dx = blk_dx(inv.slot);
  // The cell's own composition, where it varies, for the dual energy's floors (hydro/dual_energy.wgsl).
  comp_use(comp_tex(c));
  let u1 = textureLoad(N1, c, 0);
  let u2 = textureLoad(N2, c, 0);
  let th = ap_theta(textureLoad(CHI, c, 0).x, textureLoad(W2, c, 0).y, blk_dx(inv.slot));
  textureStore(PN, c, vec4<f32>(textureLoad(PBT, c, 0).x + textureLoad(PRS, c).x / th, 0.0, 0.0, 0.0));
  if (abs(u2.w) > 0.5) {
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    apwall_store(c);
    return;
  }
  let dtdx = clock.dt / g_dx;
  let dc = textureLoad(PRS, c).x;
  let pc = P.ohm.w + textureLoad(PBT, c, 0).x + dc;
  var mom = u1.yzw;
  var de = 0.0;
  var dvh = 0.0;
  // The field's work, in flux form: the cell's density times half the sum of its two faces'
  // kicks times their velocities (pressure/correct_gas.wgsl has why, and why by its density).
  var ek = 0.0;
  var c1 = vec4<f32>(0.0);
  var c2 = vec3<f32>(0.0);
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    let fh = ap_gface(inv.slot, inv.l, a, true, dc, dtdx, true);
    let fl = ap_gface(inv.slot, inv.l, a, false, dc, dtdx, true);
    // (The cell's kick is the mean of its two faces'.)
    mom[a] += 0.5 * (fh.k + fl.k) * u1.x - dtdx * (fh.pv.x - fl.pv.x);
    // Forces on bodies (amr/pressure/pwalls_on.wgsl).
    apwall_face(inv.slot, inv.l, a, true, fh.pv.x);
    apwall_face(inv.slot, inv.l, a, false, fl.pv.x);
    ek += 0.5 * u1.x * (fh.k * fh.m + fl.k * fl.m);
    de += (P.ohm.w + fh.pv.x) * fh.pv.y - (P.ohm.w + fl.pv.x) * fl.pv.y;
    dvh += fh.pv.y - fl.pv.y;
    c1 += fh.c1 - fl.c1;
    c2 += fh.c2 - fl.c2;
    ap_mf_face(inv.slot, inv.l, a, false, fl.c1.x);
    ap_mf_face(inv.slot, inv.l, a, true, fh.c1.x);
  }
  let rho = max(u1.x - dtdx * c1.x, P.rho_floor);
  mom -= dtdx * c1.yzw;
  let E = u2.x + ek - dtdx * (de + c2.x);
  let eint = u2.y - dtdx * (pc * dvh + c2.y);
  g_kphi = 0.0;
  let en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);
  textureStore(O1, c, vec4<f32>(rho, mom));
  textureStore(O2, c, vec4<f32>(en.x, en.y, u2.z - dtdx * c2.z, u2.w));
  apwall_store(c);
}

// The pressure equation of the implicit step on the block grid, for the leaves (the uniform
// grid's is pressure/source.wgsl): b = alpha p(rho) - div(m*) / dt, with the divergence of the
// face momenta of amr/pressure/faces.wgsl. Also the pressure the solve starts from: the last
// step's, which the state carries in its internal-energy slot (amr/pressure/correct.wgsl), so
// that a regrid moves it with the rest of the state. And the old density, for the correction.
//
// Time-centered (amr/pressure/theta.wgsl, one theta for the mesh): the solve is for
// x = theta p' + (1 - theta) p^n, started from p^n, with
//   b = alpha p(rho) / theta + alpha (1 - theta) p^n / theta^2 - (theta div(m*) + (1 - theta) div(m^n)) / (theta dt)
// and alpha / theta^2 on the diagonal (amr/pressure/coef_liquid.wgsl), as on the uniform grid
// (pressure/source.wgsl). The state the step started from goes to the correction (OLD) for its
// face momenta and its density, and p^n with it (PNL).
#include "amr/pressure/common.wgsl"
#include "amr/pressure/theta.wgsl"
@group(0) @binding(34) var OLD: texture_storage_3d<rgba32float, write>;

@group(0) @binding(2) var PRS: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var TAU: texture_storage_3d<r32float, write>;
@group(0) @binding(13) var PNL: texture_storage_3d<r32float, write>;
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
// (The source's face fluxes are the advected momentum's: amr/pressure/faces.wgsl.)
fn ap_mix() -> f32 {
  return 1.0;
}
// The face momentum the step started from, between two leaves' cells (amr/pressure/correct.wgsl
// forms it from OLD the same way).
fn ap_mn(slo: i32, llo: vec3<i32>, shi: i32, lhi: vec3<i32>, a: i32) -> f32 {
  return 0.5 * (textureLoad(U1N, atl(blk_origin(slo), llo), 0)[a + 1] + textureLoad(U1N, atl(blk_origin(shi), lhi), 0)[a + 1]);
}
#include "amr/pressure/faces.wgsl"

// The face momentum the step started from through face (a, up) of leaf cell l: an inlet's or an
// open face's as it is now (the advected one), none through a wall, the fine faces' mean beside
// a refined block.
fn ap_mn_face(slot: i32, l: vec3<i32>, a: i32, up: bool, rho: f32) -> f32 {
  let n = ap_nb(slot, l, a, up);
  if (n.kind == 2) { return 0.0; }
  if (n.kind == 3 || n.kind == 4) { return ap_face(slot, l, a, up, 0.0, rho, 0.0, false).y; }
  if (n.kind == 0 && n.fine) {
    var acc = 0.0;
    let nf = ap_nfine();
    for (var t = 0; t < nf; t++) {
      let f = ap_fine(n, a, up, t);
      let fm = ap_mark(f.slot, f.l);
      if (fm > 0.5) { continue; }
      var e = vec3<i32>(0);
      e[a] = select(-1, 1, up);
      if (fm < -0.5) { acc += ap_mom(f.slot, f.l, a); continue; }
      if (up) { acc += ap_mn(slot, l, f.slot, f.l, a); } else { acc += ap_mn(f.slot, f.l, slot, l, a); }
    }
    return acc / f32(nf);
  }
  if (up) { return ap_mn(slot, l, n.slot, n.l, a); }
  return ap_mn(n.slot, n.l, slot, l, a);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  let un = textureLoad(U1N, c, 0);
  let rho = un.x;
  let p = eos_pressure(max(rho, P.rho_floor), 0.0);
  let pn = textureLoad(U2N, c, 0).y;
  textureStore(OLD, c, un);
  textureStore(PNL, c, vec4<f32>(pn, 0.0, 0.0, 0.0));
  textureStore(PRS, c, vec4<f32>(pn, 0.0, 0.0, 0.0));
  textureStore(TAU, c, vec4<f32>(0.0));
  let dt = clock.dt;
  if (!(dt > 0.0) || ap_held(inv.slot, inv.l)) {
    textureStore(RHS, c, vec4<f32>(0.0));
    return;
  }
  var div = 0.0;
  var divn = 0.0;
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    div += ap_face(inv.slot, inv.l, a, true, 0.0, rho, 0.0, false).y - ap_face(inv.slot, inv.l, a, false, 0.0, rho, 0.0, false).y;
    divn += ap_mn_face(inv.slot, inv.l, a, true, rho) - ap_mn_face(inv.slot, inv.l, a, false, rho);
  }
  let th = ap_theta_liquid();
  let alpha = ap_alpha();
  let rhs = alpha * (p + (1.0 - th) / th * pn) / th - (th * div + (1.0 - th) * divn) / (th * blk_dx(inv.slot) * dt);
  textureStore(RHS, c, vec4<f32>(rhs, 0.0, 0.0, 0.0));
}

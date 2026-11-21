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
//
// Time-centered (pressure/theta.wgsl, amr/pressure/theta.wgsl): with theta the cell's, the solve
// is for e = theta d, the change of the pressure the step's force is from, as on the uniform grid
// (pressure/source_gas.wgsl): the right-hand side takes theta div(v*) + (1 - theta) div(v^n) for
// the divergence and is divided by theta, and alpha by theta^2 (amr/pressure/coef_gas.wgsl). The
// face velocities the step started from, v^n, are formed from the state it started from, which
// this pass keeps for the correction (OLD): a face's is (m_lo + m_hi) / (rho_lo + rho_hi) of its
// two cells (the coarse cell and the fine one at a refinement face, from either side). And the
// cell's switch chi (CHI), from the divergence of the advected face velocities without the
// field's kicks.
#include "amr/pressure/common.wgsl"
#include "amr/pressure/theta.wgsl"

@group(0) @binding(15) var W1: texture_3d<f32>;
@group(0) @binding(22) var W2: texture_3d<f32>;
@group(0) @binding(34) var OLD: texture_storage_3d<rgba32float, write>;
@group(0) @binding(35) var CHI: texture_storage_3d<r32float, write>;

@group(0) @binding(3) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var TAU: texture_storage_3d<r32float, write>;
@group(0) @binding(25) var PBT: texture_3d<f32>;
@group(0) @binding(26) var PDT: texture_3d<f32>;
@group(0) @binding(14) var U1N: texture_3d<f32>;
@group(0) @binding(16) var U2N: texture_3d<f32>;
@group(0) @binding(48) var FLO: texture_3d<f32>;   // (27 is the composition's, hydro/common.wgsl)
@group(0) @binding(49) var FHI: texture_3d<f32>;

fn prs(slot: i32, l: vec3<i32>) -> f32 {
  return 0.0;
}
// (The source's face velocity is the corrected one: amr/pressure/faces_gas.wgsl.)
fn ap_vc(slo: i32, llo: vec3<i32>, shi: i32, lhi: vec3<i32>, a: i32) -> vec2<f32> {
  return vec2<f32>(0.0, 1.0);
}
#include "amr/pressure/faces_gas.wgsl"

// The face velocity the step started from, between two leaves' cells: one expression, whichever
// side asks (amr/pressure/correct_gas.wgsl forms it from OLD the same way).
fn vn_face(lo: vec4<f32>, hi: vec4<f32>, a: i32) -> f32 {
  return (lo[a + 1] + hi[a + 1]) / (max(lo.x, P.rho_floor) + max(hi.x, P.rho_floor));
}
fn u_old(slot: i32, l: vec3<i32>) -> vec4<f32> {
  return textureLoad(U1N, atl(blk_origin(slot), l), 0);
}
// (v^n of face (a, up) of leaf cell l, its advected velocity without kicks): what the
// correction carries with and what the switch measures. A refined neighbor: the fine faces' mean.
fn ap_vfaces(slot: i32, l: vec3<i32>, a: i32, up: bool) -> vec2<f32> {
  let n = ap_nb(slot, l, a, up);
  var e = vec3<i32>(0);
  e[a] = select(-1, 1, up);
  let own = u_old(slot, l);
  let w = textureLoad(W1, atl(blk_origin(slot), l), 0);
  if (n.kind == 2) { return vec2<f32>(0.0); }
  // An inlet, an open face: as the face velocity is now (amr/pressure/faces_gas.wgsl).
  if (n.kind == 4 || n.kind == 3) {
    let f = ap_gface(slot, l, a, up, 0.0, clock.dt / blk_dx(slot), false).pv.y;
    return vec2<f32>(f, f);
  }
  if (n.kind == 0 && n.fine) {
    var acc = vec2<f32>(0.0);
    let nf = ap_nfine();
    for (var t = 0; t < nf; t++) {
      let f = ap_fine(n, a, up, t);
      if (ap_mark(f.slot, f.l) > 0.5) { continue; }
      let wf = textureLoad(W1, atl(blk_origin(f.slot), f.l), 0);
      let wg = textureLoad(W1, atl(blk_origin(f.slot), f.l - e), 0);
      let va = (abs(wf.x) * wf[a + 1] + abs(wg.x) * wg[a + 1]) / (abs(wf.x) + abs(wg.x));
      if (up) { acc += vec2<f32>(vn_face(own, u_old(f.slot, f.l), a), va); }
      else { acc += vec2<f32>(vn_face(u_old(f.slot, f.l), own, a), va); }
    }
    return acc / f32(nf);
  }
  let q = u_old(n.slot, n.l);
  let wq = textureLoad(W1, atl(blk_origin(slot), l + e), 0);
  let va = (abs(w.x) * w[a + 1] + abs(wq.x) * wq[a + 1]) / (abs(w.x) + abs(wq.x));
  if (up) { return vec2<f32>(vn_face(own, q, a), va); }
  return vec2<f32>(vn_face(q, own, a), va);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let c = atl(blk_origin(inv.slot), inv.l);
  textureStore(TAU, c, vec4<f32>(0.0));
  textureStore(OLD, c, textureLoad(U1N, c, 0));
  textureStore(CHI, c, vec4<f32>(0.5, 0.0, 0.0, 0.0));
  let dt = clock.dt;
  if (!(dt > 0.0) || ap_held(inv.slot, inv.l)) {
    textureStore(RHS, c, vec4<f32>(0.0));
    return;
  }
  let h = blk_dx(inv.slot);
  var div = 0.0;
  var divn = 0.0;
  var diva = 0.0;
  let axes = ap_axes();
  for (var a = 0; a < axes; a++) {
    div += ap_gface(inv.slot, inv.l, a, true, 0.0, dt / h, false).pv.y - ap_gface(inv.slot, inv.l, a, false, 0.0, dt / h, false).pv.y;
    let vh = ap_vfaces(inv.slot, inv.l, a, true);
    let vl = ap_vfaces(inv.slot, inv.l, a, false);
    divn += vh.x - vl.x;
    diva += vh.y - vl.y;
  }
  let chi = ps_chi(dt, diva / h);
  textureStore(CHI, c, vec4<f32>(chi, 0.0, 0.0, 0.0));
  // alpha = 1 / (rho c^2 dt^2) of the advected state (amr/pressure/coef_gas.wgsl, before theta).
  let w1 = textureLoad(W1, c, 0);
  let cs = textureLoad(W2, c, 0).y;
  let alpha = 1.0 / (abs(w1.x) * cs * cs * dt * dt);
  let th = ap_theta(chi, cs, h);
  // p^a - q: the state's pressure, advected, less the carried one.
  comp_use(comp_tex(c));   // (The cell's own composition, where it varies: species/mixture.ts.)
  let ps = eos_pressure(max(textureLoad(U1N, c, 0).x, P.rho_floor), textureLoad(U2N, c, 0).y) - P.ohm.w;
  let pa = ps + textureLoad(PDT, c, 0).x - textureLoad(PBT, c, 0).x;
  textureStore(RHS, c, vec4<f32>((alpha * pa - (th * div + (1.0 - th) * divn) / (h * dt)) / th, 0.0, 0.0, 0.0));
}

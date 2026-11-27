// The conservative update of one cell from its net face fluxes, with the first gravity half
// kick, dual-energy synchronization and the tenuous-gas limits.
// Requires P, hydro/flux.wgsl (Net), hydro/dual_energy.wgsl, hydro/fluff.wgsl, g_dx and g_dt set.
#include <refdev>
struct NewState { u1: vec4<f32>, u2: vec4<f32> }

// A gas stored as its deviation from a reference (hydro/ref_on.wgsl): the cell's reference,
// which the update sets before advance. The fluxes are of the whole gas, and the reference does
// not change, so the deviations take them as they are; the half kick, (rho - rho_ref) g, and the
// half-step pressure's deviation come in as they are too.
var<private> g_aref: vec4<f32> = vec4<f32>(0.0);
fn advance_dev(u1: vec4<f32>, u2: vec4<f32>, net: Net, dw: f32, p_half: f32, kick: vec3<f32>, ekick: f32) -> NewState {
  let r = g_aref;
  let dtdx = g_dt / g_dx;
  var rd = u1.x + dtdx * net.m;
  let rho = max(rd + r.x, P.rho_floor);
  if (rd + r.x < P.rho_floor) { rd = rho - r.x; }
  let mom = u1.yzw + kick + dtdx * net.mom;
  let Ed = u2.x + ekick + dtdx * (net.e + dw);
  let pdv = max(p_half + r.y, P.p_floor) * net.vn;
  let eid = u2.y + dtdx * (net.ei + pdv);
  let sc = u2.z + dtdx * net.s;
  let en = dual_energy_dev(rho, 0.5 * dot(mom, mom) / rho, Ed, eid, r.z);
  return NewState(vec4<f32>(rd, mom), vec4<f32>(en.x, en.y, sc, 0.0));
}

// `dw` is the flux-form gravitational work per unit dt/dx; `kick` the momentum of the half
// kick and `ekick` its kinetic energy (with the flux form, only the point masses' and the
// frame's share: gravity/ext.wgsl);
// `p_half` the cell's half-step pressure.
fn advance(u1: vec4<f32>, u2: vec4<f32>, net: Net, dw: f32, p_half: f32, kick: vec3<f32>, ekick: f32) -> NewState {
  if (REF_DEV) { return advance_dev(u1, u2, net, dw, p_half, kick, ekick); }
  let dtdx = g_dt / g_dx;
  let rho = max(u1.x + dtdx * net.m, P.rho_floor);
  let mom = u1.yzw + kick + dtdx * net.mom;
  let E = u2.x + ekick + dtdx * (net.e + dw);
  // Internal energy: advection plus -p div(v), with the half-step pressure and face velocities.
  // (With implicit pressure the work is the correction's: pressure/correct_gas.wgsl.)
  let pdv = select(max(p_half, P.p_floor) * net.vn, 0.0, PSPLIT);
  let eint = u2.y + dtdx * (net.ei + pdv);
  let sc = u2.z + dtdx * net.s;

  g_kphi = swirl_ke(rho, mom);
  var en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);
  var mout = mom;
  let tame = tame_fluff(rho, mom, en.y);
  if (tame.changed) {
    mout = tame.mom;
    en = vec2<f32>(0.5 * dot(mout, mout) / rho + tame.eint, tame.eint);
  }
  return NewState(vec4<f32>(rho, mout), vec4<f32>(en.x, en.y, sc, 0.0));
}

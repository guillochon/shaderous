// The conservative update of one cell from its net face fluxes, with the first gravity half
// kick, dual-energy synchronization and the tenuous-gas limits.
// Requires P, hydro/flux.wgsl (Net), hydro/dual_energy.wgsl, hydro/fluff.wgsl, g_dx and g_dt set.
struct NewState { u1: vec4<f32>, u2: vec4<f32> }

// `dw` is the flux-form gravitational work per unit dt/dx; `kick` the momentum of the half
// kick and `ekick` its kinetic energy (zero when the flux form carries the energy);
// `p_half` the cell's half-step pressure.
fn advance(u1: vec4<f32>, u2: vec4<f32>, net: Net, dw: f32, p_half: f32, kick: vec3<f32>, ekick: f32) -> NewState {
  let dtdx = g_dt / g_dx;
  let rho = max(u1.x + dtdx * net.m, P.rho_floor);
  let mom = u1.yzw + kick + dtdx * net.mom;
  let E = u2.x + ekick + dtdx * (net.e + dw);
  // Internal energy: advection plus -p div(v), with the half-step pressure and face velocities.
  let eint = u2.y + dtdx * (net.ei + max(p_half, P.p_floor) * net.vn);
  let sc = u2.z + dtdx * net.s;

  var en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);
  var mout = mom;
  let tame = tame_fluff(rho, mom, en.y);
  if (tame.changed) {
    mout = tame.mom;
    en = vec2<f32>(0.5 * dot(mout, mout) / rho + tame.eint, tame.eint);
  }
  return NewState(vec4<f32>(rho, mout), vec4<f32>(en.x, en.y, sc, 0.0));
}

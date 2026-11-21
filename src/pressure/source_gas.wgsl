// The pressure equation of the implicit step for a gas with an energy (Kwatra et al. 2009;
// plans/incompressible.md 4.1 and 4.2.1),
//   alpha p - div(beta grad p) = alpha p^a - div(v*) / dt,
//   alpha = 1 / (rho c^2 dt^2) from the state at the start of the step,  beta = 1 / rho_f,
// solved for the change d = p - q from the last step's solved pressure q (PF, which the
// correction keeps, less the reference pressure P.ohm.w):
//   alpha d - div(beta grad d) = alpha (p^a - q) - div(v*) / dt + div(beta grad q).
// The right-hand side is then made of differences of neighboring values, each good to single
// precision of its own size, and the solve starts from zero. A stratified column in discrete
// balance has a right-hand side of exactly zero, whatever its range of pressure (plan 4.7): a
// solve for the whole pressure, to a tolerance set by the densest gas, lost the thin top of a
// column spanning 9e6 in pressure. The change is from the solved pressure, not the state's:
// at Mach 0.001 the state's is single-precision noise as large as the flow's own pressure, and
// its Laplacian blew the run up. (The state's enters only through alpha, which is small there.)
// The pass hands q on to the correction (PNO).
// p^a is the pressure advected over the step, by first-order upwind differences of the cells'
// own pressures and velocities (the prototype found it as good as the flux form). A face's
// velocity is the density-weighted mean of its two cells' advected ones, and its density their
// mean: symmetric in the two, so both form the same bits (pressure/correct_gas.wgsl does too).
// Reads the state at the start of the step (U1N, U2N) and the advected one (U1S, U2S).
//
// Time-centered (pressure/theta.wgsl): with theta the cell's, the solve is for e = theta d, the
// change of the pressure the step's force is from, p_theta = q + e:
//   (alpha / theta^2) e - div(beta grad e)
//     = [alpha (p^a - q) - (theta (div(v*) - dt div(beta grad q)) + (1 - theta) div(v^n)) / dt] / theta,
// with v^n the face velocities the step started from. The operator stays symmetric with theta
// varying from cell to cell. The correction takes the force from p_theta and hands on
// p' = q + e / theta. v^n goes to the correction too, in PNO with q: it carries the step's mass on
// the mean of v^n and the corrected face velocity.
#include "hydro/common.wgsl"
#include <eos>
#include "pressure/theta_cell.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1N: texture_3d<f32>;
@group(0) @binding(3) var U1S: texture_3d<f32>;
@group(0) @binding(4) var U2S: texture_3d<f32>;
@group(0) @binding(5) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var COEF: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var PRS: texture_storage_3d<r32float, write>;
@group(0) @binding(10) var PNO: texture_storage_3d<rgba32float, write>;
// The cell's compression switch chi (pressure/theta.wgsl), for the correction.
@group(0) @binding(7) var CHI: texture_storage_3d<r32float, write>;
@group(0) @binding(11) var PF: texture_3d<f32>;
@group(0) @binding(12) var G: texture_3d<f32>;   // the gravity field: (acceleration, half-step potential)

#include "gravity/ext_bind.wgsl"
#include "pressure/kick.wgsl"
@group(0) @binding(9) var U2N: texture_3d<f32>;

#include "pressure/faces.wgsl"

// The pressure of the state at the start of the step in cell c.
fn p_old(c: vec3<i32>) -> f32 {
  comp_use(comp_tex(c));   // (The cell's own composition, where it varies: species/mixture.ts.)
  return eos_pressure(max(textureLoad(U1N, c, 0).x, P.rho_floor), textureLoad(U2N, c, 0).y);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  g_dx = P.dx;
  let p = p_old(c);
  // The solve starts from no change.
  textureStore(PRS, c, vec4<f32>(0.0));
  let pf = textureLoad(PF, c, 0).x;
  textureStore(CHI, c, vec4<f32>(0.5, 0.0, 0.0, 0.0));
  if (abs(textureLoad(U2S, c, 0).w) > 0.5) {
    textureStore(PNO, c, vec4<f32>(pf, 0.0, 0.0, 0.0));
    textureStore(RHS, c, vec4<f32>(0.0));
    textureStore(COEF, c, vec4<f32>(0.0));
    return;
  }
  let dt = clock.dt;
  if (!(dt > 0.0)) {
    textureStore(PNO, c, vec4<f32>(pf, 0.0, 0.0, 0.0));
    textureStore(RHS, c, vec4<f32>(0.0));
    textureStore(COEF, c, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let un = textureLoad(U1N, c, 0);
  let rho = max(un.x, P.rho_floor);
  let v = un.yzw / rho;
  comp_use(comp_tex(c));
  // (At the pressure floor, as the explicit primitives are: gas at zero pressure has no sound
  // speed, and 1 / (rho c^2) below would be infinite. A liquid's floor is -1e30, src/sim.ts.)
  let cs = eos_sound_speed(rho, max(p, P.p_floor));
  let us = textureLoad(U1S, c, 0);
  var adv = 0.0;
  var div = 0.0;
  // The divergence of the face velocities the step started from, and those of the upper faces.
  var divn = 0.0;
  var vn = vec3<f32>(0.0);
  // And of the advected ones without the field's kicks, for the switch.
  var diva = 0.0;
  // h^2 div(beta grad p^n): the faces' beta (p_far - p), which a face in balance with the
  // field's kick cancels in `div`.
  var lap = 0.0;
  var beta = vec3<f32>(0.0);
  let axes = select(3, 2, P.dims.z == 1u);
  for (var a = 0; a < axes; a++) {
    let hi = pface(c, a, true);
    let lo = pface(c, a, false);
    // The advected pressure: the cell's own beyond a wall, an inlet or an open face.
    var ph = p;
    var pl = p;
    if (hi.kind == FK_GAS) { ph = p_old(hi.q); }
    if (lo.kind == FK_GAS) { pl = p_old(lo.q); }
    adv += max(v[a], 0.0) * (p - pl) + min(v[a], 0.0) * (ph - p);
    var fh = 0.0;
    var fl = 0.0;
    // Gravity kicks the face velocities (pressure/kick.wgsl).
    let kg = dt * cell_accel(c)[a];
    if (hi.kind == FK_GAS) {
      let q = textureLoad(U1S, hi.q, 0);
      let va = (us[a + 1] + q[a + 1]) / (us.x + q.x);
      fh = va + face_kick(c, hi.q, a, dt);
      diva += va;
      beta[a] = 2.0 / (us.x + q.x);
      lap += beta[a] * (textureLoad(PF, hi.q, 0).x - pf);
      let qn = textureLoad(U1N, hi.q, 0);
      vn[a] = (un[a + 1] + qn[a + 1]) / (rho + max(qn.x, P.rho_floor));
    }
    if (hi.kind == FK_INLET) { let q = textureLoad(U1S, hi.q, 0); fh = q[a + 1] / q.x; }
    // An open face: the gas leaves at its own velocity and pressure (no gradient across it), so
    // the solve sees it as closed and the divergence takes the advected velocity there.
    if (hi.kind == FK_OPEN) { fh = open_mom(us[a + 1] + kg * us.x, true, hi.only) / us.x; }
    if (hi.kind != FK_GAS) {
      vn[a] = fh;
      diva += fh;
    }
    divn += vn[a];
    if (lo.kind == FK_GAS) {
      let q = textureLoad(U1S, lo.q, 0);
      let va = (q[a + 1] + us[a + 1]) / (q.x + us.x);
      fl = va + face_kick(lo.q, c, a, dt);
      diva -= va;
      lap += 2.0 / (q.x + us.x) * (textureLoad(PF, lo.q, 0).x - pf);
      let qn = textureLoad(U1N, lo.q, 0);
      divn -= (qn[a + 1] + un[a + 1]) / (max(qn.x, P.rho_floor) + rho);
    }
    if (lo.kind == FK_INLET) { let q = textureLoad(U1S, lo.q, 0); fl = q[a + 1] / q.x; }
    if (lo.kind == FK_OPEN) { fl = open_mom(us[a + 1] + kg * us.x, false, lo.only) / us.x; }
    // (An inlet or an open face takes its velocity as it is now in place of v^n; a wall has none.)
    if (lo.kind != FK_GAS) {
      divn -= fl;
      diva -= fl;
    }
    div += fh - fl;
  }
  let alpha = 1.0 / (rho * cs * cs * dt * dt);
  let pa = p - dt / P.dx * adv;
  let chi = ps_chi(dt, diva / P.dx);
  textureStore(CHI, c, vec4<f32>(chi, 0.0, 0.0, 0.0));
  let th = cell_theta(c, dt, chi);
  let rhs = alpha * ((pa - P.ohm.w) - pf) - (th * (div - dt / P.dx * lap) + (1.0 - th) * divn) / (P.dx * dt);
  textureStore(RHS, c, vec4<f32>(rhs / th, 0.0, 0.0, 0.0));
  textureStore(COEF, c, vec4<f32>(beta, alpha / (th * th)));
  textureStore(PNO, c, vec4<f32>(pf, vn));
}

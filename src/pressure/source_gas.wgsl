// The pressure equation of the implicit step for a gas with an energy (Kwatra et al. 2009;
// plans/incompressible.md 4.1 and 4.2.1), for p' = p - p_ref (P.ohm.w, the mean pressure the run
// started with, so that single precision holds the part that varies):
//   alpha p' - div(beta grad p') = alpha (p^a - p_ref) - div(v*) / dt,
//   alpha = 1 / (rho c^2 dt^2) from the state at the start of the step,  beta = 1 / rho_f.
// p^a is the pressure advected over the step, by first-order upwind differences of the cells'
// own pressures and velocities (the prototype found it as good as the flux form). A face's
// velocity is the density-weighted mean of its two cells' advected ones, and its density their
// mean: symmetric in the two, so both form the same bits (pressure/correct_gas.wgsl does too).
// Reads the state at the start of the step (U1N, U2N) and the advected one (U1S, U2S).
#include "hydro/common.wgsl"
#include <eos>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1N: texture_3d<f32>;
@group(0) @binding(3) var U1S: texture_3d<f32>;
@group(0) @binding(4) var U2S: texture_3d<f32>;
@group(0) @binding(5) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var COEF: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var PRS: texture_3d<f32>;
@group(0) @binding(9) var U2N: texture_3d<f32>;

#include "pressure/faces.wgsl"

// The pressure of the state at the start of the step in cell c.
fn p_old(c: vec3<i32>) -> f32 {
  return eos_pressure(max(textureLoad(U1N, c, 0).x, P.rho_floor), textureLoad(U2N, c, 0).y);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  g_dx = P.dx;
  if (abs(textureLoad(U2S, c, 0).w) > 0.5) {
    textureStore(RHS, c, vec4<f32>(0.0));
    textureStore(COEF, c, vec4<f32>(0.0));
    return;
  }
  let dt = clock.dt;
  if (!(dt > 0.0)) {
    textureStore(RHS, c, vec4<f32>(textureLoad(PRS, c, 0).x, 0.0, 0.0, 0.0));
    textureStore(COEF, c, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let un = textureLoad(U1N, c, 0);
  let rho = max(un.x, P.rho_floor);
  let v = un.yzw / rho;
  let p = p_old(c);
  let cs = eos_sound_speed(rho, p);
  let us = textureLoad(U1S, c, 0);
  var adv = 0.0;
  var div = 0.0;
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
    if (hi.kind == FK_GAS) {
      let q = textureLoad(U1S, hi.q, 0);
      fh = (us[a + 1] + q[a + 1]) / (us.x + q.x);
      beta[a] = 2.0 / (us.x + q.x);
    }
    if (hi.kind == FK_INLET) { let q = textureLoad(U1S, hi.q, 0); fh = q[a + 1] / q.x; }
    // An open face: the gas leaves at its own velocity and pressure (no gradient across it), so
    // the solve sees it as closed and the divergence takes the advected velocity there.
    if (hi.kind == FK_OPEN) { fh = open_mom(us[a + 1], true, hi.only) / us.x; }
    if (lo.kind == FK_GAS) {
      let q = textureLoad(U1S, lo.q, 0);
      fl = (q[a + 1] + us[a + 1]) / (q.x + us.x);
    }
    if (lo.kind == FK_INLET) { let q = textureLoad(U1S, lo.q, 0); fl = q[a + 1] / q.x; }
    if (lo.kind == FK_OPEN) { fl = open_mom(us[a + 1], false, lo.only) / us.x; }
    div += fh - fl;
  }
  let alpha = 1.0 / (rho * cs * cs * dt * dt);
  let pa = p - dt / P.dx * adv;
  textureStore(RHS, c, vec4<f32>(alpha * (pa - P.ohm.w) - div / (P.dx * dt), 0.0, 0.0, 0.0));
  textureStore(COEF, c, vec4<f32>(beta, alpha));
}

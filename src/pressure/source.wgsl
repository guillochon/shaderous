// The pressure equation of the implicit step, for a gas whose pressure follows its density (the
// liquid, eos/liquid.wgsl): backward Euler for sound,
//   rho' = rho - dt div(m'),    m' = m* - dt grad p',    p' = p(rho) + c^2 (rho' - rho),
// with m* the momentum the explicit advection (and viscosity, driving) left. Eliminating rho'
// and m':
//   p' / (c^2 dt^2) - lap p' = p(rho) / (c^2 dt^2) - div(m*) / dt.
// Reads the state at the start of the step (U1N) and the advected one (U1S, U2S); writes the
// right-hand side, the finest level's coefficients (pressure/common.wgsl) and the old density,
// which the correction (pressure/correct.wgsl) needs after the old state is overwritten.
// The divergence is of face momenta, each the mean of its two cells': the correction forms the
// same ones.
#include "hydro/common.wgsl"
#include <eos>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1N: texture_3d<f32>;
@group(0) @binding(3) var U1S: texture_3d<f32>;
@group(0) @binding(4) var U2S: texture_3d<f32>;
@group(0) @binding(5) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var COEF: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var RHO0: texture_storage_3d<r32float, write>;
@group(0) @binding(8) var PRS: texture_3d<f32>;

#include "pressure/faces.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let rho = textureLoad(U1N, c, 0).x;
  textureStore(RHO0, c, vec4<f32>(rho, 0.0, 0.0, 0.0));
  // A solid or held cell is not part of the solve.
  if (abs(textureLoad(U2S, c, 0).w) > 0.5) {
    textureStore(RHS, c, vec4<f32>(0.0));
    textureStore(COEF, c, vec4<f32>(0.0));
    return;
  }
  let dt = clock.dt;
  // No step (the run is holding at a stop): the pressure stays as it is.
  if (!(dt > 0.0)) {
    textureStore(RHS, c, vec4<f32>(textureLoad(PRS, c, 0).x, 0.0, 0.0, 0.0));
    textureStore(COEF, c, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let ms = textureLoad(U1S, c, 0).yzw;
  var div = 0.0;
  var beta = vec3<f32>(0.0);
  let axes = select(3, 2, P.dims.z == 1u);
  for (var a = 0; a < axes; a++) {
    let hi = pface(c, a, true);
    let lo = pface(c, a, false);
    var fh = 0.0;
    var fl = 0.0;
    // A uniform field's kick is the faces': dt g times the mean of the two cells' densities,
    // which is what the pressure difference across the face balances (pressure/correct.wgsl).
    let kg = dt * P.gext[a];
    if (hi.kind == FK_GAS) { fh = 0.5 * (ms[a] + textureLoad(U1S, hi.q, 0)[a + 1]) + kg * 0.5 * (rho + textureLoad(U1N, hi.q, 0).x); }
    if (hi.kind == FK_INLET) { fh = textureLoad(U1S, hi.q, 0)[a + 1]; }
    if (hi.kind == FK_OPEN) { fh = open_mom(ms[a] + kg * rho, true, hi.only); }
    if (lo.kind == FK_GAS) { fl = 0.5 * (textureLoad(U1S, lo.q, 0)[a + 1] + ms[a]) + kg * 0.5 * (textureLoad(U1N, lo.q, 0).x + rho); }
    if (lo.kind == FK_INLET) { fl = textureLoad(U1S, lo.q, 0)[a + 1]; }
    if (lo.kind == FK_OPEN) { fl = open_mom(ms[a] + kg * rho, false, lo.only); }
    div += fh - fl;
    beta[a] = select(0.0, 1.0, hi.kind == FK_GAS || hi.kind == FK_OPEN);
  }
  let p = eos_pressure(rho, 0.0);
  let cs = eos_sound_speed(rho, p);
  let alpha = 1.0 / (cs * cs * dt * dt);
  textureStore(RHS, c, vec4<f32>(alpha * p - div / (P.dx * dt), 0.0, 0.0, 0.0));
  textureStore(COEF, c, vec4<f32>(beta, alpha));
}

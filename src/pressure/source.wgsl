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
//
// Time-centered (pressure/theta.wgsl): the step's mass flux is theta m'_f + (1 - theta) m^n_f
// and its force that of x = theta p' + (1 - theta) p^n, p^n the last step's p' (PL). The solve
// is for x:
//   (alpha / theta^2) x - lap x = alpha p(rho) / theta + alpha (1 - theta) p^n / theta^2
//                                 - (theta div(m*) + (1 - theta) div(m^n)) / (theta dt),
// started from p^n. p^n and the face momenta the step started from, m^n_f on the cell's upper
// faces, go to the correction (PNO), which hands on p' = (x - (1 - theta) p^n) / theta. The
// liquid's sound speed is one constant, so theta is the same in every cell.
#include "hydro/common.wgsl"
#include <eos>
#include "pressure/theta_cell.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1N: texture_3d<f32>;
@group(0) @binding(3) var U1S: texture_3d<f32>;
@group(0) @binding(4) var U2S: texture_3d<f32>;
@group(0) @binding(5) var RHS: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var COEF: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var RHO0: texture_storage_3d<r32float, write>;
@group(0) @binding(8) var PRS: texture_storage_3d<r32float, write>;
@group(0) @binding(10) var PNO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(11) var PL: texture_3d<f32>;

#include "pressure/faces.wgsl"
#include "hydro/moving_flags.wgsl"
#include "hydro/cut_flags.wgsl"

// A moving body's wall (hydro/moving.ts): the momentum through the face that the body's motion
// imposes, the cell's density times the wall's velocity along the axis (its solid cell holds that
// velocity in its momentum). The solve then sees the body push and pull the liquid; no mass
// crosses the face in the correction. 0 for a wall at rest, or of the box.
// (With fractions the swept volume carries the push instead: the cell's gas fraction falls.)
fn wall_mom(c: vec3<i32>, a: i32, up: bool, rho: f32) -> f32 {
  if (!MOVING || MOVING_FRACTIONS) { return 0.0; }
  var q = c;
  q[a] += select(-1, 1, up);
  if (q[a] < 0 || q[a] >= i32(P.dims[a])) { return 0.0; }
  if (textureLoad(U2S, q, 0).w < 0.5) { return 0.0; }
  let u = textureLoad(U1S, q, 0);
  return rho * u[a + 1] / max(u.x, P.rho_floor);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let un = textureLoad(U1N, c, 0);
  let rho = un.x;
  textureStore(RHO0, c, vec4<f32>(rho, 0.0, 0.0, 0.0));
  // The solve starts from the last step's pressure.
  let pn = textureLoad(PL, c, 0).x;
  textureStore(PRS, c, vec4<f32>(pn, 0.0, 0.0, 0.0));
  // A solid or held cell is not part of the solve.
  if (abs(textureLoad(U2S, c, 0).w) > 0.5) {
    textureStore(PNO, c, vec4<f32>(pn, 0.0, 0.0, 0.0));
    textureStore(RHS, c, vec4<f32>(0.0));
    textureStore(COEF, c, vec4<f32>(0.0));
    return;
  }
  let dt = clock.dt;
  // No step (the run is holding at a stop): the pressure stays as it is.
  if (!(dt > 0.0)) {
    textureStore(PNO, c, vec4<f32>(pn, 0.0, 0.0, 0.0));
    textureStore(RHS, c, vec4<f32>(pn, 0.0, 0.0, 0.0));
    textureStore(COEF, c, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }
  let ms = textureLoad(U1S, c, 0).yzw;
  // Partly filled cells (hydro/cut.ts): the cell's gas share and its upper faces' open shares,
  // which weigh the face momenta and the coefficients; the state holds the gas's own density.
  let gf = select(gas_fractions(c), cut_at(c), CUT);
  var div = 0.0;
  // The divergence of the face momenta the step started from, and those of the upper faces.
  var divn = 0.0;
  var mn = vec3<f32>(0.0);
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
    if (hi.kind == FK_WALL) { fh = wall_mom(c, a, true, rho); }
    if (lo.kind == FK_WALL) { fl = wall_mom(c, a, false, rho); }
    // (An inlet or an open face takes its momentum as it is now in place of m^n; a wall has none,
    // or a moving one's.)
    var fhn = fh;
    var fln = fl;
    if (hi.kind == FK_GAS) { fhn = 0.5 * (un[a + 1] + textureLoad(U1N, hi.q, 0)[a + 1]); }
    if (lo.kind == FK_GAS) { fln = 0.5 * (textureLoad(U1N, lo.q, 0)[a + 1] + un[a + 1]); }
    mn[a] = fhn;
    let ah = select(1.0, gf[a + 1], CUT);
    let al = select(1.0, cut_lo(c, a), CUT);
    div += ah * fh - al * fl;
    divn += ah * fhn - al * fln;
    beta[a] = select(0.0, 1.0, hi.kind == FK_GAS || hi.kind == FK_OPEN) * gf[a + 1];
  }
  // Near a moving body (plans/wind-tunnel.md 8.1's cheap form): the cell holds gas in a fraction
  // phi of its volume (at the step's end), so its true density is rho / phi, and its
  // compressibility term is phi's share: as a body sweeps through the cell its pressure rises
  // and the solve pushes the liquid out, before the cell becomes the body's. (1 elsewhere.)
  let phi = select(max(gf.x, 0.02), gf.x, CUT);
  let p = eos_pressure(select(rho / phi, rho, CUT), 0.0);
  let cs = eos_sound_speed(rho, p);
  let alpha = phi / (cs * cs * dt * dt);
  let th = cell_theta(c, dt, 0.5);
  let rhs = alpha * (p + (1.0 - th) / th * pn) / th - (th * div + (1.0 - th) * divn) / (th * P.dx * dt);
  textureStore(RHS, c, vec4<f32>(rhs, 0.0, 0.0, 0.0));
  textureStore(COEF, c, vec4<f32>(beta, alpha / (th * th)));
  textureStore(PNO, c, vec4<f32>(pn, mn));
}

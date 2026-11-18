// The correction of the implicit step for a gas with an energy (pressure/source_gas.wgsl). With
// the solved pressure, on each face
//   p_f = (rho_R p_L + rho_L p_R) / (rho_L + rho_R),    v^_f = v*_f - dt (p_R - p_L) / (rho_f dx),
// and the cell takes -dt grad p from the p_f, -dt div(p v^) into E and -dt p div(v^) into the
// internal energy: all differences of face values (Kwatra et al. 2009).
//
// And what the published scheme leaves out (plans/incompressible.md 4.2.1, 3a): advection
// carried the mass, momentum and energy on the face velocities of the predictor, which are
// divergence-free only to first order in the step, and the density of a gas has nothing to
// push such an error back. So each face also moves (v^_f - u_f) times the upwind advected
// state, u_f being the velocity that carried the advective flux (FU, kept by the update): the
// density and the energy then see the divergence the solve left. In the prototype that takes a
// vortex's density error from 7% to 0.07% and removes the need for a ceiling on the step.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1S: texture_3d<f32>;
@group(0) @binding(3) var U2S: texture_3d<f32>;
@group(0) @binding(4) var PRS: texture_3d<f32>;
@group(0) @binding(6) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var FU: texture_3d<f32>;

#include "pressure/faces.wgsl"

// What a face between a lower and an upper gas cell does: (its pressure, its corrected velocity)
// and, in `carry`, what the corrected velocity carries beyond the advective flux: (mass,
// momentum) and (energy, internal energy, scalar). One piece of code, whichever cell asks.
struct GFace { pv: vec2<f32>, c1: vec4<f32>, c2: vec3<f32> }

fn gas_face(lo: vec3<i32>, hi: vec3<i32>, a: i32, dtdx: f32) -> GFace {
  let ul = textureLoad(U1S, lo, 0);
  let ur = textureLoad(U1S, hi, 0);
  let pl = textureLoad(PRS, lo, 0).x;
  let pr = textureLoad(PRS, hi, 0).x;
  let sum = ul.x + ur.x;
  let pf = P.ohm.w + (ur.x * pl + ul.x * pr) / sum;
  let vh = (ul[a + 1] + ur[a + 1]) / sum - dtdx * (2.0 / sum) * (pr - pl);
  let du = vh - textureLoad(FU, lo, 0)[a];
  var d1 = ul;
  var d2 = textureLoad(U2S, lo, 0).xyz;
  if (du < 0.0) {
    d1 = ur;
    d2 = textureLoad(U2S, hi, 0).xyz;
  }
  return GFace(vec2<f32>(pf, vh), du * d1, du * d2);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  g_dx = P.dx;
  let u1 = textureLoad(U1S, c, 0);
  let u2 = textureLoad(U2S, c, 0);
  if (abs(u2.w) > 0.5) {
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    return;
  }
  let dtdx = clock.dt / P.dx;
  let pc = P.ohm.w + textureLoad(PRS, c, 0).x;
  var mom = u1.yzw;
  var de = 0.0;
  var dvh = 0.0;
  var c1 = vec4<f32>(0.0);
  var c2 = vec3<f32>(0.0);
  let axes = select(3, 2, P.dims.z == 1u);
  for (var a = 0; a < axes; a++) {
    let hi = pface(c, a, true);
    let lo = pface(c, a, false);
    // A wall: the cell's pressure, and nothing through it.
    var fh = vec2<f32>(pc, 0.0);
    var fl = vec2<f32>(pc, 0.0);
    if (hi.kind == FK_GAS) {
      let f = gas_face(c, hi.q, a, dtdx);
      fh = f.pv;
      c1 += f.c1;
      c2 += f.c2;
    }
    if (hi.kind == FK_INLET) { let q = textureLoad(U1S, hi.q, 0); fh.y = q[a + 1] / q.x; }
    // An open face: the cell's own pressure and advected velocity (pressure/source_gas.wgsl).
    if (hi.kind == FK_OPEN) { fh.y = open_mom(u1[a + 1], true, hi.only) / u1.x; }
    if (lo.kind == FK_GAS) {
      let f = gas_face(lo.q, c, a, dtdx);
      fl = f.pv;
      c1 -= f.c1;
      c2 -= f.c2;
    }
    if (lo.kind == FK_INLET) { let q = textureLoad(U1S, lo.q, 0); fl.y = q[a + 1] / q.x; }
    if (lo.kind == FK_OPEN) { fl.y = open_mom(u1[a + 1], false, lo.only) / u1.x; }
    mom[a] -= dtdx * (fh.x - fl.x);
    de += fh.x * fh.y - fl.x * fl.y;
    dvh += fh.y - fl.y;
  }
  let rho = max(u1.x - dtdx * c1.x, P.rho_floor);
  mom -= dtdx * c1.yzw;
  let E = u2.x - dtdx * (de + c2.x);
  let eint = u2.y - dtdx * (pc * dvh + c2.y);
  g_kphi = 0.0;
  let en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);
  textureStore(O1, c, vec4<f32>(rho, mom));
  textureStore(O2, c, vec4<f32>(en.x, en.y, u2.z - dtdx * c2.z, u2.w));
}

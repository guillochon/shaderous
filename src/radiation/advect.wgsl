// Radiation carried with the gas on the uniform grid (radiation/radiation.ts; plan section 4.2):
// after the hydro update, E and F move with the mass fluxes the update recorded
// (hydro/species_on.wgsl), as E / rho and F / rho per unit mass, the way species move; and E
// takes the flux of the work term, v . P. Float64 mirror: RefRad1D.advect.
//
// Each face carries MF q, with MF its mass flux and q = (E, F) / rho of the upwind cell plus its
// limited slope toward the face, over (1 - Courant number) / 2 (second order for a steady
// velocity, and stable with one forward step). Trapped radiation thus moves exactly as the gas
// does. The work flux v . P uses the arithmetic means of the two cells' velocities and pressure
// tensors, so both cells compute the same bits; with the sweep's - v . g it makes the
// compression work P : grad v. Solid and fixed cells keep their radiation.
//
// Each cell takes its change times its own reduction factor r, as the sweep does with the rest:
// every term but the time derivative carries r, so a steady state is the true one whatever r is,
// in moving gas too, and what is conserved is E_gas + E / r. Where r = 1 (opaque cells, where
// radiation is trapped) the radiation moves exactly with the gas. (Without r here, a radiative
// shock whose precursor's cells were reduced came out 6% too cool: docs/numerics.md section 24.)
#include "hydro/common.wgsl"
#include <eos>
#include <opacity>
#include "radiation/common.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;   // state at the start of the step
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(5) var RI: texture_3d<f32>;
@group(0) @binding(6) var RO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> MF: array<f32>;

fn mf_index(c: vec3<i32>, axis: i32) -> u32 {
  let n = vec3<u32>(P.dims.xyz) + 1u;
  let u = vec3<u32>(c);
  return ((u.z * n.y + u.y) * n.x + u.x) * 3u + u32(axis);
}

#include "radiation/carry.wgsl"

fn adv_load(c: vec3<i32>, a: i32) -> ACell {
  let m = bc_map(c);
  let u = textureLoad(U1, m.c, 0);
  let R = textureLoad(RI, m.c, 0);
  let rho = max(u.x, P.rho_floor);
  let F = R.yzw * m.flip;
  var o: ACell;
  o.q = vec4<f32>(R.x, F) / rho;
  o.rho = rho;
  o.v = bc_velocity(u.yzw / rho, m);
  o.prow = R.x * rad_eddington_row(rad_reduced(R.x, F), a);
  return o;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let R = textureLoad(RI, c, 0);
  if (has_solids() && abs(textureLoad(U2, c, 0).w) > 0.5) {
    textureStore(RO, c, R);
    return;
  }
  g_inner = stencil_inside(c, 2);
  let dtdx = clock.dt / P.dx;
  let flat = P.dims.z == 1u;
  var acc = vec4<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    if (a == 2 && flat) { break; }
    var e = vec3<i32>(0);
    e[a] = 1;
    let m2 = adv_load(c - 2 * e, a);
    let m1 = adv_load(c - e, a);
    let q = adv_load(c, a);
    let p1 = adv_load(c + e, a);
    let p2 = adv_load(c + 2 * e, a);
    acc += face_carry(a, MF[mf_index(c, a)], dtdx, m2, m1, q, p1);
    acc -= face_carry(a, MF[mf_index(c + e, a)], dtdx, m1, q, p1, p2);
  }
  let u = textureLoad(U1, c, 0);
  let rho = max(u.x, P.rho_floor);
  comp_use(comp_tex(c));
  let r = rad_reduction(kappa_rosseland(rho, eos_temperature(rho, textureLoad(U2, c, 0).y)) * rho * P.dx);
  var out = R + (r * dtdx) * acc;
  out.x = max(out.x, 0.0);
  let fm = length(out.yzw);
  if (fm > rad_c() * out.x) { out = vec4<f32>(out.x, out.yzw * (rad_c() * out.x / fm)); }
  textureStore(RO, c, out);
}

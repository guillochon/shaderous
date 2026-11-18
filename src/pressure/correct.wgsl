// The correction of the implicit step (pressure/source.wgsl): with the new pressure p',
//   m' = m* - dt grad p',    rho' = rho - dt div(m*_f - dt d_n p'),
// both as differences of face values, so that mass and momentum are conserved to round-off: a
// face's pressure is the mean of its two cells', its mass flux the mean of their advected
// momenta less dt times the pressure difference across it. Reads the advected state (U1S, U2S),
// the pressure and the old density; writes the new state into the other parity.
#include "hydro/common.wgsl"
#include <eos>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1S: texture_3d<f32>;
@group(0) @binding(3) var U2S: texture_3d<f32>;
@group(0) @binding(4) var PRS: texture_3d<f32>;
@group(0) @binding(5) var RHO0: texture_3d<f32>;
@group(0) @binding(6) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var O2: texture_storage_3d<rgba32float, write>;

#include "pressure/faces.wgsl"

// The mass flux through the face between a lower and an upper cell, from their advected momenta
// along the face normal and their pressures: one expression, whichever of the two cells asks.
// `kg` is dt times the uniform field along the normal, and r_lo, r_hi the old densities: the
// field's kick on the face.
fn mass_flux(m_lo: f32, m_hi: f32, p_lo: f32, p_hi: f32, dtdx: f32, kg: f32, r_lo: f32, r_hi: f32) -> f32 {
  return 0.5 * (m_lo + m_hi) + kg * 0.5 * (r_lo + r_hi) - dtdx * (p_hi - p_lo);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let u1 = textureLoad(U1S, c, 0);
  let u2 = textureLoad(U2S, c, 0);
  // Solid and held cells keep their state.
  if (abs(u2.w) > 0.5) {
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    return;
  }
  let dtdx = clock.dt / P.dx;
  let pc = textureLoad(PRS, c, 0).x;
  let rc = textureLoad(RHO0, c, 0).x;
  var mom = u1.yzw;
  var dm = 0.0;
  let axes = select(3, 2, P.dims.z == 1u);
  for (var a = 0; a < axes; a++) {
    let hi = pface(c, a, true);
    let lo = pface(c, a, false);
    // A uniform field (plans/incompressible.md 4.7): its kick is formed on the faces, with the
    // mean of the two cells' old densities, and the cell takes the mean of its two faces'. A
    // column at rest whose pressure differences obey p_hi - p_lo = dx g (rho_lo + rho_hi) / 2
    // then has no mass flux through any face and no change of momentum, to round-off. A wall's
    // face lies half a cell down the field from the cell: its pressure is the cell's plus that.
    let kg = clock.dt * P.gext[a];
    let half = 0.5 * P.dx * P.gext[a] * rc;
    // (pressure at the face, mass flux through it, the density the field's kick takes there).
    var fh = vec3<f32>(pc + half, 0.0, rc);
    var fl = vec3<f32>(pc - half, 0.0, rc);
    if (hi.kind == FK_GAS) {
      let pq = textureLoad(PRS, hi.q, 0).x;
      let rq = textureLoad(RHO0, hi.q, 0).x;
      fh = vec3<f32>(0.5 * (pc + pq), mass_flux(u1[a + 1], textureLoad(U1S, hi.q, 0)[a + 1], pc, pq, dtdx, kg, rc, rq), 0.5 * (rc + rq));
    }
    if (hi.kind == FK_INLET) { fh.y = textureLoad(U1S, hi.q, 0)[a + 1]; }
    if (hi.kind == FK_OPEN) { fh = vec3<f32>(0.0, open_mom(u1[a + 1] + kg * rc, true, hi.only) + 2.0 * dtdx * pc, rc); }
    if (lo.kind == FK_GAS) {
      let pq = textureLoad(PRS, lo.q, 0).x;
      let rq = textureLoad(RHO0, lo.q, 0).x;
      fl = vec3<f32>(0.5 * (pq + pc), mass_flux(textureLoad(U1S, lo.q, 0)[a + 1], u1[a + 1], pq, pc, dtdx, kg, rq, rc), 0.5 * (rq + rc));
    }
    if (lo.kind == FK_INLET) { fl.y = textureLoad(U1S, lo.q, 0)[a + 1]; }
    if (lo.kind == FK_OPEN) { fl = vec3<f32>(0.0, open_mom(u1[a + 1] + kg * rc, false, lo.only) - 2.0 * dtdx * pc, rc); }
    mom[a] += kg * 0.5 * (fh.z + fl.z) - dtdx * (fh.x - fl.x);
    dm += fh.y - fl.y;
  }
  let rho = max(rc - dtdx * dm, P.rho_floor);
  let eint = eos_eint_from_p(rho, eos_pressure(rho, 0.0));
  // The passive scalar rode on the advective mass flux: it keeps its value per unit mass.
  let sc = u2.z * rho / max(u1.x, P.rho_floor);
  textureStore(O1, c, vec4<f32>(rho, mom));
  textureStore(O2, c, vec4<f32>(0.5 * dot(mom, mom) / rho + eint, eint, sc, u2.w));
}

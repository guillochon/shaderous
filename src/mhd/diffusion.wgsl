// Viscosity and conduction with a magnetic field (plans/mhd-plan.md 7b): hydro/diffusion_core.wgsl's
// isotropic fluxes added to an MHD face flux, on either grid. The field carries no solid cells,
// so there are no walls. Resistivity adds its Poynting flux (mhd/ohm.wgsl). Requires
// mhd/state.wgsl (MQ), MFlux, hydro/diffusion_core.wgsl and load_b(c).
#include "mhd/ohm.wgsl"

// Set by the flux pass for a face in a line-tied wall of the domain (hydro/boundary.wgsl): no
// electric field along it, so no Poynting flux through it, resistive or not.
var<private> g_tied_face: bool = false;

fn dside_mq(q: MQ) -> DSide {
  return DSide(q.a.x, q.a.yzw, q.b.x, false);
}

// The flux F through the face between cell a (state l) and cell a + e (state r), along +axis,
// with the diffusive fluxes added.
fn mhd_add_diffusion(Fin: MFlux, a: vec3<i32>, e: vec3<i32>, axis: i32, l: MQ, r: MQ) -> MFlux {
  var F = Fin;
  if (VISCOSITY) {
    let d = visc_flux(a, e, axis, dside_mq(l), dside_mq(r));
    F.mom += d.xyz;
    F.e += d.w;
  }
  if (CONDUCTION) {
    let q = cond_flux(a, e, dside_mq(l), dside_mq(r));
    F.e += q;
    F.ei += q;
  }
  if (RESISTIVITY && !g_tied_face) { F.e += ohm_eflux(a, e, axis, l.B, r.B); }
  return F;
}

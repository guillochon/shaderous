// Viscosity and thermal conduction for the gas alone: hydro/diffusion_core.wgsl's fluxes added
// to the Riemann flux of a face.
//
// The fluxes are taken from the primitives at the start of the step and added to the Riemann
// flux of the face, so everything that holds for that flux holds for them: the two cells
// sharing a face compute the same bits, and across a refinement boundary or between blocks
// with different timesteps both sides use one flux (amr/update.wgsl). hydro/diff_rate.wgsl
// limits the timestep.
//
// Requires P, the equation of state, hydro/q.wgsl, Flux (hydro/flux.wgsl), g_dx set, and what
// hydro/diffusion_core.wgsl asks of the grid (load_vel, face_rim, load_comp).
#include "hydro/diffusion_core.wgsl"

fn dside_of(q: Q) -> DSide {
  return DSide(q.a.x, q.a.yzw, q.b.x, q_solid(q));
}

// The flux F through the face between cell a (state l) and cell a + e (state r), along +axis,
// with the diffusive fluxes added. l and r are the cells as stored: either may be solid.
fn add_diffusion(Fin: Flux, a: vec3<i32>, e: vec3<i32>, axis: i32, l: Q, r: Q) -> Flux {
  var F = Fin;
  if (!VISCOSITY && !CONDUCTION) { return F; }
  let sl = dside_of(l);
  let sr = dside_of(r);
  if (VISCOSITY) {
    let d = visc_flux(a, e, axis, sl, sr);
    F.mom += d.xyz;
    F.e += d.w;
    F.d = d;
  }
  if (CONDUCTION) {
    let q = cond_flux(a, e, sl, sr);
    F.e += q;
    F.ei += q;
  }
  return F;
}

// hydro/diffusion_core.wgsl's visc_heat_d for a face's flux.
fn visc_heat(f: Flux, v: vec3<f32>) -> f32 {
  return visc_heat_d(f.d, v);
}

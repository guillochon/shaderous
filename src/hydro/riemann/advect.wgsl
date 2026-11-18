// The advective flux of the implicit-pressure step (plans/incompressible.md 4.3): the flux of
// the system with the pressure taken out, (rho, rho v, E) v_n, in the frame where x is the face
// normal. Its only wave speed is the flow's, so the flux is the mean of the two sides' less a
// local Lax-Friedrichs term of that speed. The pressure's part of the momentum and energy fluxes
// comes from the solve (src/pressure). w_hll is ignored.
fn riemann_flux(l: Side, r: Side, w_hll: f32) -> Flux {
  let a = max(abs(l.v.x), abs(r.v.x));
  let ml = l.rho * l.v;
  let mr = r.rho * r.v;
  let el = l.eint + 0.5 * l.rho * dot(l.v, l.v);
  let er = r.eint + 0.5 * r.rho * dot(r.v, r.v);
  var out: Flux;
  out.m = 0.5 * (ml.x + mr.x) - 0.5 * a * (r.rho - l.rho);
  out.mom = 0.5 * (ml * l.v.x + mr * r.v.x) - 0.5 * a * (mr - ml);
  out.e = 0.5 * (el * l.v.x + er * r.v.x) - 0.5 * a * (er - el);
  out.vn = 0.5 * (l.v.x + r.v.x);
  // A state and its image in a wall (or two streams meeting head on): nothing crosses, and the
  // Lax-Friedrichs term would push on the wall as a pressure does. The wall's push is the solve's.
  if (l.v.x == -r.v.x && l.rho == r.rho) { out.mom.x = 0.0; }
  // Internal energy and the passive scalar ride on the mass flux, upwinded.
  let up = out.m >= 0.0;
  out.ei = out.m * select(r.eint / r.rho, l.eint / l.rho, up);
  out.s = out.m * select(r.s, l.s, up);
  return out;
}

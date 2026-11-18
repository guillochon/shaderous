// Dual-energy synchronization (after Bryan et al. 1995). Given the conservatively updated total
// energy and the separately advected internal energy, returns the pair (E, e_int) to store.
// E - KE is used where float32 resolves it; elsewhere the advected e_int is kept and E is
// rebuilt from it. Barotropic equations of state slave e_int to the density. The least e_int
// accepted is the equation of state's own (eos_eint_floor).
//
// An axisymmetric run (cylindrical()) asks more of E - KE where the gas swirls. Its radial faces
// carry the total energy weighed by R and the angular momentum by R^2 (hydro/flux.wgsl
// cyl_face), so the kinetic energy of the swirl that the new angular momentum gives and the one
// the energy update holds differ, by a few percent of it at a steep edge; in cold gas carried
// round by its rotation, where the swirl's energy is a hundred times the heat, E - KE was then
// that error. At a torus's inner edge it drained the pressure in a few hundredths of an orbit,
// and the edge piled up (docs/numerics.md section 22). So there E - KE is taken only where it is
// at least SWIRL_ETA of the swirl's kinetic energy, g_kphi, which the caller sets (0 elsewhere).
// Requires P and the equation of state.
const SWIRL_ETA = 0.1;
var<private> g_kphi: f32 = 0.0;
fn swirl_ke(rho: f32, mom: vec3<f32>) -> f32 {
  return select(0.0, 0.5 * mom.z * mom.z / max(rho, P.rho_floor), cylindrical());
}
fn dual_energy(rho: f32, ke: f32, E: f32, eint: f32) -> vec2<f32> {
  if (EOS_HAS_ENERGY) {
    let eth = E - ke;
    let emin = eos_eint_floor(rho);
    if (eth > P.eta_dual * E && eth >= emin && eth >= SWIRL_ETA * g_kphi) {
      return vec2<f32>(E, eth);          // E stays exactly as the conservative update made it
    }
    var ei = eint;
    if (!(ei >= emin)) { ei = emin; }
    if (EOS_KEEP_TOTAL) { return vec2<f32>(E, ei); }
    return vec2<f32>(ke + ei, ei);
  }
  let ei = eos_eint_from_p(rho, eos_pressure(rho, 0.0));
  return vec2<f32>(ke + ei, ei);
}

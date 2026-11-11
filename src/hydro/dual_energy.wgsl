// Dual-energy synchronization (after Bryan et al. 1995). Given the conservatively updated total
// energy and the separately advected internal energy, returns the pair (E, e_int) to store.
// E - KE is used where float32 resolves it; elsewhere the advected e_int is kept and E is
// rebuilt from it. Barotropic equations of state slave e_int to the density.
// Requires P and the equation of state.
fn dual_energy(rho: f32, ke: f32, E: f32, eint: f32) -> vec2<f32> {
  if (EOS_HAS_ENERGY) {
    let eth = E - ke;
    let emin = eos_eint_from_p(rho, P.p_floor);
    if (eth > P.eta_dual * E && eth >= emin) {
      return vec2<f32>(E, eth);          // E stays exactly as the conservative update made it
    }
    var ei = eint;
    if (!(ei >= emin)) { ei = emin; }
    return vec2<f32>(ke + ei, ei);
  }
  let ei = eos_eint_from_p(rho, eos_pressure(rho, 0.0));
  return vec2<f32>(ke + ei, ei);
}

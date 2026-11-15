// Limits on tenuous gas. Next to a body a million times denser, the few-cell layer of ambient
// gas picks up the dense side's truncation error as heat and speed, and a handful of such cells
// then dictate the timestep for the whole grid. Below a density threshold the specific internal
// energy and the speed are capped. This is not conservative, but the gas involved carries a
// negligible share of the mass. Requires P.
struct Tamed { mom: vec3<f32>, eint: f32, changed: bool }

fn tame_fluff(rho: f32, mom: vec3<f32>, eint: f32) -> Tamed {
  var o = Tamed(mom, eint, false);
  if (rho < P.fluff.x) {
    let emax = rho * P.fluff.y;
    if (eint > emax) { o.eint = emax; o.changed = true; }
    let pmax = rho * P.fluff.z;
    let pm = length(mom);
    if (pm > pmax) { o.mom = mom * (pmax / pm); o.changed = true; }
  }
  return o;
}

// The Alfven speed ceiling (MHD): where the field is strong for the gas it threads, mass is added
// at rest until v_A = |B| / sqrt(rho) is P.fluff.w at most (0 for none). Field that rises out
// of a disk into its thin corona, or that a sink leaves behind once it has taken the gas, would
// otherwise set an ever smaller timestep. Not conservative: the mass it adds is the price.
fn alfven_floor(rho: f32, b2: f32) -> f32 {
  let va = P.fluff.w;
  if (!(va > 0.0)) { return rho; }
  return max(rho, b2 / (va * va));
}

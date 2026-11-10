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

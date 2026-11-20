// The point masses' pull and the accelerating frame's, as the hydro passes see them. The field
// texture holds self-gravity alone (acceleration and potential), and each pass adds these
// where it needs them, from the masses as they are when it runs. A pass that kicks the gas
// takes the energy of this part of the kick from the very acceleration it applied (the kick
// form), while self-gravity's work stays in flux form: the tidal field changes the kinetic
// energy of cold, fast debris by hundreds of times its internal energy each step, and the flux
// form's truncation-level mismatch with the kick would land in E - KE.
// A rotating frame's centrifugal force (gravity/rot.wgsl) is a potential too, and is added here
// with the same rules, so every pass that sees the masses sees it.
// Requires P, a binding `BH: Masses` (gravity/ext_bind.wgsl declares one) and gravity/bh.wgsl.
#include "gravity/bh.wgsl"
#include "gravity/rot.wgsl"

fn ext_on() -> bool {
  return P.bh.z != 0.0 || P.rot.w != 0.0 || P.shear.w != 0.0;
}

// Acceleration at x: the masses' pull less the frame's own acceleration, and the centrifugal
// force of a rotating frame.
fn ext_accel(x: vec3<f32>) -> vec3<f32> {
  if (!ext_on()) { return vec3<f32>(0.0); }
  var g = -BH.frame.xyz;
  if (P.rot.w != 0.0) { g += rot_accel(x); }
  // A frame that follows a binary turns faster or slower: the Euler force, -dOmega/dt x x. (Not a
  // potential, so it is in neither ext_dphi nor ext_phi.)
  if (BH.track.y != 0.0) { g += BH.track.y * vec3<f32>(x.y, -x.x, 0.0); }
  if (P.shear.w != 0.0) { g += tidal_accel(x); }
  for (var i = 0u; i < mass_count(); i++) { g += bh_accel(x, BH.list[i]); }
  return g;
}

// phi(x + d) - phi(x) for the masses (moved on by `ahead`) and the frame, whose acceleration a
// acts as a uniform field -a of potential a . x (each mass's part: gravity/bh.wgsl bh_dphi).
fn ext_dphi(x: vec3<f32>, d: vec3<f32>, ahead: f32) -> f32 {
  if (!ext_on()) { return 0.0; }
  var s = dot(BH.frame.xyz, d);
  if (P.rot.w != 0.0) { s += rot_dphi(x, d); }
  if (P.shear.w != 0.0) { s += tidal_dphi_vertical(x, d); }
  for (var i = 0u; i < mass_count(); i++) { s += bh_dphi(BH.list[i], x, d, ahead); }
  return s;
}

// The masses' potential at x (moved on by `ahead`), without the frame's.
fn ext_phi_masses(x: vec3<f32>, ahead: f32) -> f32 {
  var phi = 0.0;
  if (!ext_on()) { return phi; }
  for (var i = 0u; i < mass_count(); i++) { phi += bh_phi(x, BH.list[i], ahead); }
  return phi;
}

// The masses' potential and the frame's, a . x (and a rotating frame's centrifugal potential:
// the effective potential), at x: what the display adds to self-gravity's.
fn ext_phi(x: vec3<f32>) -> f32 {
  if (!ext_on()) { return 0.0; }
  var t = 0.0;
  if (P.shear.w != 0.0) { t = tidal_phi(x); }
  return dot(BH.frame.xyz, x) + rot_phi(x) + t + ext_phi_masses(x, 0.0);
}

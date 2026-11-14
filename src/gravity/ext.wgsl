// The point masses' pull and the accelerating frame's, as the hydro passes see them. The field
// texture holds self-gravity alone (acceleration and potential), and each pass adds these
// where it needs them, from the masses as they are when it runs. A pass that kicks the gas
// takes the energy of this part of the kick from the very acceleration it applied (the kick
// form), while self-gravity's work stays in flux form: the tidal field changes the kinetic
// energy of cold, fast debris by hundreds of times its internal energy each step, and the flux
// form's truncation-level mismatch with the kick would land in E - KE.
// Requires P, a binding `BH: Masses` (gravity/ext_bind.wgsl declares one) and gravity/bh.wgsl.
#include "gravity/bh.wgsl"

fn ext_on() -> bool {
  return P.bh.z != 0.0;
}

// Acceleration at x: the masses' pull less the frame's own acceleration.
fn ext_accel(x: vec3<f32>) -> vec3<f32> {
  if (!ext_on()) { return vec3<f32>(0.0); }
  var g = -BH.frame.xyz;
  for (var i = 0u; i < mass_count(); i++) { g += bh_accel(x, BH.list[i]); }
  return g;
}

// phi(x + d) - phi(x) for the masses (moved on by `ahead`) and the frame, whose acceleration a
// acts as a uniform field -a of potential a . x. Each mass's part is formed as
// G M (q1 - q0) / (r0 r1 (r0 + r1)), q = r^2 + soft^2, not as the difference of two potentials
// thousands of times larger than it.
fn ext_dphi(x: vec3<f32>, d: vec3<f32>, ahead: f32) -> f32 {
  if (!ext_on()) { return 0.0; }
  var s = dot(BH.frame.xyz, d);
  for (var i = 0u; i < mass_count(); i++) {
    let m = BH.list[i];
    let r = x - (m.pos.xyz + ahead * m.vel.xyz);
    let q0 = dot(r, r) + m.vel.w * m.vel.w;
    let dq = dot(2.0 * r + d, d);
    let r0 = sqrt(q0);
    let r1 = sqrt(q0 + dq);
    s += m.pos.w * dq / (r0 * r1 * (r0 + r1));
  }
  return s;
}

// The masses' potential at x (moved on by `ahead`), without the frame's.
fn ext_phi_masses(x: vec3<f32>, ahead: f32) -> f32 {
  var phi = 0.0;
  if (!ext_on()) { return phi; }
  for (var i = 0u; i < mass_count(); i++) { phi += bh_phi(x, BH.list[i], ahead); }
  return phi;
}

// The masses' potential and the frame's, a . x, at x: what the display adds to self-gravity's.
fn ext_phi(x: vec3<f32>) -> f32 {
  if (!ext_on()) { return 0.0; }
  return dot(BH.frame.xyz, x) + ext_phi_masses(x, 0.0);
}

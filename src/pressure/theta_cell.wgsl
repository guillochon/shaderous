// A cell's theta (pressure/theta.wgsl) from its advected state (U1S, U2S) and its switch chi: the
// source and the correction both form it, with the same bits.
#include "pressure/theta.wgsl"

fn cell_theta(c: vec3<i32>, dt: f32, chi: f32) -> f32 {
  let rho = max(textureLoad(U1S, c, 0).x, P.rho_floor);
  comp_use(comp_tex(c));   // (The cell's own composition, where it varies: species/mixture.ts.)
  let cs = eos_sound_speed(rho, eos_pressure(rho, textureLoad(U2S, c, 0).y));
  return max(chi, select(1.0, ps_theta(cs, dt, P.dx), cs > 0.0));
}

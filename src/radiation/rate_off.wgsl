// Without radiation: nothing adds to the timestep's sound speed (radiation/rate_on.wgsl).
const RAD_RATE = false;
fn rad_cs2(c: vec3<i32>, rho: f32, eint: f32) -> f32 {
  return 0.0;
}

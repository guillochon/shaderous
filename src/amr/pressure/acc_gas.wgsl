// Implicit pressure for a gas with an energy on blocks: the acceleration of the carried solved
// pressure (amr/pressure/lag_fill.wgsl, ghosts filled), which limits the step as on the uniform
// grid (hydro/rates.wgsl). `dx` is the block's cell size.
@group(0) @binding(47) var PFA: texture_3d<f32>;

fn ap_pressure_accel(c: vec3<i32>, rho: f32, dx: f32) -> vec3<f32> {
  var g = vec3<f32>(0.0);
  let axes = select(3, 2, is2d());
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    g[a] = textureLoad(PFA, c + e, 0).x - textureLoad(PFA, c - e, 0).x;
  }
  return -g / (2.0 * dx * rho);
}

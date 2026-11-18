// Implicit pressure for a gas with an energy on blocks: the update keeps the velocities that
// carried each cell's lower and upper faces' advective fluxes, as the cell itself computed them
// (amr/pressure/correct_gas.wgsl reads a fine cell's where a coarse one needs them).
@group(0) @binding(45) var FLO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(46) var FHI: texture_storage_3d<rgba32float, write>;

fn store_face_velocities(c: vec3<i32>, lo: vec3<f32>, hi: vec3<f32>) {
  textureStore(FLO, c, vec4<f32>(lo, 0.0));
  textureStore(FHI, c, vec4<f32>(hi, 0.0));
}

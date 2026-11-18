// Implicit pressure for a gas with an energy: the update keeps the velocity that carried each
// cell's upper faces' advective fluxes, for the correction (pressure/correct_gas.wgsl), which
// moves what the corrected velocity carries beyond it.
@group(0) @binding(32) var FUO: texture_storage_3d<rgba32float, write>;

fn store_face_velocity(c: vec3<i32>, v: vec3<f32>) {
  textureStore(FUO, c, vec4<f32>(v, 0.0));
}

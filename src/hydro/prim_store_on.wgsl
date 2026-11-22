// Stored primitives on the uniform grid (hydro/prim_load_on.wgsl): the timestep-rates pass
// writes each cell's primitives, as load_q in hydro/recon.wgsl returns them, for the predictor
// and the update of the same step.
@group(0) @binding(61) var PW1o: texture_storage_3d<rgba32float, write>;
@group(0) @binding(62) var PW2o: texture_storage_3d<rgba32float, write>;
fn prim_store(c: vec3<i32>, a: vec4<f32>, b: vec4<f32>) {
  textureStore(PW1o, c, a);
  textureStore(PW2o, c, b);
}

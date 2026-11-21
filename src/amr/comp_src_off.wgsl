// One composition everywhere (species/mixture.ts): amr/assemble.wgsl has nothing to gather.
fn comp_assembled(slot: i32, l: vec3<i32>, s: Src) -> vec4<f32> {
  return vec4<f32>(1.0, 1.0, 0.0, 0.0);
}
// (mhd/amr_assemble.wgsl gathers through comp_rhox, under COMP.)
fn comp_rhox(slot: i32, l: vec3<i32>) -> array<vec4<f32>, COMP_NG> {
  return array<vec4<f32>, COMP_NG>();
}

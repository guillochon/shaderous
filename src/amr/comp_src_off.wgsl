// One composition everywhere (species/mixture.ts): amr/assemble.wgsl has nothing to gather.
fn comp_assembled(slot: i32, l: vec3<i32>, s: Src) -> vec4<f32> {
  return vec4<f32>(1.0, 1.0, 0.0, 0.0);
}

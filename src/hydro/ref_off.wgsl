// No reference state: the gas is stored as it is (the slot <refdev>; hydro/ref_on.wgsl has the
// other). Every function here is a stand-in the compiler drops.
const REF_DEV = false;
fn ref_at(c: vec3<i32>) -> vec4<f32> { return vec4<f32>(0.0); }
fn ref_face(rm: vec4<f32>, rc: vec4<f32>, rp: vec4<f32>, sgn: f32, df: f32) -> vec4<f32> { return vec4<f32>(0.0); }
fn ref_slope(rm: vec4<f32>, rc: vec4<f32>, rp: vec4<f32>) -> f32 { return 0.0; }

// The <closure> slot without the turbulence closure: nothing is read and the fluxes are the
// solver's own (the branches on CLOSURE in hydro/flux.wgsl compile out).
const CLOSURE = false;
fn closure_face(a: vec3<i32>, b: vec3<i32>) -> vec3<f32> { return vec3<f32>(0.0); }

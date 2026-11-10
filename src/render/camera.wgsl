// Shared by the ray-marcher and the slice view: camera block and a fullscreen triangle.
struct Cam {
  eye: vec4<f32>,
  right: vec4<f32>,
  up: vec4<f32>,
  fwd: vec4<f32>,
  p: vec4<f32>,   // tan(fov / 2), aspect, opacity scale, samples across the box
  q: vec4<f32>,   // slice axis, slice position in [0, 1), cells per side, unused
}
@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var VOL: texture_3d<f32>;
@group(0) @binding(2) var TF: texture_2d<f32>;
@group(0) @binding(3) var S: sampler;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,    // [-1, 1], y up
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  return VOut(vec4<f32>(p, 0.0, 1.0), p);
}

const BG = vec3<f32>(0.05, 0.058, 0.078);

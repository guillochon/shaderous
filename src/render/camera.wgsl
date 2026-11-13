// Shared by the ray-marcher and the slice view: camera block and a fullscreen triangle.
struct Cam {
  // The slice view at zoom 1, in units of its width: eye.w how far its middle sits left of
  // the canvas's middle, right.w and up.w half the canvas's width and height.
  eye: vec4<f32>,
  right: vec4<f32>,
  up: vec4<f32>,
  fwd: vec4<f32>,   // w: volume view, opacity of the isosurfaces
  p: vec4<f32>,   // tan(fov / 2), aspect, opacity scale, samples across the box
  q: vec4<f32>,   // slice axis, slice position in [0, 1), cells per side, cells along z
  r: vec4<f32>,   // 1 to outline the blocks of an adaptive mesh, slice zoom, slice view center in [0, 1]^2
  k: vec4<f32>,   // contours (isosurfaces in the volume view): how many (0 for none), log10 of the range they divide, the one highlighted (0 for none)
  sc: vec4<f32>,  // the color scale: 1 if logarithmic, its ends (in log10 when logarithmic); w: volume view, 1 to draw the gas as fog
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

// The color of contour level k (counted from 1): C.k.x levels evenly spaced in log10 |value| from
// C.k.y to C.k.z, the ends left out. `neg`: the level of a field of either sign taken below zero.
fn level_color(k: f32, neg: bool) -> vec3<f32> {
  let lg = C.k.y + k * (C.k.z - C.k.y) / (C.k.x + 1.0);
  var x = select(pow(10.0, lg), -pow(10.0, lg), neg);
  if (C.sc.x != 0.0) { x = lg; }
  let t = clamp((x - C.sc.y) / max(C.sc.z - C.sc.y, 1.0e-20), 0.0, 1.0);
  return textureSampleLevel(TF, S, vec2<f32>(t, 0.5), 0.0).rgb;
}
// Obstacles: the field volume holds -1 in solid cells.
const SOLID = vec3<f32>(0.34, 0.37, 0.43);

// Lines over the volume view: the blocks of an adaptive mesh and the guides through a tool.
// Positions are in box units ([-0.5, 0.5]^3), projected with the ray-marcher's camera. Each
// fragment's depth is that of the field lines' tubes (render/fieldtubes.wgsl), so the block
// outlines pass behind the tubes in front of them.
#include "render/camera.wgsl"

struct LIn {
  @location(0) pos: vec3<f32>,
  @location(1) color: vec4<f32>,
}
struct LOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) w: f32,   // distance in front of the eye, along the view
}
struct LFrag {
  @location(0) color: vec4<f32>,
  @builtin(frag_depth) depth: f32,
}

@vertex
fn line_vs(v: LIn) -> LOut {
  let d = v.pos - C.eye.xyz;
  // The ray-marcher's ray through uv is fwd + uv.x aspect tan right + uv.y tan up.
  let w = dot(d, C.fwd.xyz);
  let clip = vec4<f32>(dot(d, C.right.xyz) / (C.p.x * C.p.y), dot(d, C.up.xyz) / C.p.x, 0.0, w);
  return LOut(clip, v.color, w);
}

@fragment
fn line_fs(in: LOut) -> LFrag {
  return LFrag(in.color, clamp(in.w / (in.w + 1.0), 0.0, 1.0));
}

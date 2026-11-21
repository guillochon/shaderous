// Forces on bodies on the block grid: hydro/forces.wgsl over the leaves' cells.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"

@group(0) @binding(1) var WP: texture_3d<f32>;
@group(0) @binding(2) var WV: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> REFS: array<vec4<f32>, 4>;
@group(0) @binding(4) var WX: texture_3d<f32>;   // moving bodies' exchange (hydro/moving_exchange.wgsl), or one texel

#include <force_put>

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  if (!blk_leaf(slot)) { return red_identity(); }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let a = atl(blk_origin(slot), l);
  let wp = textureLoad(WP, a, 0);
  let wv = textureLoad(WV, select(vec3<i32>(0), a, FORCE_VISC), 0);
  let fv = select(vec3<f32>(0.0), wv.xyz, FORCE_VISC);
  // (Moving bodies are not on blocks: WX is one texel there.)
  let wx0 = textureLoad(WX, vec3<i32>(0), 0).w * 0.0;
  let m = max(wp.w, select(0.0, wv.w, FORCE_VISC)) + wx0;
  if (m < 0.5) { return red_identity(); }
  let b = u32(clamp(round(m) - 1.0, 0.0, 3.0));
  var x = blk_pos(slot, l);
  if (is2d()) { x.z = 0.0; }
  return force_put(b, wp.xyz, fv, cross(x - REFS[b].xyz, wp.xyz + fv));
}

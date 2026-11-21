// Forces on bodies, reduction source on the uniform grid (plans/wind-tunnel.md 4.2): each cell's
// wall-face forces (hydro/walls_on.wgsl, pressure/pwalls_on.wgsl), summed per body, with their
// torque about the body's point (REFS). Channel order: per body, the pressure's force (3),
// viscosity's (3), the torque of both (3); bodies one after another (src/hydro/forces.ts).
// A cell's forces act at its faces' centers; the torque takes them at the cell's center, which
// is exact for the pressure (its force on a face is along the face's normal, through the center
// offset) and off by half a cell's lever for viscosity's.
#include "hydro/common.wgsl"

@group(0) @binding(1) var WP: texture_3d<f32>;
@group(0) @binding(2) var WV: texture_3d<f32>;   // one texel of nothing where no pass writes it
@group(0) @binding(3) var<storage, read> REFS: array<vec4<f32>, 4>;
@group(0) @binding(4) var WX: texture_3d<f32>;   // moving bodies' exchange (hydro/moving_exchange.wgsl), or one texel

#include <force_put>

fn red_cell(c: vec3<i32>) -> Red {
  let wp = textureLoad(WP, c, 0);
  let wv = textureLoad(WV, select(vec3<i32>(0), c, FORCE_VISC), 0);
  let fv = select(vec3<f32>(0.0), wv.xyz, FORCE_VISC);
  // A moving body's push where its cells changed kind counts with the pressure's part.
  let wx = textureLoad(WX, select(vec3<i32>(0), c, FORCE_MOVING), 0);
  let fp = wp.xyz + select(vec3<f32>(0.0), wx.xyz, FORCE_MOVING);
  let m = max(max(wp.w, select(0.0, wv.w, FORCE_VISC)), select(0.0, wx.w, FORCE_MOVING));
  if (m < 0.5) { return red_identity(); }
  let b = u32(clamp(round(m) - 1.0, 0.0, 3.0));
  var x = (vec3<f32>(c) + 0.5) * P.dx + P.org.xyz;
  if (P.dims.z == 1u) { x.z = 0.0; }
  return force_put(b, fp, fv, cross(x - REFS[b].xyz, fp + fv));
}

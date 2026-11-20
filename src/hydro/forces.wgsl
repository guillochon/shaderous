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

#include <force_put>

fn red_cell(c: vec3<i32>) -> Red {
  let wp = textureLoad(WP, c, 0);
  let wv = textureLoad(WV, select(vec3<i32>(0), c, FORCE_VISC), 0);
  let fv = select(vec3<f32>(0.0), wv.xyz, FORCE_VISC);
  let m = max(wp.w, select(0.0, wv.w, FORCE_VISC));
  if (m < 0.5) { return red_identity(); }
  let b = u32(clamp(round(m) - 1.0, 0.0, 3.0));
  var x = (vec3<f32>(c) + 0.5) * P.dx + P.org.xyz;
  if (P.dims.z == 1u) { x.z = 0.0; }
  return force_put(b, wp.xyz, fv, cross(x - REFS[b].xyz, wp.xyz + fv));
}

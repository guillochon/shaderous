// The field brushes on the uniform grid (edit/field_brush_core.wgsl), in two passes from the
// current parity into the other:
//   faces   every face texel: the field plus the brush's change, and the compensations as
//           they were;
//   energy  every cell: the state as it was, its total energy changed by that of the field.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "edit/brush_core.wgsl"

@group(0) @binding(2) var BF: texture_3d<f32>;   // faces before
@group(0) @binding(3) var CF: texture_3d<f32>;   // their compensations
@group(0) @binding(4) var BO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(5) var CO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var BN: texture_3d<f32>;   // faces after (energy pass)
@group(0) @binding(7) var N1: texture_3d<f32>;
@group(0) @binding(8) var N2: texture_3d<f32>;
@group(0) @binding(9) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(10) var O2: texture_storage_3d<rgba32float, write>;

#include "mhd/bcell.wgsl"

fn loop_count(n: i32) -> i32 {
  return select(n, 0, P.dims.x == 0u);
}
fn fb_cells() -> f32 {
  return f32(P.dims.x);
}
fn fb_field(x: vec3<f32>) -> vec3<f32> {
  let n = vec3<i32>(P.dims.xyz);
  return bcell(clamp(vec3<i32>(floor(x * vec3<f32>(n))), vec3<i32>(0), n - 1));
}

#include "edit/field_brush_core.wgsl"

@compute @workgroup_size(8, 8, 4)
fn faces(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= textureDimensions(BF))) { return; }
  let t = vec3<i32>(gid);
  let h = 1.0 / f32(P.dims.x);
  var b = textureLoad(BF, t, 0);
  for (var a = 0; a < loop_count(3); a++) {
    b[a] += fb_face(a, vec3<f32>(t) * h, h, 1);
  }
  textureStore(BO, t, b);
  textureStore(CO, t, textureLoad(CF, t, 0));
}

// The cell-centered field from a face texture, as bcell() takes it (in 2D the one z face).
fn centered(F: texture_3d<f32>, c: vec3<i32>) -> vec3<f32> {
  let n = vec3<i32>(P.dims.xyz);
  let up = c + 1;
  let wrap = vec3<bool>(up.x >= n.x && P.bc_lo[0] == 0u, up.y >= n.y && P.bc_lo[1] == 0u, up.z >= n.z && P.bc_lo[2] == 0u);
  let hi = select(up, vec3<i32>(0), wrap);
  let lo = textureLoad(F, c, 0);
  let z = select(textureLoad(F, vec3<i32>(c.x, c.y, hi.z), 0).z, lo.z, n.z == 1);
  return 0.5 * vec3<f32>(lo.x + textureLoad(F, vec3<i32>(hi.x, c.y, c.z), 0).x, lo.y + textureLoad(F, vec3<i32>(c.x, hi.y, c.z), 0).y, lo.z + z);
}

@compute @workgroup_size(8, 8, 4)
fn energy(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let bo = centered(BF, c);
  let bn = centered(BN, c);
  var u2 = textureLoad(N2, c, 0);
  u2.x += 0.5 * dot(bn - bo, bn + bo);
  textureStore(O1, c, textureLoad(N1, c, 0));
  textureStore(O2, c, u2);
}

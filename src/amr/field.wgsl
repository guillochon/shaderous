// Resamples the block grid onto the uniform display volume that the ray-marcher and slice
// view read. Each display voxel takes the value of the leaf cell under its centre.
//   r = displayed quantity, normalised to [0, 1]
//   g, b, a = 1 on the lower boundary of a block, one display voxel wide, as seen in a z-, x-
//             or y-normal slice
#include "hydro/common.wgsl"
#include <eos>
#include "amr/grid.wgsl"

struct View {
  field: u32,      // 0 density, 1 temperature, 2 passive scalar, 3 pressure, 4 speed
  logscale: u32,
  lo: f32,         // range ends, in log10 units when logscale is set
  hi: f32,
  nd: vec4<u32>,   // display volume size
}
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var VOL: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var IDX: texture_3d<u32>;   // leaf slot per deepest-level block

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= V.nd.xyz)) { return; }
  var frac = (vec3<f32>(gid) + 0.5) / vec3<f32>(V.nd.xyz);
  if (is2d()) { frac.z = 0.0; }
  let m = i32(P.amr.y << P.amr.z);
  let slot = i32(textureLoad(IDX, vec3<i32>(frac * f32(m)), 0).x);
  g_dx = blk_dx(slot);
  let nblk = i32(P.amr.y) * (1 << u32(bt(slot, 0u)));   // blocks across the domain at this level
  let ncell = nblk * BS;
  var l = clamp(vec3<i32>(frac * f32(ncell)) - blk_coord(slot) * BS, vec3<i32>(0), vec3<i32>(BS - 1));
  if (is2d()) { l.z = 0; }
  let c = atl(blk_origin(slot), l);
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  let rho = max(u1.x, P.rho_floor);
  var x = rho;
  switch (V.field) {
    case 1u { x = eos_temperature(rho, u2.y); }
    case 2u { x = u2.z / rho; }
    case 3u { x = eos_pressure(rho, u2.y); }
    case 4u { x = length(u1.yzw) / rho; }
    default {}
  }
  if (V.logscale != 0u) { x = log(max(x, 1.0e-30)) * 0.4342944819; }
  var t = clamp((x - V.lo) / max(V.hi - V.lo, 1.0e-20), 0.0, 1.0);
  if (u2.w > 0.5) { t = -1.0; }   // solid: drawn as a body, not as gas

  // Position within the block in display voxels; the first voxel along an axis is the outline.
  let within = (frac * f32(nblk) - vec3<f32>(blk_coord(slot))) * vec3<f32>(V.nd.xyz) / f32(nblk);
  let edge = vec3<f32>(within < vec3<f32>(1.0));
  textureStore(VOL, vec3<i32>(gid), vec4<f32>(t, max(edge.x, edge.y), max(edge.y, edge.z), max(edge.x, edge.z)));
}

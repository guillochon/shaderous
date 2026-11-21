// Field lines (render/fieldlines.wgsl), block grid: the cell-centered magnetic field and the
// velocity of the leaf cell under the center of each voxel of the field-line volume, and the
// largest |B| in it. The field is the mean of the cell's two faces along each axis, from the
// leaf's own faces (texel BS holds its upper ones), as the display's magnetic fields take it.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"

@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(5) var IDX: texture_3d<u32>;   // leaf slot per deepest-level block
@group(0) @binding(13) var BF: texture_3d<f32>;
@group(0) @binding(14) var BV: texture_storage_3d<rgba16float, write>;   // B, |B|
@group(0) @binding(15) var VV: texture_storage_3d<rgba16float, write>;   // velocity, density
@group(0) @binding(16) var<storage, read_write> FR: array<atomic<u32>, 4>;
// View only, one field held (amr/field.wgsl BAKED): no state, so the density comes from the baked
// field's atlas (its second channel) and the velocity is taken as zero.
override BAKED: u32 = 0u;
@group(0) @binding(17) var BK: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m = textureDimensions(BV);
  if (any(gid >= m)) { return; }
  var frac = (vec3<f32>(gid) + 0.5) / vec3<f32>(m);
  if (is2d()) { frac.z = 0.0; }
  let roots = i32(P.amr.y << P.amr.z);
  let slot = i32(textureLoad(IDX, min(vec3<i32>(frac * f32(roots)), vec3<i32>(roots - 1)), 0).x);
  let ncell = i32(P.amr.y) * (1 << u32(bt(slot, 0u))) * BS;
  var l = clamp(vec3<i32>(frac * f32(ncell)) - blk_coord(slot) * BS, vec3<i32>(0), vec3<i32>(BS - 1));
  if (is2d()) { l.z = 0; }
  let c = atl(blk_origin(slot), l);
  let lo = textureLoad(BF, c, 0).xyz;
  var hi = lo;
  for (var a = 0; a < 3; a++) {
    if (a == 2 && is2d()) { break; }
    var e = vec3<i32>(0);
    e[a] = 1;
    hi[a] = textureLoad(BF, c + e, 0)[a];
  }
  let B = 0.5 * (lo + hi);
  var u = textureLoad(U1, c, 0);
  if (BAKED != 0u) { u = vec4<f32>(textureLoad(BK, bk_texel(slot, l), 0).y, 0.0, 0.0, 0.0); }
  let rho = max(u.x, P.rho_floor);
  let b = length(B);
  textureStore(BV, vec3<i32>(gid), vec4<f32>(B, b));
  textureStore(VV, vec3<i32>(gid), vec4<f32>(u.yzw / rho, rho));
  // 2D: the field-line colors' range (render/fluxfunction.wgsl) wants the in-plane |B|'s largest
  // and its mean log: FR[1] sums log2 |B| in sixteenths (offset by 64), FR[2] counts the cells.
  // (In 3D the tracer uses FR[1] and FR[2] for the lines' own range.)
  if (is2d()) {
    let bp = length(B.xy);
    atomicMax(&FR[0], bitcast<u32>(bp));
    if (bp > 0.0) {
      atomicAdd(&FR[1], u32(clamp((log2(bp) + 64.0) * 16.0, 0.0, 2047.0)));
      atomicAdd(&FR[2], 1u);
    }
  } else {
    atomicMax(&FR[0], bitcast<u32>(b));
  }
}

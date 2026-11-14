// Magnetic diagnostics on the block grid (mhd/diagnostics.wgsl on the uniform grid), over leaf
// cells: magnetic energy, the largest |div B| dx of a cell from its six faces, and the largest
// |B| at a cell center. Every face a leaf cell needs is the block's own (texel BS holds its upper
// faces).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"

@group(0) @binding(4) var BF: texture_3d<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  var r: Red;
  r[0] = 0.0;
  r[1] = 0.0;
  r[2] = 0.0;
  let slot = c.x / BS;
  if (!blk_leaf(slot)) { return r; }
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  let o = blk_origin(slot);
  let lo = textureLoad(BF, atl(o, l), 0).xyz;
  var hi = lo;
  var div = 0.0;
  for (var a = 0; a < 3; a++) {
    if (a == 2 && is2d()) { break; }
    var e = vec3<i32>(0);
    e[a] = 1;
    hi[a] = textureLoad(BF, atl(o, l + e), 0)[a];
    div += hi[a] - lo[a];
  }
  let B = 0.5 * (lo + hi);
  let dx = blk_dx(slot);
  r[0] = 0.5 * dot(B, B) * select(dx * dx * dx, dx * dx, is2d());
  r[1] = abs(div);
  r[2] = length(B);
  return r;
}

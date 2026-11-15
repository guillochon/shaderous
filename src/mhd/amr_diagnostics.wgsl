// Magnetic diagnostics on the block grid (mhd/diagnostics.wgsl on the uniform grid), over leaf
// cells: magnetic energy, the largest |div B| dx of a cell from its six faces, the largest |B|
// at a cell center, and the largest 1 / beta = B^2 / 2p (the least plasma beta). Every face a
// leaf cell needs is the block's own (texel BS holds its upper faces). Then, with a point mass,
// a disk's stresses and MRI quality factors about it (mhd/mri.wgsl).
#include "hydro/common.wgsl"
#include <eos>
#include "amr/grid.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> BH: Masses;   // (20 is the reduction's output)

#include "gravity/bh.wgsl"
#include "mhd/mri.wgsl"

fn red_cell(c: vec3<i32>) -> Red {
  var r: Red;
  r[0] = 0.0;
  r[1] = 0.0;
  r[2] = 0.0;
  r[3] = 0.0;
  for (var i = 4; i < 4 + DISK_CHANNELS; i++) { r[i] = 0.0; }
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
  let c0 = atl(o, l);
  let u1 = textureLoad(U1, c0, 0);
  let p = eos_pressure(max(u1.x, P.rho_floor), textureLoad(U2, c0, 0).y);
  r[3] = 0.5 * dot(B, B) / max(p, 1.0e-30);
  let dk = disk_channels(B, u1.x, u1.yzw, p, blk_pos(slot, l), dx, select(dx * dx * dx, dx * dx, is2d()));
  for (var i = 0; i < DISK_CHANNELS; i++) { r[4 + i] = dk[i]; }
  return r;
}

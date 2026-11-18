// Magnetic diagnostics reduction source (mhd/mhd.ts, MAG_OPS): magnetic energy, the largest
// |div B| dx of any cell (from its six faces), the largest |B| at a cell center, which
// normalizes the divergence, and the largest 1 / beta = B^2 / 2p (the least plasma beta); then,
// with a point mass, a disk's stresses and MRI quality factors about it (mhd/mri.wgsl).
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> BH: Masses;   // (20 is the reduction's output)

#include "mhd/common.wgsl"
#include "gravity/bh.wgsl"
#include "mhd/mri.wgsl"

fn red_cell(c: vec3<i32>) -> Red {
  let B = bcell(c);
  var div = 0.0;
  // (An axisymmetric run: its radial faces weighed by their areas, and its ring cells' volumes.)
  let w = select(vec2<f32>(1.0), cyl_weights(c.x), cylindrical());
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    if (a == 0) { div += w.y * bface(a, c + e) - w.x * bface(a, c); }
    else { div += bface(a, c + e) - bface(a, c); }
  }
  let dv = select(P.geom.x, 6.2831853 * (f32(c.x) + 0.5) * P.dx * P.dx * P.dx, cylindrical());
  var r: Red;
  r[0] = 0.5 * dot(B, B) * dv;
  r[1] = abs(div);
  r[2] = length(B);
  let u1 = textureLoad(U1, c, 0);
  let p = eos_pressure(max(u1.x, P.rho_floor), textureLoad(U2, c, 0).y);
  r[3] = 0.5 * dot(B, B) / max(p, 1.0e-30);
  let dk = disk_channels(B, u1.x, u1.yzw, p, cell_pos(c), P.dx, P.geom.x);
  for (var i = 0; i < DISK_CHANNELS; i++) { r[4 + i] = dk[i]; }
  return r;
}

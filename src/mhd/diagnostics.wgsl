// Magnetic diagnostics reduction source (mhd/mhd.ts, MAG_OPS): magnetic energy, the largest
// |div B| dx of any cell (from its six faces), and the largest |B| at a cell center, which
// normalizes the divergence.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;

#include "mhd/common.wgsl"

fn red_cell(c: vec3<i32>) -> Red {
  let B = bcell(c);
  var div = 0.0;
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    div += bface(a, c + e) - bface(a, c);
  }
  var r: Red;
  r[0] = 0.5 * dot(B, B) * P.geom.x;
  r[1] = abs(div);
  r[2] = length(B);
  return r;
}

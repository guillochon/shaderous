// Moves the multipole moments from the box center (as the reduction sums them,
// gravity/moments.wgsl) to the center of mass, where the boundary potential is expanded
// (gravity/multipole_eval.wgsl). One invocation, once per reduction.
//
// MP = (M, center of mass, then the MP_COUNT moments about it), positions relative to the box center.
#include "gravity/mp_index.wgsl"

@group(0) @binding(0) var<storage, read> RAW: array<f32>;
@group(0) @binding(1) var<storage, read_write> MP: array<f32>;

@compute @workgroup_size(1)
fn main() {
  let M = RAW[0];
  var cm = vec3<f32>(0.0);
  if (M > 0.0) { cm = vec3<f32>(RAW[1], RAW[2], RAW[3]) / M; }
  MP[0] = M;
  MP[1] = cm.x;
  MP[2] = cm.y;
  MP[3] = cm.z;
  // (y - c)^alpha = sum over beta <= alpha of binomials times y^beta (-c)^(alpha - beta).
  for (var d = 0; d <= MP_ORDER; d++) {
    for (var a = d; a >= 0; a--) {
      for (var b = d - a; b >= 0; b--) {
        let c = d - a - b;
        var s = 0.0;
        for (var i = 0; i <= a; i++) {
          for (var j = 0; j <= b; j++) {
            for (var k = 0; k <= c; k++) {
              let w = mp_binom(a, i) * mp_binom(b, j) * mp_binom(c, k)
                    * mp_pow(-cm.x, a - i) * mp_pow(-cm.y, b - j) * mp_pow(-cm.z, c - k);
              s += w * RAW[mp_index(i, j, k)];
            }
          }
        }
        MP[4u + mp_index(a, b, c)] = s;
      }
    }
  }
}

// Moves the multipole moments from the box center (as the reduction sums them,
// gravity/moments.wgsl) to the center of mass, where the boundary potential is expanded
// (gravity/multipole_eval.wgsl). One invocation, once per reduction.
//
// MP = (M, center of mass, then the MP_COUNT moments about it), positions relative to the box center.
#include "gravity/mp_index.wgsl"

@group(0) @binding(0) var<storage, read> RAW: array<f32>;
@group(0) @binding(1) var<storage, read_write> MP: array<f32>;

// Mirror symmetry (SimConfig.symmetry): bit a set where axis a is mirrored about its plane
// through the origin, so the grid holds one 2^k-th of the configuration. Each image of a cell
// contributes its moments with the signs of its mirrored coordinates, so the whole's moment
// sum m x^i y^j z^k is the grid's times 2 for each mirrored axis whose power is even, and zero
// if any mirrored axis's power is odd: the mass doubles per mirror, the center of mass lies on
// the planes, and every moment odd in a mirrored coordinate vanishes.
override SYM_MASK: u32 = 0u;

fn unfold(i: i32, axis: u32) -> f32 {
  if ((SYM_MASK & (1u << axis)) == 0u) { return 1.0; }
  return select(0.0, 2.0, (i & 1) == 0);
}
fn raw(i: i32, j: i32, k: i32) -> f32 {
  return RAW[mp_index(i, j, k)] * unfold(i, 0u) * unfold(j, 1u) * unfold(k, 2u);
}

@compute @workgroup_size(1)
fn main() {
  let M = raw(0, 0, 0);
  var cm = vec3<f32>(0.0);
  if (M > 0.0) { cm = vec3<f32>(raw(1, 0, 0), raw(0, 1, 0), raw(0, 0, 1)) / M; }
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
              s += w * raw(i, j, k);
            }
          }
        }
        MP[4u + mp_index(a, b, c)] = s;
      }
    }
  }
}

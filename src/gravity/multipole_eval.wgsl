// Boundary potential for isolated boundaries: the multipole expansion about the center of mass
// to l = 4 (MP_ORDER), from the moments gravity/multipole_com.wgsl leaves in MP: the mass, the
// center of mass, then the moments about it (gravity/mp_index.wgsl). With
//   1 / |x - y| = sum_l |y|^l P_l(n . y / |y|) / r^(l+1),   r = |x - c|, n = (x - c) / r,
// and T(p, k) = sum m (n . y)^p |y|^(2k) (y about the center of mass),
//   phi = -G [ T00 / r + T10 / r^2 + (3 T20 - T01) / (2 r^3) + (5 T30 - 3 T11) / (2 r^4)
//              + (35 T40 - 30 T21 + 3 T02) / (8 r^5) ].
// To the quadrupole only, two white dwarfs a third of the box from its center felt the box's
// edge as a pull toward the middle, as strong as a third of each other's: their settling
// (problems/settle.ts) never ended, and the collision would have run with it.
#include "gravity/mp_index.wgsl"

@group(0) @binding(10) var<storage, read> MP: array<f32>;

fn mp_mom(a: i32, b: i32, c: i32) -> f32 {
  return MP[4u + mp_index(a, b, c)];
}

// T(p, k) along n.
fn mp_t(n: vec3<f32>, p: i32, k: i32) -> f32 {
  var s = 0.0;
  for (var a = 0; a <= p; a++) {
    for (var b = 0; b <= p - a; b++) {
      let c = p - a - b;
      let w = mp_fact(p) / (mp_fact(a) * mp_fact(b) * mp_fact(c)) * mp_pow(n.x, a) * mp_pow(n.y, b) * mp_pow(n.z, c);
      var m = 0.0;
      if (k == 0) {
        m = mp_mom(a, b, c);
      } else if (k == 1) {
        m = mp_mom(a + 2, b, c) + mp_mom(a, b + 2, c) + mp_mom(a, b, c + 2);
      } else {
        m = mp_mom(a + 4, b, c) + mp_mom(a, b + 4, c) + mp_mom(a, b, c + 4)
          + 2.0 * (mp_mom(a + 2, b + 2, c) + mp_mom(a + 2, b, c + 2) + mp_mom(a, b + 2, c + 2));
      }
      s += w * m;
    }
  }
  return s;
}

fn phi_boundary(x: vec3<f32>) -> f32 {
  let M = MP[0];
  if (!(M > 0.0)) { return 0.0; }
  let d = x - vec3<f32>(MP[1], MP[2], MP[3]);
  let r = length(d);
  let n = d / r;
  let q2 = 0.5 * (3.0 * mp_t(n, 2, 0) - mp_t(n, 0, 1));
  let q3 = 0.5 * (5.0 * mp_t(n, 3, 0) - 3.0 * mp_t(n, 1, 1));
  let q4 = (35.0 * mp_t(n, 4, 0) - 30.0 * mp_t(n, 2, 1) + 3.0 * mp_t(n, 0, 2)) / 8.0;
  let ir = 1.0 / r;
  return -P.G * ir * (M + ir * (mp_t(n, 1, 0) + ir * (q2 + ir * (q3 + ir * q4))));
}

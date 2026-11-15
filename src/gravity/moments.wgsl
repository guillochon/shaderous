// A cell's contribution to the multipole moments (gravity/mp_index.wgsl): its mass m at x, x
// relative to the box center. The reductions that use it have MP_COUNT channels.
#include "gravity/mp_index.wgsl"

fn mp_moments(m: f32, x: vec3<f32>) -> Red {
  var r: Red;
  var i = 0u;
  for (var d = 0; d <= MP_ORDER; d++) {
    for (var a = d; a >= 0; a--) {
      for (var b = d - a; b >= 0; b--) {
        r[i] = m * mp_pow(x.x, a) * mp_pow(x.y, b) * mp_pow(x.z, d - a - b);
        i++;
      }
    }
  }
  return r;
}

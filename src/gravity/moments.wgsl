// A cell's contribution to the multipole moments (gravity/mp_index.wgsl): its mass m at x, x
// relative to the box center. The reductions that use it have MP_COUNT channels.
#include "gravity/mp_index.wgsl"

// Written out term by term, to order 4 (MP_ORDER; MP_COUNT terms): with every index a constant
// the record stays in registers, where a loop over the terms kept it in each thread's local
// memory (gpu/reduce.ts says why that matters). Powers multiply up from x as mp_pow() does.
// An axisymmetric (R, z) run's cell is a ring of mass m and radius R about the y axis, at
// height y: its moments are the ring's, x^a z^c averaged around it (nonzero for a and c even:
// <cos^2> = <sin^2> = 1/2, <cos^4> = <sin^4> = 3/8, <cos^2 sin^2> = 1/8). The center of mass is
// then on the axis, and gravity/multipole_eval.wgsl's expansion is the ring's as it stands.
fn mp_ring_moments(m: f32, R: f32, y: f32) -> Red {
  let r2 = R * R;
  let r4 = r2 * r2;
  let y2 = y * y;
  var r: Red;   // zero-initialized: the moments odd in x or z stay 0
  r[0] = m;
  r[2] = m * y;
  r[4] = 0.5 * m * r2;
  r[7] = m * y2;
  r[9] = 0.5 * m * r2;
  r[11] = 0.5 * m * r2 * y;
  r[16] = m * y2 * y;
  r[18] = 0.5 * m * r2 * y;
  r[20] = 0.375 * m * r4;
  r[23] = 0.5 * m * r2 * y2;
  r[25] = 0.125 * m * r4;
  r[30] = m * y2 * y2;
  r[32] = 0.5 * m * r2 * y2;
  r[34] = 0.375 * m * r4;
  return r;
}

fn mp_moments(m: f32, x: vec3<f32>) -> Red {
  let x1 = x.x;
  let x2 = x1 * x1;
  let x3 = x2 * x1;
  let x4 = x3 * x1;
  let y1 = x.y;
  let y2 = y1 * y1;
  let y3 = y2 * y1;
  let y4 = y3 * y1;
  let z1 = x.z;
  let z2 = z1 * z1;
  let z3 = z2 * z1;
  let z4 = z3 * z1;
  var r: Red;
  r[0] = m;
  r[1] = m * x1;
  r[2] = m * y1;
  r[3] = m * z1;
  r[4] = m * x2;
  r[5] = m * x1 * y1;
  r[6] = m * x1 * z1;
  r[7] = m * y2;
  r[8] = m * y1 * z1;
  r[9] = m * z2;
  r[10] = m * x3;
  r[11] = m * x2 * y1;
  r[12] = m * x2 * z1;
  r[13] = m * x1 * y2;
  r[14] = m * x1 * y1 * z1;
  r[15] = m * x1 * z2;
  r[16] = m * y3;
  r[17] = m * y2 * z1;
  r[18] = m * y1 * z2;
  r[19] = m * z3;
  r[20] = m * x4;
  r[21] = m * x3 * y1;
  r[22] = m * x3 * z1;
  r[23] = m * x2 * y2;
  r[24] = m * x2 * y1 * z1;
  r[25] = m * x2 * z2;
  r[26] = m * x1 * y3;
  r[27] = m * x1 * y2 * z1;
  r[28] = m * x1 * y1 * z2;
  r[29] = m * x1 * z3;
  r[30] = m * y4;
  r[31] = m * y3 * z1;
  r[32] = m * y2 * z2;
  r[33] = m * y1 * z3;
  r[34] = m * z4;
  return r;
}

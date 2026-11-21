// Partly filled cells (hydro/cut.ts): CUT, cut_at(c) (gas share, upper faces' open shares) and
// cut_wall(c), from the <cut> slot. A file of its own so that every shader that asks gets it once.
#include <cut>

// The open share of the face below cell c along `axis` (the upper face of the cell below).
fn cut_lo(c: vec3<i32>, axis: i32) -> f32 {
  var q = c;
  q[axis] -= 1;
  return cut_at(q)[axis + 1];
}

// The wall's vector in cell c: its area (in faces) times its normal, out of the gas, by closure
// (what the open shares of the cell's faces leave of its boundary).
fn cut_wall_vector(c: vec3<i32>, axes: i32) -> vec3<f32> {
  let k = cut_at(c);
  var w = vec3<f32>(0.0);
  for (var a = 0; a < axes; a++) {
    w[a] = cut_lo(c, a) - k[a + 1];
  }
  return w;
}

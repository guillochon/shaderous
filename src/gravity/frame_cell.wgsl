// What one cell contributes to the sums a rotating frame that follows a binary keeps its axis by
// (gravity/bh_kick.wgsl follow_binary, gravity/frame_sums.ts): of gas denser than the frame's
// threshold (Masses.track2.z), its mass m and
//   0 m   1 m x   2 m y   3 m x^2   4 m y^2   5 m x y   6 m v_x   7 m v_y
//   8 m x v_x   9 m y v_y   10 m (x v_y + y v_x)
// in the orbital plane (x, y relative to the box center, velocities in the frame), and across it
//   11 m z   12 m v_z
// for holding the center of mass. Requires gravity/bh.wgsl, a binding BH: Masses, and Red
// (gpu/reduce.wgsl) with 13 channels.
fn frame_cell(u1: vec4<f32>, vol: f32, x: vec3<f32>) -> Red {
  var r = red_identity();
  if (!(u1.x > BH.track2.z)) { return r; }
  let m = u1.x * vol;
  let p = u1.yz * vol;
  r[0] = m;
  r[1] = m * x.x;
  r[2] = m * x.y;
  r[3] = m * x.x * x.x;
  r[4] = m * x.y * x.y;
  r[5] = m * x.x * x.y;
  r[6] = p.x;
  r[7] = p.y;
  r[8] = x.x * p.x;
  r[9] = x.y * p.y;
  r[10] = x.x * p.y + x.y * p.x;
  r[11] = m * x.z;
  r[12] = u1.w * vol;
  return r;
}

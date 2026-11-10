// Point-mass perturber, seen from a frame that accelerates with the gas.
// Requires P. P.bh = (G M, softening length, 1 if present, 0).
struct Perturber {
  pos: vec4<f32>,     // position relative to the box centre
  vel: vec4<f32>,
  acc: vec4<f32>,     // its acceleration in this frame
  frame: vec4<f32>,   // acceleration of the frame: the mass-weighted mean pull on the gas
}

fn cell_pos(c: vec3<i32>) -> vec3<f32> {
  return (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
}

// Plummer-softened, so the perturber may pass through the grid.
fn bh_accel(x: vec3<f32>, rb: vec3<f32>) -> vec3<f32> {
  let d = x - rb;
  let r2 = dot(d, d) + P.bh.y * P.bh.y;
  return -P.bh.x * d / (r2 * sqrt(r2));
}

fn bh_phi(x: vec3<f32>, rb: vec3<f32>) -> f32 {
  let d = x - rb;
  return -P.bh.x / sqrt(dot(d, d) + P.bh.y * P.bh.y);
}

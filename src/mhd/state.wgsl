// The primitive state of a cell for MHD, on either grid, and what is computed from it alone.

// Primitive state of a cell for MHD:
//   a = (rho, vx, vy, vz)    b = (p, scalar, sound speed, e_int per volume)    B = cell-centered field
struct MQ { a: vec4<f32>, b: vec4<f32>, B: vec3<f32> }

// Fastest magnetosonic speed of a state, over all directions: sqrt(c_s^2 + B^2 / rho).
fn fast_max(rho: f32, cs: f32, B: vec3<f32>) -> f32 {
  return sqrt(cs * cs + dot(B, B) / rho);
}

// Electric field -v x B at a cell center.
fn efield(v: vec3<f32>, B: vec3<f32>) -> vec3<f32> {
  return cross(B, v);
}

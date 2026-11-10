// Primitive state as the reconstruction sees it, and limited slopes. Requires the limiter.
//   a = (rho, vx, vy, vz)    b = (p, scalar, c_s, 0)
struct Q { a: vec4<f32>, b: vec4<f32> }

fn slope(l: Q, c: Q, r: Q) -> Q {
  return Q(limit4(c.a - l.a, r.a - c.a), limit4(c.b - l.b, r.b - c.b));
}

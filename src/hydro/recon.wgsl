// Shared by the predictor and update passes: primitive loads through the boundary mapping,
// and limited slopes. Requires P, the limiter, and bindings W1, W2.

// Primitive state: a = (rho, vx, vy, vz), b = (p, scalar, c_s, 0).
struct Q { a: vec4<f32>, b: vec4<f32> }

fn load_q(c: vec3<i32>) -> Q {
  let m = bc_map(c);
  let w1 = textureLoad(W1, m.c, 0);
  let w2 = textureLoad(W2, m.c, 0);
  return Q(vec4<f32>(w1.x, bc_velocity(w1.yzw, m)), vec4<f32>(w2.x, w2.z, w2.y, 0.0));
}

fn slope(l: Q, c: Q, r: Q) -> Q {
  return Q(limit4(c.a - l.a, r.a - c.a), limit4(c.b - l.b, r.b - c.b));
}

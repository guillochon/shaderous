// Primitive state as the reconstruction sees it, and its limited reconstruction. Requires the
// limiter.
//   a = (rho, vx, vy, vz)    b = (p, scalar, c_s, 0)
struct Q { a: vec4<f32>, b: vec4<f32> }

// Twice the increment from a cell's centre to its upper face (edge_up) and from its lower face
// to its centre (edge_dn), given the cell and its two neighbours along the axis. A slope
// limiter gives the same for both; a third-order one does not.
//
// The limiter is also given the natural scale of each quantity at the cell, to judge
// smoothness by: density and pressure themselves, the sound speed for velocities. The passive
// scalar has none (zero: always limited).
fn edge_up(l: Q, c: Q, r: Q) -> Q {
  return Q(edge4(c.a - l.a, r.a - c.a, vec4<f32>(c.a.x, c.b.zzz)), edge4(c.b - l.b, r.b - c.b, vec4<f32>(c.b.x, 0.0, c.b.z, 0.0)));
}
fn edge_dn(l: Q, c: Q, r: Q) -> Q {
  if (LIMITER_SYMMETRIC) { return edge_up(l, c, r); }
  return Q(edge4(r.a - c.a, c.a - l.a, vec4<f32>(c.a.x, c.b.zzz)), edge4(r.b - c.b, c.b - l.b, vec4<f32>(c.b.x, 0.0, c.b.z, 0.0)));
}

// Primitive state as the reconstruction sees it, and its limited reconstruction. Requires the
// limiter.
//   a = (rho, vx, vy, vz)    b = (p, scalar, c_s, 1 if the cell is solid)
struct Q { a: vec4<f32>, b: vec4<f32> }

fn q_solid(q: Q) -> bool {
  return q.b.w > 0.5;
}
// The image of a gas cell in a wall normal to `axis`: the same state with the normal velocity
// reversed. Standing in for the solid cells beyond the wall, it makes the wall reflecting.
fn q_mirror(q: Q, axis: i32) -> Q {
  var a = q.a;
  a[axis + 1] = -a[axis + 1];
  return Q(a, vec4<f32>(q.b.xyz, 0.0));
}
// The same for a half-step state, whose b holds flags that the image keeps.
fn pred_mirror(p: Q, axis: i32) -> Q {
  var a = p.a;
  a[axis + 1] = -a[axis + 1];
  return Q(a, p.b);
}

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

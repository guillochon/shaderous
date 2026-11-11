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
//
// Well-balanced reconstruction of the pressure. `dl` and `dr` are the differences of the
// gravitational potential from the lower neighbour to the cell and from the cell to the upper
// neighbour (zero where there is no gravity, or the option is off: then this is the plain
// reconstruction, bit for bit). In hydrostatic balance the pressure follows dp = -rho dphi,
// and it is that balance, not the pressure, that is smooth across a stratified cell. So what
// is limited and reconstructed is each neighbour's deviation from balance with the cell,
//   (p_r - p_c) + (rho_c + rho_r) / 2 * dr      and the same below,
// and the balanced part, -rho_c * dphi to the face, is added back unlimited. Where the gas
// obeys the trapezoid rule p_r - p_c = -(rho_c + rho_r) / 2 * dr exactly, the deviations
// vanish, the two cells sharing a face agree on its pressure, and the pressures of a cell's
// two faces differ by -rho_c (phi_r - phi_l) / 2: what the gravity kick adds to its momentum,
// with the acceleration a centred difference of the potential. The static state then holds
// to round-off, however steep. (After Kappeli & Mishra 2014, with the balance defined by the
// potential differences this code's kicks use.)
//
// `df` is the potential difference towards the face the increment is for: `dr` or `dl`,
// except next to a wall. The cell beyond a wall is the image of this one and has its
// potential, so there is no deviation across the wall (dl or dr is zero), but the wall face
// still lies half a cell down the true potential: df is the true difference there, and its
// negative for the image, so that both arrive at the same pressure on the wall.
fn wb_below(l: Q, c: Q, dl: f32) -> vec4<f32> {
  var d = c.b - l.b;
  d.x += 0.5 * (l.a.x + c.a.x) * dl;
  return d;
}
fn wb_above(c: Q, r: Q, dr: f32) -> vec4<f32> {
  var d = r.b - c.b;
  d.x += 0.5 * (c.a.x + r.a.x) * dr;
  return d;
}
fn edge_up(l: Q, c: Q, r: Q, dl: f32, dr: f32, df: f32) -> Q {
  var b = edge4(wb_below(l, c, dl), wb_above(c, r, dr), vec4<f32>(c.b.x, 0.0, c.b.z, 0.0));
  b.x -= c.a.x * df;
  return Q(edge4(c.a - l.a, r.a - c.a, vec4<f32>(c.a.x, c.b.zzz)), b);
}
fn edge_dn(l: Q, c: Q, r: Q, dl: f32, dr: f32, df: f32) -> Q {
  var b: vec4<f32>;
  if (LIMITER_SYMMETRIC) {
    b = edge4(wb_below(l, c, dl), wb_above(c, r, dr), vec4<f32>(c.b.x, 0.0, c.b.z, 0.0));
  } else {
    b = edge4(wb_above(c, r, dr), wb_below(l, c, dl), vec4<f32>(c.b.x, 0.0, c.b.z, 0.0));
  }
  b.x -= c.a.x * df;
  if (LIMITER_SYMMETRIC) { return Q(edge4(c.a - l.a, r.a - c.a, vec4<f32>(c.a.x, c.b.zzz)), b); }
  return Q(edge4(r.a - c.a, c.a - l.a, vec4<f32>(c.a.x, c.b.zzz)), b);
}

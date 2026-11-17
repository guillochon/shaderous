// The field brushes: change the magnetic field inside the brush without breaking its zero
// divergence. Shared by the uniform and block grids (edit/field_brush.wgsl,
// amr/field_brush.wgsl).
//
// Every change is the curl of a change dA of the vector potential, taken on the faces as the
// circulation of dA around each face over its area, so the divergence of every cell stays what
// it was, to round-off. The potential is the field's own, in the radial (Poincare) gauge about
// the brush's center x0:
//
//   A(x) = integral over t from 0 to 1 of  t B(x0 + t (x - x0)) x (x - x0)  dt,
//
// whose curl is B wherever B is divergence-free (in 2D, fb_potential). Then
//   strength (kind 7): dA = (f - 1) A with f = exp(amount * w), so the field becomes
//                      curl(f A) = f B + grad f x A: scaled by f inside, with the flux it
//                      gains (or loses) closing around the brush's rim;
//   pull (kind 8):     dA = A(x - w d) - A(x), d the drag (B.vel.xyz): the field near the
//                      center is carried by d, and less of it toward the rim.
// w is the brush's weight (edit/brush_core.wgsl).
//
// Edges are integrated at the midpoints of their segments on the finest level of the mesh, at
// positions that are exact in float32, in a fixed order. So every copy of a shared face gets
// the same bits, and a coarse face changes by exactly the mean of the fine faces on it (to
// rounding). An edge on a domain face that is not periodic does not change: the field through
// the boundary stays as it was.
//
// Requires P, B (edit/brush_core.wgsl), loop_count, and from the grid:
//   fb_field(x)   the cell-centered field of the cell under x (fractions of the box, in it)
//   fb_cells()    cells per side of the box on the finest level

// x - x0 across periodic faces the short way, as brush_r measures it.
fn fb_rel(x: vec3<f32>) -> vec3<f32> {
  var d = x - B.pos.xyz;
  if (P.bc_lo.x == 0u) { d.x -= round(d.x); }
  if (P.bc_lo.y == 0u) { d.y -= round(d.y); }
  if (P.dims.z == 1u) { d.z = 0.0; } else if (P.bc_lo.z == 0u) { d.z -= round(d.z); }
  return d;
}

// A point brought into the box: wrapped across periodic faces, clamped at the others.
fn fb_in_box(x: vec3<f32>) -> vec3<f32> {
  var y = x;
  for (var a = 0; a < 3; a++) {
    if (P.bc_lo[a] == 0u) { y[a] -= floor(y[a]); } else { y[a] = clamp(y[a], 0.0, 1.0); }
  }
  if (P.dims.z == 1u) { y.z = 0.0; }
  return y;
}

// The field's potential at x0 + d (radial gauge about the brush's center): the midpoint rule
// in steps of about a finest cell. In 2D nothing varies along z, and the gauge is the plane's:
// the in-plane field's A_z = integral of (B x d)_z dt, without the factor t (the 3D gauge's A_z
// alone would give half of B_x and B_y, the other half coming from a dA_y / dz that the plane
// does not have); B_z's A_x and A_y keep it.
fn fb_potential(d: vec3<f32>) -> vec3<f32> {
  let n = clamp(i32(ceil(length(d) * fb_cells())), 1, 128);
  var s = vec3<f32>(0.0);
  var st = vec3<f32>(0.0);
  for (var i = 0; i < loop_count(n); i++) {
    let t = (f32(i) + 0.5) / f32(n);
    let b = fb_field(fb_in_box(B.pos.xyz + t * d));
    s += b;
    st += t * b;
  }
  let a = cross(st / f32(n), d);
  if (P.dims.z == 1u) { return vec3<f32>(a.xy, cross(s / f32(n), d).z); }
  return a;
}

// Component `b` of dA at the point x (fractions of the box).
fn fb_delta(b: i32, x: vec3<f32>) -> f32 {
  let w = brush_weight(x);
  if (w <= 0.0) { return 0.0; }
  let d = fb_rel(x);
  // Strength: one potential, scaled. Pull: the potential at the point the drag brings here,
  // less the one here. One call site, in a loop the compiler keeps (it inlines at every call).
  let strength = B.kind == 7u;
  let terms = select(2, 1, strength);
  var s = 0.0;
  for (var k = 0; k < loop_count(terms); k++) {
    let drag = select(B.vel.xyz, vec3<f32>(B.vel.xy, 0.0), P.dims.z == 1u);
    let at = select(d, d - w * drag, !strength && k == 0);
    let c = select(select(-1.0, 1.0, k == 0), exp(B.amount * w) - 1.0, strength);
    s += c * fb_potential(at)[b];
  }
  return s;
}

// Whether coordinate v (fractions of the box) of axis a lies on a domain face that is not periodic.
fn fb_on_wall(a: i32, v: f32) -> bool {
  if (a == 2 && P.dims.z == 1u) { return false; }
  return P.bc_lo[a] != 0u && (v <= 0.0 || v >= 1.0);
}

// The integral of dA_b along the edge from the corner o along axis b, of length h, in m
// segments; zero on a wall.
fn fb_edge(b: i32, o: vec3<f32>, h: f32, m: i32) -> f32 {
  for (var a = 0; a < 3; a++) {
    if (a != b && fb_on_wall(a, o[a])) { return 0.0; }
  }
  let hs = h / f32(m);
  var s = 0.0;
  for (var i = 0; i < loop_count(m); i++) {
    var x = o;
    x[b] += (f32(i) + 0.5) * hs;
    s += fb_delta(b, x) * hs;
  }
  return s;
}

// The change of the field through the lower face along `a` of the cell whose lower corner is
// o, cells of side h with m finest segments to an edge: the circulation of dA around the face
// (counterclockwise seen from +a) over its area. In 2D nothing varies along z: the x and y
// faces see only their edges along z (A_z), integrated once, and the z face its in-plane edges.
fn fb_face(a: i32, o: vec3<f32>, h: f32, m: i32) -> f32 {
  let b = (a + 1) % 3;
  let c = (a + 2) % 3;
  let flat = P.dims.z == 1u;
  var circ = 0.0;
  for (var k = 0; k < loop_count(4); k++) {
    // k: 0 along b at c = 0 (+), 1 along c at b = h (+), 2 along b at c = h (-), 3 along c at b = 0 (-)
    let along = select(b, c, (k & 1) == 1);
    let across = select(c, b, (k & 1) == 1);
    var q = o;
    if (k == 1 || k == 2) { q[across] += h; }
    // In 2D an edge offset along z has a twin at the other z that cancels it.
    if (flat && across == 2) { continue; }
    let e = fb_edge(along, q, h, select(m, 1, flat && along == 2));
    circ += select(e, -e, k >= 2);
  }
  return circ / (h * h);
}

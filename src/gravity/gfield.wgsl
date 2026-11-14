// Gravity as the hydro passes see it: the self-gravity field texture written by
// gravity/field.wgsl plus a uniform external acceleration.
// Requires P and a binding `G: texture_3d<f32>`.

fn has_gravity() -> bool {
  return P.grav_on != 0u || any(P.gext.xyz != vec3<f32>(0.0));
}

// Acceleration at an in-range cell.
fn g_load(c: vec3<i32>) -> vec3<f32> {
  var g = P.gext.xyz;
  if (P.grav_on != 0u) { g += textureLoad(G, c, 0).xyz; }
  return g;
}

// Self-gravity potential at the half step, at a cell that may lie one cell outside the grid:
// periodic potentials wrap, isolated ones are extrapolated linearly (consistent with the
// one-sided gradient).
fn phi_self(q: vec3<i32>) -> f32 {
  let n = i32(P.n);
  if (P.grav_bc == 0u) {
    return textureLoad(G, (q + n) % n, 0).w;
  }
  let cl = clamp(q, vec3<i32>(0), vec3<i32>(n - 1));
  let d = cl - q;
  return 2.0 * textureLoad(G, cl, 0).w - textureLoad(G, cl + d, 0).w;
}

// For the well-balanced pressure reconstruction (hydro/q.wgsl): the potential difference
// between the adjacent cells c + k e and c + (k + 1) e along an axis, as (seen, face).
// `seen` is what the deviations from balance are measured against: a cell beyond a boundary
// has the potential of the cell its state is taken from (its mirror image at a reflecting
// face, the edge cell at an outflow face), like the state itself. `face` is what takes the
// pressure from a cell's center to its face: the same, except across a reflecting face,
// where it is the true difference (the potential continued in a straight line), so that the
// pressure on the wall is the hydrostatic one. The external part is exact, as in dphi_half.
fn dphi_step(c: vec3<i32>, axis: i32, k: i32) -> vec2<f32> {
  var e = vec3<i32>(0);
  e[axis] = 1;
  let ia = c[axis] + k;
  let a = bc_map(c + k * e).c;
  let b = bc_map(c + (k + 1) * e).c;
  // Cells apart along the axis: 1, or what the boundary mapping left of it. (Across a periodic
  // face the two map to opposite ends of the grid; they are still one cell apart.)
  var apart = b[axis] - a[axis];
  if (abs(apart) > 1) { apart = 1; }
  var seen = -P.gext[axis] * f32(apart) * P.dx;
  if (P.grav_on != 0u) { seen += textureLoad(G, b, 0).w - textureLoad(G, a, 0).w; }
  var face = seen;
  let n = select(i32(P.n), i32(P.dims.z), axis == 2);
  let at_lo = ia == -1 && P.bc_lo[axis] == 2u;
  let at_hi = ia == n - 1 && P.bc_hi[axis] == 2u;
  if (at_lo || at_hi) {
    // Across the wall: the uniform field as it is, the rest from the step next to it inside.
    face = -P.gext[axis] * P.dx;
    if (P.grav_on != 0u && n > 1) {
      let i0 = c + (k + select(-1, 1, at_lo)) * e;
      face += textureLoad(G, i0 + e, 0).w - textureLoad(G, i0, 0).w;
    }
  }
  return vec2<f32>(seen, face);
}
fn dphi_face(a: vec3<i32>, axis: i32, k: i32) -> f32 {
  return dphi_step(a, axis, k).x;
}
// For flux_between (hydro/flux.wgsl), the face between a and a + e, whose potential step as
// the deviations see it is `across`: the true step, and for each side -1 where that side is
// the image of the other in a reflecting wall of the grid, so that it reaches the wall from its
// own side (as axis_fluxes has it); (across, 1, 1) on every other face, bit for bit.
fn wall_step(a: vec3<i32>, axis: i32, across: f32) -> vec3<f32> {
  if (!well_balanced()) { return vec3<f32>(across, 1.0, 1.0); }
  let t = dphi_step(a, axis, 0).y;
  let n = select(i32(P.n), i32(P.dims.z), axis == 2);
  let image = select(1.0, -1.0, across != t);
  return vec3<f32>(t, select(1.0, image, a[axis] < 0), select(1.0, image, a[axis] + 1 >= n));
}
// The steps below and above cell c along each axis, for the predictor.
fn dphi_cell(c: vec3<i32>, k: i32) -> PhiSteps {
  var s = PhiSteps(vec3<f32>(0.0), vec3<f32>(0.0));
  if (!well_balanced()) { return s; }
  let nax = select(3, 2, P.dims.z == 1u);
  for (var axis = 0; axis < nax; axis++) {
    let d = dphi_step(c, axis, k);
    s.seen[axis] = d.x;
    s.face[axis] = d.y;
  }
  return s;
}

// Half-step potential difference phi(c + e) - phi(c) for a unit offset e. The external part
// is exactly -g . e dx, so it is not formed as a difference of large numbers.
fn dphi_half(c: vec3<i32>, e: vec3<i32>) -> f32 {
  var d = -dot(P.gext.xyz, vec3<f32>(e)) * P.dx;
  if (P.grav_on != 0u) { d += phi_self(c + e) - phi_self(c); }
  return d;
}

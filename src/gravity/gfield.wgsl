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

// Half-step potential difference phi(c + e) - phi(c) for a unit offset e. The external part
// is exactly -g . e dx, so it is not formed as a difference of large numbers.
fn dphi_half(c: vec3<i32>, e: vec3<i32>) -> f32 {
  var d = -dot(P.gext.xyz, vec3<f32>(e)) * P.dx;
  if (P.grav_on != 0u) { d += phi_self(c + e) - phi_self(c); }
  return d;
}

// Readers for the gravity field texture written by gravity/field.wgsl.
// Requires P and a binding `G: texture_3d<f32>`.

// Acceleration at an in-range cell.
fn g_load(c: vec3<i32>) -> vec3<f32> {
  if (P.grav_on == 0u) { return vec3<f32>(0.0); }
  return textureLoad(G, c, 0).xyz;
}

// Half-step potential at a cell that may lie one cell outside the grid: periodic potentials
// wrap, isolated ones are extrapolated linearly (consistent with the one-sided gradient).
fn phi_half(q: vec3<i32>) -> f32 {
  let n = i32(P.n);
  if (P.grav_bc == 0u) {
    return textureLoad(G, (q + n) % n, 0).w;
  }
  let cl = clamp(q, vec3<i32>(0), vec3<i32>(n - 1));
  let d = cl - q;
  return 2.0 * textureLoad(G, cl, 0).w - textureLoad(G, cl + d, 0).w;
}

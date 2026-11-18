// The implicit-pressure step's lagged pressure force for the predictor (hydro/predictor.wgsl),
// as the difference of pressure across the cell that would give it: rho times the mean of the
// two faces' (p_far - p) / rho_f, with the last solve's pressure and rho_f the mean of the two
// cells' densities. That is the face form the correction uses, so in a uniform field a column
// at rest in discrete balance, p_far - p = g dx rho_f on every face, gets exactly the gradient
// that cancels the predictor's gravity kick (a centered difference over the cell's own density
// is off by the density's curvature: 1.6% where it falls by e in four cells). Without a field
// and at uniform density it is the centered difference.
// A wall, or a solid or held neighbor, has no pressure: its face is in balance with the field.
// Across an open face there is no gradient.
@group(0) @binding(30) var PP: texture_3d<f32>;

fn pp_face(c: vec3<i32>, a: i32, up: bool, own: f32, rho: f32) -> f32 {
  let n = vec3<i32>(P.dims.xyz);
  var q = c;
  q[a] += select(-1, 1, up);
  let wall = select(-1.0, 1.0, up) * P.dx * P.gext[a];
  if (q[a] < 0 || q[a] >= n[a]) {
    let k = select(P.bc_lo[a], P.bc_hi[a], up);
    if (k == BC_PERIODIC) {
      q[a] = select(n[a] - 1, 0, up);
    } else if (bc_wall(k)) {
      return wall;
    } else {
      return 0.0;
    }
  }
  if (has_solids() && abs(textureLoad(U2, q, 0).w) > 0.5) { return wall; }
  return (textureLoad(PP, q, 0).x - own) / (0.5 * (rho + textureLoad(U1, q, 0).x));
}

fn pp_gradient(c: vec3<i32>) -> vec3<f32> {
  let p0 = textureLoad(PP, c, 0).x;
  let rho = textureLoad(U1, c, 0).x;
  var g = vec3<f32>(0.0);
  let axes = select(3, 2, P.dims.z == 1u);
  for (var a = 0; a < axes; a++) {
    g[a] = 0.5 * rho * (pp_face(c, a, true, p0, rho) - pp_face(c, a, false, p0, rho));
  }
  return g;
}

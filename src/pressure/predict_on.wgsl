// The implicit-pressure step's lagged pressure gradient for the predictor (hydro/predictor.wgsl):
// the last solve's pressure, differenced across the cell along each axis. Beyond a wall of the
// box, or in a solid or held neighbor, there is no pressure: the cell's own stands for it, carried
// a cell down the uniform field, so that a column at rest has the gradient that holds it (and no
// gradient without a field). Beyond an open face the pressure is the cell's own.
@group(0) @binding(30) var PP: texture_3d<f32>;

fn pp_at(c: vec3<i32>, a: i32, up: bool, own: f32, rho: f32) -> f32 {
  let n = vec3<i32>(P.dims.xyz);
  var q = c;
  q[a] += select(-1, 1, up);
  let wall = own + select(-1.0, 1.0, up) * P.dx * P.gext[a] * rho;
  if (q[a] < 0 || q[a] >= n[a]) {
    let k = select(P.bc_lo[a], P.bc_hi[a], up);
    if (k == BC_PERIODIC) {
      q[a] = select(n[a] - 1, 0, up);
    } else if (bc_wall(k)) {
      return wall;
    } else {
      return own;
    }
  }
  if (has_solids() && abs(textureLoad(U2, q, 0).w) > 0.5) { return wall; }
  return textureLoad(PP, q, 0).x;
}

fn pp_gradient(c: vec3<i32>) -> vec3<f32> {
  let p0 = textureLoad(PP, c, 0).x;
  let rho = textureLoad(U1, c, 0).x;
  var g = vec3<f32>(0.0);
  let axes = select(3, 2, P.dims.z == 1u);
  for (var a = 0; a < axes; a++) {
    g[a] = 0.5 * (pp_at(c, a, true, p0, rho) - pp_at(c, a, false, p0, rho));
  }
  return g;
}

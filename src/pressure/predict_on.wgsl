// The implicit-pressure step's lagged pressure gradient for the predictor (hydro/predictor.wgsl):
// the last solve's pressure, differenced across the cell along each axis. A neighbor beyond a
// face of the box is the cell that stands for it (no gradient across a wall or an open face); a
// solid or held neighbor has no pressure of its own, and counts as the cell's.
@group(0) @binding(30) var PP: texture_3d<f32>;

fn pp_at(c: vec3<i32>, own: f32) -> f32 {
  let m = bc_map(c);
  if (has_solids() && abs(textureLoad(U2, m.c, 0).w) > 0.5) { return own; }
  return textureLoad(PP, m.c, 0).x;
}

fn pp_gradient(c: vec3<i32>) -> vec3<f32> {
  let p0 = textureLoad(PP, c, 0).x;
  var g = vec3<f32>(
    0.5 * (pp_at(c + vec3<i32>(1, 0, 0), p0) - pp_at(c - vec3<i32>(1, 0, 0), p0)),
    0.5 * (pp_at(c + vec3<i32>(0, 1, 0), p0) - pp_at(c - vec3<i32>(0, 1, 0), p0)),
    0.0,
  );
  if (P.dims.z != 1u) { g.z = 0.5 * (pp_at(c + vec3<i32>(0, 0, 1), p0) - pp_at(c - vec3<i32>(0, 0, 1), p0)); }
  return g;
}

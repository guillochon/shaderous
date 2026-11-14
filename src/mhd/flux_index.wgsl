// Layout of the MHD face-flux buffers (mhd/flux.wgsl): one entry per cell t in [-1, n] along
// each axis (the single layer along z in 2D), three vec4 per entry. Requires P.

// The flux grid: two more than the cells along each axis (none more along z in 2D).
fn flux_dims() -> vec3<i32> {
  let n = vec3<i32>(P.dims.xyz);
  return select(n + 2, vec3<i32>(n.xy + 2, 1), n.z == 1);
}
fn flux_offset() -> vec3<i32> {
  return select(vec3<i32>(1), vec3<i32>(1, 1, 0), P.dims.z == 1u);
}
// Entry of cell t. Along z in 2D every index is the single layer.
fn flux_index(t: vec3<i32>) -> u32 {
  let d = flux_dims();
  var s = t + flux_offset();
  if (P.dims.z == 1u) { s.z = 0; }
  return u32((s.z * d.y + s.y) * d.x + s.x);
}

// The edge-field buffer (mhd/emf.wgsl): one vec4 (E_x, E_y, E_z, 0) per texel t in [0, n] along
// each axis (the single layer along z in 2D), on the cell's lower edges: E_x at
// (i, j - 1/2, k - 1/2), E_y at (i - 1/2, j, k - 1/2), E_z at (i - 1/2, j - 1/2, k).
fn emf_dims() -> vec3<i32> {
  let n = vec3<i32>(P.dims.xyz);
  return select(n + 1, vec3<i32>(n.xy + 1, 1), n.z == 1);
}
fn emf_index(t: vec3<i32>) -> u32 {
  let d = emf_dims();
  var s = t;
  if (P.dims.z == 1u) { s.z = 0; }
  return u32((s.z * d.y + s.y) * d.x + s.x);
}

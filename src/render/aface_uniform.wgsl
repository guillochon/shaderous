// The flux function's face field (render/aflux.wgsl, the <aface> slot), uniform grid: the grid's
// own faces. Face texture BF (mhd/common.wgsl): texel t holds the lower faces of cell t, and
// texel n the grid's upper faces, except along a periodic axis, where that face is cell 0's lower.
@group(0) @binding(13) var BF: texture_3d<f32>;

// b_x on the face at x index gi (0 to n) of the cells in row gj; b_y likewise along y.
fn bx_face(gi: u32, gj: u32) -> f32 {
  let n = P.dims.x;
  let i = select(gi, 0u, gi >= n && P.bc_lo[0] == 0u);
  return textureLoad(BF, vec3<u32>(i, gj, 0u), 0).x;
}
fn by_face(gi: u32, gj: u32) -> f32 {
  let n = P.dims.y;
  let j = select(gj, 0u, gj >= n && P.bc_lo[1] == 0u);
  return textureLoad(BF, vec3<u32>(gi, j, 0u), 0).y;
}

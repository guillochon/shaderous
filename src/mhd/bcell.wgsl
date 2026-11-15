// The cell-centered magnetic field on the uniform grid, from the face texture BF
// (mhd/common.wgsl describes its layout). Requires P and a binding BF. Included by
// mhd/common.wgsl, and by passes that read the field without the rest of it
// (gravity/correct.wgsl).

// Cell-centered field of an in-range cell. Its faces are texels c and c + e along each axis,
// and need none of face_texel's mapping but one: the grid's upper face along a periodic axis is
// cell 0's lower face. (The same texels as bface() would read, so the same bits; but bface()'s
// mapping, six times over, made every shader reading the field slow to compile: the edge
// fields' four or seven cells took 0.45 s in Chrome, 0.1 s without it.)
fn bcell(c: vec3<i32>) -> vec3<f32> {
  let n = vec3<i32>(P.dims.xyz);
  let up = c + 1;
  let wrap = vec3<bool>(up.x >= n.x && P.bc_lo[0] == 0u, up.y >= n.y && P.bc_lo[1] == 0u, up.z >= n.z && P.bc_lo[2] == 0u);
  let hi = select(up, vec3<i32>(0), wrap);
  let lo = textureLoad(BF, c, 0);
  return 0.5 * vec3<f32>(
    lo.x + textureLoad(BF, vec3<i32>(hi.x, c.y, c.z), 0).x,
    lo.y + textureLoad(BF, vec3<i32>(c.x, hi.y, c.z), 0).y,
    lo.z + textureLoad(BF, vec3<i32>(c.x, c.y, hi.z), 0).z,
  );
}

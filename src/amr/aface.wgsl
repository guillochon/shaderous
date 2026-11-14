// The flux function's face field (render/aflux.wgsl, the <aface> slot), block grid: at a face of
// the finest level, the leaf's own field there. In a coarser leaf the face may lie inside one of
// its cells, where the field's normal component is interpolated linearly between the cell's two
// faces along that axis (constant along the other); with the cell's faces divergence-free, so is
// that field. A face on a cell's side is the cell's face itself, which across a refinement
// boundary is the mean of the finer faces on it.
#include "amr/grid.wgsl"

@group(0) @binding(5) var IDX: texture_3d<u32>;   // leaf slot per deepest-level block
@group(0) @binding(13) var BF: texture_3d<f32>;

// Finest cells per side.
fn n_fine() -> u32 {
  return (P.amr.y << P.amr.z) * u32(BS);
}

// Component a of the field at the face of the finest level at index g along a (0 to n) of the
// cells c along the other axis (in finest cells).
fn face_at(a: i32, g: u32, c: u32) -> f32 {
  let n = n_fine();
  // The finest cell the face is the lower side of (its last cell's upper side at the top).
  var cell = vec2<u32>(min(g, n - 1u), c);
  if (a == 1) { cell = vec2<u32>(c, min(g, n - 1u)); }
  let slot = i32(textureLoad(IDX, vec3<u32>(cell / u32(BS), 0u), 0).x);
  let shift = P.amr.z - u32(bt(slot, 0u));
  let s = 1u << shift;
  let coarse = cell >> vec2<u32>(shift);
  let l = vec3<i32>(vec2<i32>(coarse) - blk_coord(slot).xy * BS, 0);
  // Where along the coarse cell the face lies: 0 on its lower face, 1 on its upper.
  let f = f32(g - coarse[a] * s) / f32(s);
  var e = vec3<i32>(0);
  e[a] = 1;
  let o = blk_origin(slot);
  return mix(textureLoad(BF, atl(o, l), 0)[a], textureLoad(BF, atl(o, l + e), 0)[a], f);
}

fn bx_face(gi: u32, gj: u32) -> f32 {
  return face_at(0, gi, gj);
}
fn by_face(gi: u32, gj: u32) -> f32 {
  return face_at(1, gj, gi);
}

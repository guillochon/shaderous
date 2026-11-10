// One colour of a red-black Gauss-Seidel sweep over the blocks of one tree level, in place.
// The colouring is global (parity of the cell's index across the level), so cells of one
// colour only ever read the other, also across block faces.
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/gravity/common.wgsl"
#include "gravity/multipole_eval.wgsl"

@group(0) @binding(2) var PHI: texture_storage_3d<r32float, read_write>;
@group(0) @binding(3) var RHS: texture_3d<f32>;

#include "amr/gravity/stencil.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = g_inv(gid);
  if (!inv.ok) { return; }
  let g = blk_coord(inv.slot) * GL.bsize + inv.l;
  if (u32((g.x + g.y + g.z) & 1) != GL.color) { return; }
  let o = g_origin(inv.slot);
  let s = g_sum(inv.slot, o, inv.l);
  let h = g_h(inv.slot);
  let f = textureLoad(RHS, o + inv.l, 0).x;
  textureStore(PHI, o + inv.l, vec4<f32>((s.x - h * h * f) / s.y, 0.0, 0.0, 0.0));
}

// Edge impulses (dt/dx) E on the block grid, for one stage (mhd/emf.wgsl on the uniform grid):
// one invocation per texel t in [0, BS] along each axis (the single layer along z in 2D),
// from the chunk's flux scratch and the block's assembled cells. E_c is formed where t lies
// on an edge along c that bounds the block's own faces: t_c < BS. The result goes to the edge
// atlas, where a coarser neighbor reads the block's boundary edges (mhd/amr_update.wgsl).
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>
#include <diffusion>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;
@group(0) @binding(3) var W2: texture_3d<f32>;
@group(0) @binding(4) var W3: texture_3d<f32>;
@group(0) @binding(6) var<storage, read> FL: array<vec4<f32>>;
@group(0) @binding(8) var EO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(14) var<uniform> CHUNK: Chunk;

override STAGE: u32 = 1u;

#include "mhd/state.wgsl"
#include "mhd/amr_block.wgsl"

var<private> g_row: u32;

fn fc(a: i32, t: vec3<i32>) -> vec3<f32> {
  return FL[flux_entry(g_row, a, t) + 2u].xyz;
}

#include "mhd/corner.wgsl"

var<private> g_o: vec3<i32>;

fn eref(l: vec3<i32>) -> Ref {
  let q = mq_block(g_o, l);
  return Ref(efield(q.a.yzw, q.B), q.B);
}

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let inv = inv_box(wid, lix, 0, BS + 1);
  if (!inv.ok) { return; }
  let t = inv.l;
  g_row = inv.row;
  g_o = blk_origin(inv.slot);
  g_dx = blk_dx(inv.slot);
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);
  var E = vec3<f32>(0.0);
  // Resistivity (mhd/ohm.wgsl): eta J, as dx J.
  var J = vec3<f32>(0.0);
  let r111 = eref(t);
  let r100 = eref(t - ey);
  let r010 = eref(t - ex);
  let r000 = eref(t - ex - ey);
  if (is2d()) {
    // E_z at the vertices; E_x on the y faces (-F_y(B_z)), E_y on the x faces (F_x(B_z)).
    E.z = corner(2, t, r000, r100, r010, r111);
    if (t.x < BS) { E.x = -fc(1, t).x; }
    if (t.y < BS) { E.y = fc(0, t).y; }
    if (RESISTIVITY) { J = vec3<f32>(r111.b.z - r100.b.z, r010.b.z - r111.b.z, edge_current(2, r000, r100, r010, r111)); }
  } else {
    let rz0 = eref(t - ez);
    let ryz = eref(t - ey - ez);
    let rzx = eref(t - ez - ex);
    if (t.z < BS) { E.z = corner(2, t, r000, r100, r010, r111); }
    if (t.x < BS) { E.x = corner(0, t, ryz, rz0, r100, r111); }
    if (t.y < BS) { E.y = corner(1, t, rzx, r010, rz0, r111); }
    if (RESISTIVITY) { J = vec3<f32>(edge_current(0, ryz, rz0, r100, r111), edge_current(1, rzx, r010, rz0, r111), edge_current(2, r000, r100, r010, r111)); }
  }
  if (RESISTIVITY) {
    E += select(vec3<f32>(0.0), (P.ohm.x / g_dx) * J, vec3<bool>(t.x < BS, t.y < BS, t.z < BS || is2d()));
  }
  let dt = stage_dt(inv.slot, STAGE);
  textureStore(EO, atl(g_o, t), vec4<f32>((dt / g_dx) * E, 0.0));
}

// Radiation on re-meshing (after amr/regrid.wgsl): every leaf of the new tree from the old one,
// as the state is. An unchanged block is copied; a merged block takes the mean of its old
// children; a new child takes its old parent's cell plus minmod slopes toward its quarter, from
// the parent's neighbors with ghosts (RA, assembled on the old mesh just before), so E and F are
// conserved, E stays positive and F within c E.
#include "hydro/common.wgsl"
#include "radiation/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(2) var RI: texture_3d<f32>;    // radiation on the old mesh
@group(0) @binding(3) var RA: texture_3d<f32>;    // the same with ghosts
@group(0) @binding(6) var RO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> RG: array<i32>;

const RG_STRIDE = 12u;

fn minmod4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return select(vec4<f32>(0.0), sign(a) * min(abs(a), abs(b)), a * b > vec4<f32>(0.0));
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let l = inv.l;
  let r = u32(inv.slot) * RG_STRIDE;
  let kind = RG[r];
  let src = RG[r + 1u];
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  var u: vec4<f32>;
  if (kind == 2) {
    let oct = RG[r + 2u];
    let lc = (l >> vec3<u32>(1u)) + (BS / 2) * vec3<i32>(oct & 1, (oct >> 1u) & 1, (oct >> 2u) & 1);
    let o = blk_origin(src);
    let x = textureLoad(RA, atl(o, lc), 0);
    let off = vec3<f32>(l & vec3<i32>(1)) - 0.5;
    u = textureLoad(RI, atl(o, lc), 0);
    let axes = select(3, 2, is2d());
    for (var a = 0; a < axes; a++) {
      var e = vec3<i32>(0);
      e[a] = 1;
      u += 0.5 * off[a] * minmod4(x - textureLoad(RA, atl(o, lc - e), 0), textureLoad(RA, atl(o, lc + e), 0) - x);
    }
    u.x = max(u.x, 0.0);
    let fm = length(u.yzw);
    if (fm > rad_c() * u.x) { u = vec4<f32>(u.x, u.yzw * (rad_c() * u.x / fm)); }
  } else if (kind == 3) {
    let ch = RG[r + 4u + u32(h.x | (h.y << 1u) | (h.z << 2u))];
    let o = blk_origin(ch);
    let f = 2 * (l - (BS / 2) * h);
    let nz = select(2, 1, is2d());
    u = vec4<f32>(0.0);
    for (var k = 0; k < nz; k++) {
      for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) { u += textureLoad(RI, atl(o, f + vec3<i32>(i, j, k)), 0); }
      }
    }
    u *= select(0.125, 0.25, is2d());
  } else {
    u = textureLoad(RI, atl(blk_origin(src), l), 0);
  }
  textureStore(RO, atl(blk_origin(inv.slot), l), u);
}

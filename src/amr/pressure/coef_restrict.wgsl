// A gas's coefficients on the parents of a tree level (amr/pressure/coef_gas.wgsl): each
// parent cell takes the mean of its children's density and of their 1 / (rho c^2). Run over
// the levels from the deepest parents down, once a step: a child that is a leaf gives the
// advected state's primitives, one that is a parent what the pass before this one left.
#include "amr/pressure/common.wgsl"

@group(0) @binding(15) var W1: texture_3d<f32>;
@group(0) @binding(22) var W2: texture_3d<f32>;
@group(0) @binding(7) var KR: texture_storage_3d<r32float, read_write>;
@group(0) @binding(8) var KA: texture_storage_3d<r32float, read_write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok || blk_leaf(inv.slot)) { return; }
  let l = inv.l;
  let o = l / (BS / 2);
  let child = bt(inv.slot, 12u + u32(o.x | (o.y << 1u) | (o.z << 2u)));
  let leaf = blk_leaf(child);
  let f0 = 2 * (l - o * (BS / 2));
  let nk = select(8, 4, is2d());
  var k = vec2<f32>(0.0);
  for (var i = 0; i < nk; i++) {
    let cf = atl(blk_origin(child), f0 + vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1));
    if (leaf) {
      let rho = abs(textureLoad(W1, cf, 0).x);
      let cs = textureLoad(W2, cf, 0).y;
      k += vec2<f32>(rho, 1.0 / (rho * cs * cs));
    } else {
      k += vec2<f32>(textureLoad(KR, cf).x, textureLoad(KA, cf).x);
    }
  }
  k /= f32(nk);
  let c = atl(blk_origin(inv.slot), l);
  textureStore(KR, c, vec4<f32>(k.x, 0.0, 0.0, 0.0));
  textureStore(KA, c, vec4<f32>(k.y, 0.0, 0.0, 0.0));
}

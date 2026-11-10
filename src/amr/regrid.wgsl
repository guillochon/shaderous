// Re-meshing: fills every leaf of the new tree from the old one, into the other state texture.
//   copy      the block is unchanged
//   prolong   a new child: conservative, limited linear interpolation from its old parent
//   restrict  a merged block: the average of its old children
// The slopes for prolongation come from the primitives of the previous step, which include the
// old parent's ghost cells.
#include "hydro/common.wgsl"
#include <limiter>
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var W1: texture_3d<f32>;
@group(0) @binding(4) var W2: texture_3d<f32>;
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> RG: array<i32>;

#include "amr/cons.wgsl"

const RG_STRIDE = 12u;

// Conserved state of a cell (ghosts included) rebuilt from the stored primitives.
fn cons_from_w(c: vec3<i32>) -> MaybeCons {
  let w1 = textureLoad(W1, c, 0);
  let w2 = textureLoad(W2, c, 0);
  let m = w1.x * w1.yzw;
  return MaybeCons(Cons(vec4<f32>(w1.x, m), vec4<f32>(w2.w + 0.5 * dot(m, w1.yzw), w2.w, w1.x * w2.z, 0.0)), true);
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
  var u: Cons;
  if (kind == 2) {
    let oct = RG[r + 2u];
    let lc = (l >> vec3<u32>(1u)) + (BS / 2) * vec3<i32>(oct & 1, (oct >> 1u) & 1, (oct >> 2u) & 1);
    let o = blk_origin(src);
    let c = atl(o, lc);
    let uc = Cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
    let ex = vec3<i32>(1, 0, 0);
    let ey = vec3<i32>(0, 1, 0);
    var zm = MaybeCons(uc, false);
    var zp = zm;
    if (!is2d()) {
      zm = cons_from_w(c - vec3<i32>(0, 0, 1));
      zp = cons_from_w(c + vec3<i32>(0, 0, 1));
    }
    u = prolong(uc, cons_from_w(c - ex), cons_from_w(c + ex), cons_from_w(c - ey), cons_from_w(c + ey), zm, zp, l & vec3<i32>(1));
  } else if (kind == 3) {
    let ch = RG[r + 4u + u32(h.x | (h.y << 1u) | (h.z << 2u))];
    let o = blk_origin(ch);
    let f = 2 * (l - (BS / 2) * h);
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    let nz = select(2, 1, is2d());
    for (var k = 0; k < nz; k++) {
      for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
          let c = atl(o, f + vec3<i32>(i, j, k));
          a += textureLoad(U1, c, 0);
          b += textureLoad(U2, c, 0);
        }
      }
    }
    let w = select(0.125, 0.25, is2d());
    u = Cons(a * w, b * w);
  } else {
    let c = atl(blk_origin(src), l);
    u = Cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
  }
  let c = atl(blk_origin(inv.slot), l);
  textureStore(O1, c, u.a);
  textureStore(O2, c, u.b);
}

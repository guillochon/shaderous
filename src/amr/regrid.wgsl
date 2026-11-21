// Re-meshing: fills every leaf of the new tree from the old one, into the other state texture.
//   copy      the block is unchanged
//   prolong   a new child: conservative, limited linear interpolation from its old parent
//   restrict  a merged block: the average of its old children
// A new child near an obstacle's surface (amr/bodies.ts) takes its solid cells from the body's
// shape at its own level (PAINT), not from its parent's: a cell the shape covers becomes solid
// and at rest, and one it does not, that was solid in a body of the shape's, becomes gas (the gas
// kept inside, at rest).
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
// [0] where the marks start, [1] the bodies' marks as bits, [2 + slot] 1 + the block's place
// among the painted (0: not painted), then four marks to a word per painted block.
@group(0) @binding(8) var<storage, read> PAINT: array<u32>;

#include "amr/cons.wgsl"

const RG_STRIDE = 12u;

// Conserved state of a cell (ghosts included) rebuilt from the stored primitives.
fn cons_from_w(c: vec3<i32>) -> MaybeCons {
  var w1 = textureLoad(W1, c, 0);
  w1.x = abs(w1.x);   // negative marks a solid cell
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
    if (cylindrical()) { g_cyl_ic = f32(blk_coord(src).x * BS + lc.x) + 0.5; }
    if (spherical()) { g_sph_pro = sph_centroid(bt(src, 0u), (blk_coord(src) * BS + lc).xy); }
    let s = prolong_slopes(uc, cons_from_w(c - ex), cons_from_w(c + ex), cons_from_w(c - ey), cons_from_w(c + ey), zm, zp);
    u = prolong_child(uc, s, l & vec3<i32>(1));
    // The total energy: each child's own kinetic energy and its interpolated internal energy,
    // plus one offset shared by all the children that keeps their mean the coarse cell's.
    // Interpolated on its own, E carries its own slope, and E - KE in a child is then off by
    // terms of the order of the kinetic energy's variation over the coarse cell: in cold gas
    // moving fast, many times its internal energy. Where the flow converges E is little more than
    // KE, so the dual-energy switch (hydro/dual_energy.wgsl) takes that difference as heat: a
    // Zel'dovich pancake refined at its center was heated a hundred-million-fold there, and its
    // density was off by 6.5% at a = 0.05 against 0.6% without the re-mesh. The offset is the
    // kinetic energy the children's spread adds to the coarse cell's (never positive, as KE is
    // convex in density and momentum), taken from the heat: where that leaves E - KE below the
    // switch, the interpolated internal energy carries on.
    let nz = select(2, 1, is2d());
    var ke = 0.0;
    for (var k = 0; k < nz; k++) {
      for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
          let ch = prolong_child(uc, s, vec3<i32>(i, j, k));
          ke += child_weight(src, lc, i, j) * 0.5 * dot(ch.a.yzw, ch.a.yzw) / max(ch.a.x, P.rho_floor);
        }
      }
    }
    ke *= select(0.125, 0.25, is2d());
    u.b.x = 0.5 * dot(u.a.yzw, u.a.yzw) / max(u.a.x, P.rho_floor) + u.b.y + (uc.b.x - uc.b.y - ke);
    let pb = PAINT[2u + u32(inv.slot)];
    if (pb > 0u) {
      let li = u32((l.z * BS + l.y) * BS + l.x);
      let words = u32(BS * BS * select(BS, 1, is2d())) / 4u;
      let m = (PAINT[PAINT[0] + (pb - 1u) * words + (li >> 2u)] >> (8u * (li & 3u))) & 255u;
      if (m > 0u) {
        // Covered: at rest, as the gas kept inside an obstacle is.
        u.b.x -= 0.5 * dot(u.a.yzw, u.a.yzw) / max(u.a.x, P.rho_floor);
        u.a = vec4<f32>(u.a.x, 0.0, 0.0, 0.0);
        u.b.w = f32(m);
      } else if (u.b.w >= 0.5 && ((PAINT[1] >> u32(round(u.b.w))) & 1u) != 0u) {
        u.b.w = 0.0;
      }
    }
  } else if (kind == 3) {
    let ch = RG[r + 4u + u32(h.x | (h.y << 1u) | (h.z << 2u))];
    let o = blk_origin(ch);
    let f = 2 * (l - (BS / 2) * h);
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    let nz = select(2, 1, is2d());
    // (An axisymmetric or spherical run: volume-weighted, amr/grid.wgsl child_weight.)
    for (var k = 0; k < nz; k++) {
      for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
          let c = atl(o, f + vec3<i32>(i, j, k));
          let wr = child_weight(inv.slot, l, i, j);
          a += wr * textureLoad(U1, c, 0);
          b += wr * textureLoad(U2, c, 0);
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

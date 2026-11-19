// Reduction source for the point masses on the block grid, over leaf cells: gas mass, each
// mass's total pull on the gas, and what each sink takes this step (gravity/bh_cell.wgsl).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include <bins>
#include "amr/blk_dt.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> BH: Masses;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var<storage, read> clock: Clock;

#include "gravity/bh_cell.wgsl"

// A leaf cell's mark (amr/pressure/common.wgsl ap_mark).
fn rc_mark(slot: i32, l: vec3<i32>) -> f32 {
  if (!has_solids() || !blk_leaf(slot)) { return 0.0; }
  return textureLoad(U2, atl(blk_origin(slot), l), 0).w;
}
// One face's kick by mass b, per unit time: between a lower and an upper cell (or ghost) of
// cell size h (amr/pressure/faces_gas.wgsl g_face).
fn rc_face(b: Mass, xlo: vec3<f32>, xhi: vec3<f32>, a: i32, h: f32) -> f32 {
  let len = f32(P.amr.y) * f32(BS) * P.dx;
  if (xlo[a] < P.org[a] || xhi[a] > P.org[a] + len) {
    var xl = xlo;
    var xh = xhi;
    if (xl[a] < P.org[a]) { xl[a] += len; }
    if (xh[a] > P.org[a] + len) { xh[a] -= len; }
    return 0.5 * (bh_accel(xl, b)[a] + bh_accel(xh, b)[a]);
  }
  return -bh_dphi(b, xlo, xhi - xlo, 0.5 * clock.dt) / h;
}
// A mass's pull on a leaf cell's gas, per unit mass: the cell center's, or with implicit
// pressure the mean of its faces' kicks as the correction formed them (gravity/bh_moment.wgsl
// has why; amr/pressure/faces_gas.wgsl ap_gface is what this follows, kind for kind).
fn bh_pull(x: vec3<f32>, b: Mass) -> vec3<f32> {
  let at = bh_accel(x, b);
  if (!(P.ohm.y > 0.0)) { return at; }
  let slot = g_rc.w;
  let l = g_rc.xyz;
  let h = blk_dx(slot);
  var g = vec3<f32>(0.0);
  let axes = select(3, 2, is2d());
  for (var a = 0; a < axes; a++) {
    for (var u = 0; u < 2; u++) {
      let up = u == 1;
      var e = vec3<i32>(0);
      e[a] = select(-1, 1, up);
      // What lies across the face (amr/pressure/common.wgsl ap_nb): 0 a cell of this level
      // (ns, nl; `fine` if that block is refined), 1 a coarser leaf's cell, 2 nothing to kick
      // through.
      var kind = 0;
      var ns = slot;
      var nl = l + e;
      var fine = false;
      if (nl[a] < 0 || nl[a] >= BS) {
        let f = u32(2 * a) + u32(up);
        let nb = bt(slot, 6u + f);
        if (nb >= 0) {
          ns = nb;
          nl[a] += select(BS, -BS, up);
          fine = !blk_leaf(nb);
        } else if (nb == -1) {
          kind = 1;
          ns = bt(slot, 20u + f);
          nl = ((blk_coord(slot) * BS + l + e) >> vec3<u32>(1u)) - blk_coord(ns) * BS;
          nl[a] = select(BS - 1, 0, up);
          if (is2d()) { nl.z = 0; }
        } else {
          kind = 2;
        }
      }
      if (kind != 2 && abs(rc_mark(ns, nl)) > 0.5) { kind = 2; }
      var k = at[a];
      if (kind == 0 && fine) {
        // The fine faces that tile this one (ap_fine), each between a fine cell and its ghost.
        k = 0.0;
        let nf = select(4, 2, is2d());
        for (var t = 0; t < nf; t++) {
          var gq = 2 * (blk_coord(ns) * BS + nl);
          gq[a] += select(1, 0, up);
          let b1 = (a + 1) % 3;
          let b2 = (a + 2) % 3;
          gq[b1] += t & 1;
          if (!is2d()) { gq[b2] += (t >> 1) & 1; } else if (b1 == 2) { gq[b2] += t & 1; gq[b1] -= t & 1; }
          let fb = gq >> vec3<u32>(4u);
          let oct = (fb.x & 1) | ((fb.y & 1) << 1u) | ((fb.z & 1) << 2u);
          let fs = bt(ns, 12u + u32(oct));
          let fl = gq - fb * BS;
          if (abs(rc_mark(fs, fl)) > 0.5) {
            k += at[a];
          } else if (up) {
            k += rc_face(b, blk_pos(fs, fl - e), blk_pos(fs, fl), a, 0.5 * h);
          } else {
            k += rc_face(b, blk_pos(fs, fl), blk_pos(fs, fl - e), a, 0.5 * h);
          }
        }
        k /= f32(nf);
      } else if (kind != 2) {
        if (up) { k = rc_face(b, x, blk_pos(slot, l + e), a, h); } else { k = rc_face(b, blk_pos(slot, l + e), x, a, h); }
      }
      g[a] += 0.5 * k;
    }
  }
  return g;
}

fn red_cell(c: vec3<i32>) -> Red {
  let slot = c.x / BS;
  let l = vec3<i32>(c.x % BS, c.y, c.z);
  g_rc = vec4<i32>(l, slot);
  let at = atl(blk_origin(slot), l);
  var u1 = textureLoad(U1, at, 0);
  if (!blk_leaf(slot) || abs(textureLoad(U2, at, 0).w) > 0.5) { u1 = vec4<f32>(0.0); }
  let dx = blk_dx(slot);
  // 2D quantities are per unit length along z.
  // With per-block timesteps the sinks take gas from a block as it finishes a step, over the
  // length of that step (amr/correct.wgsl); the pull on the gas is summed over every cell.
  let dt = select(0.0, blk_dt(slot), blk_active(slot));
  // (An axisymmetric run's cell is a ring of volume 2 pi R dx^2.)
  var vol = select(dx * dx * dx, dx * dx, is2d());
  if (cylindrical()) { vol = 6.2831853 * (f32(blk_coord(slot).x * BS + l.x) + 0.5) * dx * dx * dx; }
  return bh_cell(u1, vol, blk_pos(slot, l), dt);
}

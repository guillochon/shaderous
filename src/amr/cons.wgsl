// Conserved state on the block grid. Requires amr/grid.wgsl, the limiter, and bindings U1, U2.
//   a = (rho, mx, my, mz)    b = (E, e_int, rho * scalar, 0)
struct Cons { a: vec4<f32>, b: vec4<f32> }

fn u_leaf(slot: i32, l: vec3<i32>) -> Cons {
  let c = atl(blk_origin(slot), l);
  return Cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
}

// Interior cell l of any block. A leaf stores it. A parent stores nothing: its value is the
// average of the child cells covering it (conservative restriction, done on the fly).
fn u_at(slot: i32, l: vec3<i32>) -> Cons {
  if (blk_leaf(slot)) { return u_leaf(slot, l); }
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  let ch = bt(slot, 12u + u32(h.x | (h.y << 1u) | (h.z << 2u)));
  if (ch < 0 || !blk_leaf(ch)) {
    // The child is refined further. 2:1 balance guarantees no leaf's stencil reaches here.
    return Cons(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 0.0, 0.0));
  }
  let f = 2 * (l - (BS / 2) * h);
  var a = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  let nz = select(2, 1, is2d());
  for (var k = 0; k < nz; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        let u = u_leaf(ch, f + vec3<i32>(i, j, k));
        a += u.a;
        b += u.b;
      }
    }
  }
  let w = select(0.125, 0.25, is2d());
  return Cons(a * w, b * w);
}

// A cell of block `slot` that may lie one cell outside it: used for interpolation slopes.
// Returns ok = false where the value would need a second level of coarsening.
struct MaybeCons { u: Cons, ok: bool }

fn u_near(slot: i32, l: vec3<i32>) -> MaybeCons {
  let s = ghost_src(slot, l);
  if (s.kind != 0) { return MaybeCons(Cons(vec4<f32>(0.0), vec4<f32>(0.0)), false); }
  var u = u_at(s.slot, s.l);
  u.a = vec4<f32>(u.a.x, u.a.yzw * s.flip);
  return MaybeCons(u, true);
}

// Conservative, limited linear interpolation from a coarse cell to one of the fine cells it
// covers. `par` is the fine cell's parity (0 or 1) along each axis.
fn prolong(uc: Cons, xm: MaybeCons, xp: MaybeCons, ym: MaybeCons, yp: MaybeCons, zm: MaybeCons, zp: MaybeCons, par: vec3<i32>) -> Cons {
  var out = uc;
  let off = (vec3<f32>(par) - 0.5) * 0.5;   // -1/4 or +1/4 of a coarse cell
  if (xm.ok && xp.ok) {
    out.a += off.x * limit4(uc.a - xm.u.a, xp.u.a - uc.a);
    out.b += off.x * limit4(uc.b - xm.u.b, xp.u.b - uc.b);
  }
  if (ym.ok && yp.ok) {
    out.a += off.y * limit4(uc.a - ym.u.a, yp.u.a - uc.a);
    out.b += off.y * limit4(uc.b - ym.u.b, yp.u.b - uc.b);
  }
  if (zm.ok && zp.ok && !is2d()) {
    out.a += off.z * limit4(uc.a - zm.u.a, zp.u.a - uc.a);
    out.b += off.z * limit4(uc.b - zm.u.b, zp.u.b - uc.b);
  }
  return out;
}

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

// Slope of the momentum along one axis, built from limited slopes of the density and of the
// velocity (the product rule) and not from the momentum's own differences. Where the density
// has a local minimum beside much denser gas, its slope is limited to zero while the momentum,
// which mostly follows the density, still has a large one-sided slope. The fine cells then
// share one density but get momenta far apart: velocities many times any real one, and a
// coarse cell that feeds them through the matched fluxes loses momentum faster each step.
// Built this way a fine cell's velocity differs from the coarse one by at most the limited
// velocity slope times the density ratio, which the scaling in prolong() keeps under two.
fn mom_slope(uc: vec4<f32>, um: vec4<f32>, up: vec4<f32>, srho: f32) -> vec3<f32> {
  let vc = uc.yzw / max(uc.x, P.rho_floor);
  let vm = um.yzw / max(um.x, P.rho_floor);
  let vp = up.yzw / max(up.x, P.rho_floor);
  let sv = limit4(vec4<f32>(vc - vm, 0.0), vec4<f32>(vp - vc, 0.0)).xyz;
  return vc * srho + uc.x * sv;
}
fn slope_between(uc: Cons, m: Cons, p: Cons) -> Cons {
  let a = limit4(uc.a - m.a, p.a - uc.a);
  return Cons(vec4<f32>(a.x, mom_slope(uc.a, m.a, p.a, a.x)), limit4(uc.b - m.b, p.b - uc.b));
}

// Conservative, limited linear interpolation from a coarse cell to one of the fine cells it
// covers. `par` is the fine cell's parity (0 or 1) along each axis.
//
// Each axis's slope is limited on its own, which keeps the fine values between the coarse
// neighbors along that axis. In a corner of a steep gradient the three add up, and the fine
// cell furthest down the gradient can undershoot to nothing or below: a density that is then
// floored while its momentum is not, which makes a ghost cell moving at millions. So all the
// slopes are scaled back together, by the one factor that keeps the density and both energies
// of every fine cell at least half the coarse cell's. The factor depends on the coarse data
// only, so the fine cells still average to the coarse one.
fn prolong(uc: Cons, xm: MaybeCons, xp: MaybeCons, ym: MaybeCons, yp: MaybeCons, zm: MaybeCons, zp: MaybeCons, par: vec3<i32>) -> Cons {
  var sx = Cons(vec4<f32>(0.0), vec4<f32>(0.0));
  var sy = sx;
  var sz = sx;
  if (xm.ok && xp.ok) { sx = slope_between(uc, xm.u, xp.u); }
  if (ym.ok && yp.ok) { sy = slope_between(uc, ym.u, yp.u); }
  if (zm.ok && zp.ok && !is2d()) { sz = slope_between(uc, zm.u, zp.u); }
  // Largest drop below the coarse value among the fine cells: (density, total energy, internal energy).
  let drop = 0.25 * (abs(vec3<f32>(sx.a.x, sx.b.x, sx.b.y)) + abs(vec3<f32>(sy.a.x, sy.b.x, sy.b.y)) + abs(vec3<f32>(sz.a.x, sz.b.x, sz.b.y)));
  let room = 0.5 * max(vec3<f32>(uc.a.x, uc.b.x, uc.b.y), vec3<f32>(0.0));
  let fit = min(vec3<f32>(1.0), room / max(drop, vec3<f32>(1.0e-37)));
  let f = min(fit.x, min(fit.y, fit.z));
  let off = f * (vec3<f32>(par) - 0.5) * 0.5;   // -1/4 or +1/4 of a coarse cell, scaled back
  return Cons(
    uc.a + off.x * sx.a + off.y * sy.a + off.z * sz.a,
    uc.b + off.x * sx.b + off.y * sy.b + off.z * sz.b,
  );
}

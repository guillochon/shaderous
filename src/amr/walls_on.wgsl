// Forces on bodies on the block grid (hydro/walls_on.wgsl has the uniform grid's, and what WP
// and WV hold): each leaf cell's wall faces, from the fluxes it computed itself. A body's surface
// is always at the finest level (AmrSim.refineOutline), so a wall face is never a refinement
// face, and the sum over leaves counts each wall face once. Half way through a block's step
// (deposit, per-block timesteps) nothing is written: the block keeps the force of its own step.
// A cell is solid as the fluxes see it (its primitives' negative density, ghosts included); its
// body comes from its mark, read where the cell is stored (its own block, or the neighbor across
// a block face; 1, the first body, where neither holds it).
// Requires P, U2, load_q, the block table, PSPLIT and VISCOSITY.
@group(0) @binding(56) var WP: texture_storage_3d<rgba32float, write>;
@group(0) @binding(57) var WV: texture_storage_3d<rgba32float, write>;

fn wall_mark(slot: i32, l: vec3<i32>, c: vec3<i32>, a: i32, up: bool) -> f32 {
  var e = vec3<i32>(0);
  e[a] = select(-1, 1, up);
  if (load_q(c + e).b.w < 0.5) { return 0.0; }
  var q = l + e;
  if (q[a] >= 0 && q[a] < BS) { return max(textureLoad(U2, atl(blk_origin(slot), q), 0).w, 1.0); }
  let nb = bt(slot, 6u + u32(2 * a) + u32(up));
  if (nb < 0 || !blk_leaf(nb)) { return 1.0; }
  q[a] += select(BS, -BS, up);
  return max(textureLoad(U2, atl(blk_origin(nb), q), 0).w, 1.0);
}

struct WallSum { p: vec3<f32>, v: vec3<f32>, body: f32 }

fn wall_add(s: WallSum, f: Flux, m: f32, sgn: f32) -> WallSum {
  if (m < 0.5) { return s; }
  return WallSum(s.p + sgn * (f.mom - f.d.xyz), s.v + sgn * f.d.xyz, max(s.body, m));
}

fn store_wall_forces(slot: i32, l: vec3<i32>, c: vec3<i32>, fx: Faces, fy: Faces, fz: Faces) {
  var s = WallSum(vec3<f32>(0.0), vec3<f32>(0.0), 0.0);
  if (has_solids()) {
    s = wall_add(s, fx.hi, wall_mark(slot, l, c, 0, true), 1.0);
    s = wall_add(s, fx.lo, wall_mark(slot, l, c, 0, false), -1.0);
    s = wall_add(s, fy.hi, wall_mark(slot, l, c, 1, true), 1.0);
    s = wall_add(s, fy.lo, wall_mark(slot, l, c, 1, false), -1.0);
    if (!is2d()) {
      s = wall_add(s, fz.hi, wall_mark(slot, l, c, 2, true), 1.0);
      s = wall_add(s, fz.lo, wall_mark(slot, l, c, 2, false), -1.0);
    }
  }
  let area = select(g_dx * g_dx, g_dx, is2d());
  if (PSPLIT) {
    textureStore(WV, c, vec4<f32>((s.p + s.v) * area, s.body));
    return;
  }
  textureStore(WP, c, vec4<f32>(s.p * area, s.body));
  if (VISCOSITY) { textureStore(WV, c, vec4<f32>(s.v * area, s.body)); }
}

fn store_wall_none(c: vec3<i32>) {
  if (!PSPLIT) { textureStore(WP, c, vec4<f32>(0.0)); }
  if (VISCOSITY || PSPLIT) { textureStore(WV, c, vec4<f32>(0.0)); }
}

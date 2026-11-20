// Forces on bodies with implicit pressure on the block grid (pressure/pwalls_on.wgsl has the
// uniform grid's): the correction's wall-face pressures of each leaf cell, times the face's
// area, with the body's mark. A body's surface is at the finest level, so no wall face is a
// refinement face. Requires the definitions of amr/pressure/common.wgsl.
@group(0) @binding(56) var WP: texture_storage_3d<rgba32float, write>;

var<private> g_pw: vec4<f32>;

fn apwall_face(slot: i32, l: vec3<i32>, a: i32, up: bool, p: f32) {
  if (!has_solids()) { return; }
  let n = ap_nb(slot, l, a, up);
  if (n.kind != 2) { return; }
  // (A wall of the box comes back as this cell itself, which is gas.)
  let m = ap_mark(n.slot, n.l);
  if (m < 0.5) { return; }
  let h = blk_dx(slot);
  g_pw[a] += select(-1.0, 1.0, up) * p * select(h * h, h, is2d());
  g_pw.w = max(g_pw.w, m);
}

fn apwall_store(c: vec3<i32>) {
  textureStore(WP, c, g_pw);
}

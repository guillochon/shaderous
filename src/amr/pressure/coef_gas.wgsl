// The block solver's coefficients for a gas with an energy (amr/pressure/stencil.wgsl; the
// uniform grid's are pressure/source_gas.wgsl): alpha = 1 / (rho c^2 dt^2) and, on a face,
// beta = 1 / rho_f with rho_f the mean of the two cells' densities. An open face carries no
// pressure gradient.
//
// A leaf's are the advected state's primitives (W1, W2: amr/assemble.wgsl), ghosts included:
// the far cell of a face is the block's own cell or ghost, which is what the source and the
// correction use (amr/pressure/faces_gas.wgsl). A parent's are the means of its children's
// (KR, KA: amr/pressure/coef_restrict.wgsl), and the far cell of its face is read where it
// is. (The primitives of a parent would not do: where its children are refined themselves
// they are a placeholder, amr/cons.wgsl u_at, and the levels below would solve another
// equation.)
@group(0) @binding(15) var W1: texture_3d<f32>;
@group(0) @binding(22) var W2: texture_3d<f32>;
@group(0) @binding(7) var KR: texture_3d<f32>;
@group(0) @binding(8) var KA: texture_3d<f32>;
// The leaves' switches (amr/pressure/source_gas.wgsl): their alpha is divided by theta^2
// (amr/pressure/theta.wgsl), which the parents' means then take too.
@group(0) @binding(35) var CHI: texture_3d<f32>;
#include "amr/pressure/theta.wgsl"

// A leaf cell's 1 / (rho c^2 theta^2), from its advected primitives and switch.
fn ap_ka_leaf(slot: i32, l: vec3<i32>) -> f32 {
  let c = atl(blk_origin(slot), l);
  let rho = abs(textureLoad(W1, c, 0).x);
  let cs = textureLoad(W2, c, 0).y;
  let th = ap_theta(textureLoad(CHI, c, 0).x, cs, blk_dx(slot));
  return 1.0 / (rho * cs * cs * th * th);
}

const AP_OPEN = 0.0;

// (rho, 1 / (rho c^2 theta^2)) of a block's cell. (Every reader goes through here, so that each pass
// that uses a coefficient binds all four textures.)
fn ap_k(slot: i32, l: vec3<i32>) -> vec2<f32> {
  let c = atl(blk_origin(slot), l);
  if (blk_leaf(slot)) { return vec2<f32>(abs(textureLoad(W1, c, 0).x), ap_ka_leaf(slot, l)); }
  return vec2<f32>(textureLoad(KR, c, 0).x, textureLoad(KA, c, 0).x);
}
fn ap_beta(slot: i32, l: vec3<i32>, a: i32, up: bool) -> f32 {
  let own = ap_k(slot, l).x;
  if (blk_leaf(slot)) {
    var e = vec3<i32>(0);
    e[a] = select(-1, 1, up);
    // Toward a coarser cell: the ghost's density as the face values take it
    // (amr/pressure/faces_gas.wgsl g_ghost_rho).
    if (l[a] == select(0, BS - 1, up) && bt(slot, 6u + u32(2 * a) + u32(up)) == -1) {
      let n = ap_nb(slot, l, a, up);
      if (n.kind == 1) {
        let rc = ap_k(n.slot, n.l).x;
        return 2.0 / (own + max((4.0 * rc - own) / 3.0, 0.25 * min(own, rc)));
      }
    }
    return 2.0 / (own + abs(textureLoad(W1, atl(blk_origin(slot), l + e), 0).x));
  }
  let n = ap_nb(slot, l, a, up);
  var far = own;
  if (n.kind <= 1) { far = ap_k(n.slot, n.l).x; }
  return 2.0 / (own + far);
}
fn ap_alpha_at(slot: i32, l: vec3<i32>) -> f32 {
  return ap_k(slot, l).y / (clock.dt * clock.dt);
}

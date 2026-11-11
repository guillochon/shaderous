// Ghost data from a block that is part way through a longer step (per-block timesteps).
//
// Such a block's stored state is already that of the end of its step. Its neighbour one bin
// faster starts a step exactly at its midpoint, and what the block has for that time is its
// half-step prediction, kept from the start of its step: the scheme's own estimate of the
// state there, second-order where the flow is smooth. The prediction covers the block's cells
// and its first ghost layer, which is what interpolating to a finer neighbour needs.
// Requires amr/cons.wgsl, the equation of state and <bins>.
@group(0) @binding(7) var MPA: texture_3d<f32>;   // (rho, v)
@group(0) @binding(8) var MPB: texture_3d<f32>;   // (p, scalar, flags)

fn mid_block(slot: i32) -> bool {
  return blk_leaf(slot) && !blk_active(slot);
}

// Conserved state of cell l (interior or face ghost) of block `slot` at the middle of its step.
fn mid_cons(slot: i32, l: vec3<i32>) -> Cons {
  let c = atl(blk_origin(slot), l);
  let a = textureLoad(MPA, c, 0);
  let b = textureLoad(MPB, c, 0);
  let inside = all(l >= vec3<i32>(0)) && all(l < vec3<i32>(BS, BS, select(BS, 1, is2d())));
  if (inside) {
    // Solid and fixed cells do not change, and keep their mark.
    let u = u_leaf(slot, l);
    if (abs(u.b.w) > 0.5) { return u; }
  }
  let rho = a.x;
  let eint = eos_eint_from_p(rho, b.x);
  return Cons(vec4<f32>(rho, rho * a.yzw), vec4<f32>(eint + 0.5 * rho * dot(a.yzw, a.yzw), eint, rho * b.y, 0.0));
}

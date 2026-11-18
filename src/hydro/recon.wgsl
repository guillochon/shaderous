// Uniform grid: the primitives of a cell, converted from the conserved state as it is loaded,
// through the boundary mapping. There are no primitive textures: a pass over them costs far
// more in memory traffic than the conversions do in arithmetic (the flux update is bound by
// texture reads, not by its Riemann solves). Requires P, the equation of state,
// hydro/pfloor.wgsl, hydro/prim.wgsl, hydro/region.wgsl, the limiter, g_dx set, and bindings
// U1, U2.
// (Block grids define load_q and load_comp themselves: their ghost cells are stored.)
#include "hydro/q.wgsl"

// Composition moments of a cell, or of the cell a ghost maps to (species/mixture.ts).
fn load_comp(c: vec3<i32>) -> vec4<f32> {
  if (!COMP) { return vec4<f32>(0.0); }
  if (INTERIOR) { return comp_tex(c); }
  return comp_tex(bc_map(c).c);
}

fn load_q(c: vec3<i32>) -> Q {
  if (INTERIOR) {
    comp_use(comp_tex(c));
    let u2 = textureLoad(U2, c, 0);
    let w = prim_from_cons(textureLoad(U1, c, 0), u2);
    return Q(w.w1, vec4<f32>(w.w2.x, w.w2.z, w.w2.y, u2.w));
  }
  let m = bc_map(c);
  comp_use(comp_tex(m.c));
  let u2 = textureLoad(U2, m.c, 0);
  let w = prim_from_cons(textureLoad(U1, m.c, 0), u2);
  let q = Q(vec4<f32>(w.w1.x, bc_velocity(w.w1.yzw, m)), vec4<f32>(w.w2.x, w.w2.z, w.w2.y, u2.w));
  if (!shear_ghost(c)) { return q; }
  // Beyond a shearing box's x face: between two sheared cells, linearly.
  let c1 = shear_next(m.c);
  let v2 = textureLoad(U2, c1, 0);
  let w1 = prim_from_cons(textureLoad(U1, c1, 0), v2);
  let q1 = Q(vec4<f32>(w1.w1.x, bc_velocity(w1.w1.yzw, m)), vec4<f32>(w1.w2.x, w1.w2.z, w1.w2.y, v2.w));
  let f = shear_frac(c);
  return Q(mix(q.a, q1.a, f), mix(q.b, q1.b, f));
}

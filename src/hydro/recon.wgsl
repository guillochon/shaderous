// Uniform grid: the primitives of a cell, converted from the conserved state as it is loaded,
// through the boundary mapping. There are no primitive textures: a pass over them costs far
// more in memory traffic than the conversions do in arithmetic (the flux update is bound by
// texture reads, not by its Riemann solves). Requires P, the equation of state,
// hydro/pfloor.wgsl, hydro/prim.wgsl, hydro/region.wgsl, the limiter, g_dx set, and bindings
// U1, U2.
// (Block grids define load_q themselves: their ghost cells are stored.)
#include "hydro/q.wgsl"

fn load_q(c: vec3<i32>) -> Q {
  if (INTERIOR) {
    let w = prim_from_cons(textureLoad(U1, c, 0), textureLoad(U2, c, 0));
    return Q(w.w1, vec4<f32>(w.w2.x, w.w2.z, w.w2.y, 0.0));
  }
  let m = bc_map(c);
  let w = prim_from_cons(textureLoad(U1, m.c, 0), textureLoad(U2, m.c, 0));
  return Q(vec4<f32>(w.w1.x, bc_velocity(w.w1.yzw, m)), vec4<f32>(w.w2.x, w.w2.z, w.w2.y, 0.0));
}

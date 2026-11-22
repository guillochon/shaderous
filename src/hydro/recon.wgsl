// Uniform grid: the primitives of a cell, through the boundary mapping. By default they are
// converted from the conserved state as each cell is loaded, about twenty times a cell in a 3D
// step: on a GPU with little memory bandwidth to spare a pass of primitive textures cost more
// in memory traffic than the conversions do in arithmetic. With stored primitives (primStored
// in src/sim.ts, hydro/prim_load_on.wgsl) the timestep-rates pass converts each cell once and
// load_q reads the result; the values are the same. Requires P, the equation of state,
// hydro/pfloor.wgsl, hydro/prim.wgsl, hydro/region.wgsl, the limiter, g_dx set, and bindings
// U1, U2.
// (Block grids define load_q and load_comp themselves: their ghost cells are stored.)
#include "hydro/q.wgsl"
#include <prim_load>

// Composition moments of a cell, or of the cell a ghost maps to (species/mixture.ts).
fn load_comp(c: vec3<i32>) -> vec4<f32> {
  if (!COMP) { return vec4<f32>(0.0); }
  if (INTERIOR) { return comp_tex(c); }
  return comp_tex(bc_map(c).c);
}

// The primitives of the cell at c itself, as (rho, v) and (p, gamma term, sound speed, solid
// flag): stored by the timestep-rates pass, or converted here with the composition that
// comp_use last set.
fn cell_q(c: vec3<i32>) -> Q {
  if (PRIM_STORED) { return Q(prim_a(c), prim_b(c)); }
  let u2 = textureLoad(U2, c, 0);
  let w = prim_from_cons(textureLoad(U1, c, 0), u2);
  return Q(w.w1, vec4<f32>(w.w2.x, w.w2.z, w.w2.y, u2.w));
}

// (comp_use is called whether the primitives are stored or not: the equation of state reads the
// composition it sets afterwards, at the faces.)
fn load_q(c: vec3<i32>) -> Q {
  if (INTERIOR) {
    comp_use(comp_tex(c));
    return cell_q(c);
  }
  let m = bc_map(c);
  comp_use(comp_tex(m.c));
  let w = cell_q(m.c);
  let q = Q(vec4<f32>(w.a.x, bc_velocity(w.a.yzw, m)), w.b);
  if (!shear_ghost(c)) { return q; }
  // Beyond a shearing box's x face: between two sheared cells, linearly.
  let w1 = cell_q(shear_next(m.c));
  let q1 = Q(vec4<f32>(w1.a.x, bc_velocity(w1.a.yzw, m)), w1.b);
  let f = shear_frac(c);
  return Q(mix(q.a, q1.a, f), mix(q.b, q1.b, f));
}

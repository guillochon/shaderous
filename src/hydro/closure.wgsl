// The turbulence closure's pass on a uniform 2D grid (closure/closure.ts): each cell's
// coefficients from the state at the start of the step, written to a texture that the flux
// update reads on both sides of every face (hydro/closure_on.wgsl). One pass over the grid
// before the predictor: five cells loaded and one small network run per cell.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var CL: texture_storage_3d<rgba32float, write>;

#include "hydro/closure_net.wgsl"

struct ClLoad { q: ClCell, cs: f32, solid: bool }

fn cl_load(c: vec3<i32>) -> ClLoad {
  let m = bc_map(c);
  comp_use(comp_tex(m.c));
  let u2 = textureLoad(U2, m.c, 0);
  let w = prim_from_cons(textureLoad(U1, m.c, 0), u2);
  return ClLoad(ClCell(w.w1.x, bc_velocity(w.w1.yzw, m).xy, w.w2.x, w.w2.z), w.w2.y, u2.w > 0.5);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  g_dx = P.dx;
  let c = vec3<i32>(gid);
  let q = cl_load(c);
  let xm = cl_load(c + vec3<i32>(-1, 0, 0));
  let xp = cl_load(c + vec3<i32>(1, 0, 0));
  let ym = cl_load(c + vec3<i32>(0, -1, 0));
  let yp = cl_load(c + vec3<i32>(0, 1, 0));
  var out = vec4<f32>(0.0);
  // Beside an obstacle the neighbors are no gas to take a gradient across: no closure.
  if (!(q.solid || xm.solid || xp.solid || ym.solid || yp.solid)) {
    out = cl_cell(q.q, xm.q, xp.q, ym.q, yp.q, q.cs);
  }
  textureStore(CL, c, out);
}

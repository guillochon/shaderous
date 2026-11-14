// Edge electric fields for constrained transport, from the face fluxes of mhd/flux.wgsl
// (Gardiner & Stone 2005, the "E_z^contact" construction, in the form Athena++ uses): the mean
// of the four face-centered fields around the edge, plus their differences from the
// cell-centered reference field -v x B, upwinded by the direction of the mass flux.
//
// By the duality between the induction equation's fluxes and the electric field, the face
// normal to a (frame t1 = a + 1, t2 = a + 2) carries E_t2 = -F_a(B_t1) and E_t1 = F_a(B_t2).
//
// One invocation per texel t in [0, n] along each axis (along z, the slab's layers and the
// plane above it): E_x at (i, j - 1/2, k - 1/2), E_y at (i - 1/2, j, k - 1/2), E_z at
// (i - 1/2, j - 1/2, k). In 2D, E_x and E_y lie on the y and x faces, and are those faces'
// values. What is stored is the stage's impulse (dt/dx) E: the very number that every face
// around the edge adds or subtracts (mhd/update.wgsl), so that their sum over a cell's faces
// cancels exactly.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> FX: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> FY: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> FZ: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> EMF: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(14) var<uniform> SLAB: Slab;

override STAGE: u32 = 1u;

#include "mhd/common.wgsl"
#include "mhd/flux_index.wgsl"
#include "mhd/corner.wgsl"

// (F_a(B_t1), F_a(B_t2), upwind weight) of the lower face along a of cell t.
fn fc(a: i32, t: vec3<i32>) -> vec3<f32> {
  let i = 3u * flux_index(t) + 2u;
  if (a == 0) { return FX[i].xyz; }
  if (a == 1) { return FY[i].xyz; }
  return FZ[i].xyz;
}

fn eref(c: vec3<i32>) -> Ref {
  let q = mq_load(c);
  return Ref(efield(q.a.yzw, q.B), q.a.yzw, q.b.z);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ed = emf_dims();
  if (any(vec3<i32>(gid) >= ed)) { return; }
  g_dx = P.dx;
  let t = vec3<i32>(gid) + vec3<i32>(0, 0, emf_first());
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);
  let r000 = eref(t - ex - ey);
  let r100 = eref(t - ey);
  let r010 = eref(t - ex);
  let r111 = eref(t);
  // E_z: a = x, b = y.
  let Ez = corner(2, t, r000, r100, r010, r111);
  var E: vec3<f32>;
  if (P.dims.z == 1u) {
    // 2D: E_x on the y faces (-F_y(B_z)), E_y on the x faces (F_x(B_z)).
    E = vec3<f32>(-fc(1, t).x, fc(0, t).y, Ez);
  } else {
    let rz0 = eref(t - ez);
    // E_x: a = y, b = z, cells t - ey - ez, t - ez, t - ey, t.
    let Ex = corner(0, t, eref(t - ey - ez), rz0, r100, r111);
    // E_y: a = z, b = x, cells t - ez - ex, t - ex, t - ez, t.
    let Ey = corner(1, t, eref(t - ez - ex), r010, rz0, r111);
    E = vec3<f32>(Ex, Ey, Ez);
  }
  let dt = select(clock.dt, 0.5 * clock.dt, STAGE == 1u);
  EMF[emf_index(t)] = vec4<f32>((dt / P.dx) * E, 0.0);
}

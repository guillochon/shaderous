// Edge electric fields for constrained transport, from the face fluxes of mhd/flux.wgsl
// (Gardiner & Stone 2005, the "E_z^contact" construction, in the form Athena++ uses): the mean
// of the four face-centered fields around the edge, plus their differences from the
// cell-centered reference field -v x B, upwinded by the direction of the mass flux.
//
// By the duality between the induction equation's fluxes and the electric field, the face
// normal to a (frame t1 = a + 1, t2 = a + 2) carries E_t2 = -F_a(B_t1) and E_t1 = F_a(B_t2).
//
// One invocation per texel t in [0, n] along each axis: E_x at (i, j - 1/2, k - 1/2), E_y at
// (i - 1/2, j, k - 1/2), E_z at (i - 1/2, j - 1/2, k). In 2D, E_x and E_y lie on the y and x
// faces, and are those faces' values.
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

#include "mhd/common.wgsl"
#include "mhd/flux_index.wgsl"

// (F_a(B_t1), F_a(B_t2), upwind weight) of the lower face along a of cell t.
fn fc(a: i32, t: vec3<i32>) -> vec3<f32> {
  let i = 3u * flux_index(t) + 2u;
  if (a == 0) { return FX[i].xyz; }
  if (a == 1) { return FY[i].xyz; }
  return FZ[i].xyz;
}

// Reference field at a cell center.
fn eref(c: vec3<i32>) -> vec3<f32> {
  let q = mq_load(c);
  return efield(q.a.yzw, q.B);
}

// E along axis c at the edge on the lower a and b faces of cell t (a = c + 1, b = c + 2),
// given the reference fields of the four cells around it: e00 at t - ea - eb, e10 at t - eb,
// e01 at t - ea, e11 at t.
fn corner(c: i32, t: vec3<i32>, e00: f32, e10: f32, e01: f32, e11: f32) -> f32 {
  let a = (c + 1) % 3;
  let b = (c + 2) % 3;
  var ea = vec3<i32>(0);
  ea[a] = 1;
  var eb = vec3<i32>(0);
  eb[b] = 1;
  // Faces normal to a, at b positions j (t) and j - 1 (t - eb): E_c = -F_a(B_b).
  let fa1 = fc(a, t);
  let fa0 = fc(a, t - eb);
  // Faces normal to b, at a positions i (t) and i - 1 (t - ea): E_c = F_b(B_a).
  let fb1 = fc(b, t);
  let fb0 = fc(b, t - ea);
  let ea1 = -fa1.x;
  let ea0 = -fa0.x;
  let eb1 = fb1.y;
  let eb0 = fb0.y;
  // Differences toward the edge along b, upwinded across a by the a faces' mass flux.
  let dl2 = (1.0 - fa0.z) * (eb1 - e10) + fa0.z * (eb0 - e00);
  let dr2 = (1.0 - fa1.z) * (eb1 - e11) + fa1.z * (eb0 - e01);
  // Along a, upwinded across b.
  let dl1 = (1.0 - fb0.z) * (ea1 - e01) + fb0.z * (ea0 - e00);
  let dr1 = (1.0 - fb1.z) * (ea1 - e11) + fb1.z * (ea0 - e10);
  return 0.25 * (dl1 + dr1 + dl2 + dr2 + eb0 + eb1 + ea0 + ea1);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ed = emf_dims();
  if (any(vec3<i32>(gid) >= ed)) { return; }
  g_dx = P.dx;
  let t = vec3<i32>(gid);
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);
  let e000 = eref(t - ex - ey);
  let e100 = eref(t - ey);
  let e010 = eref(t - ex);
  let e111 = eref(t);
  // E_z: a = x, b = y.
  let Ez = corner(2, t, e000.z, e100.z, e010.z, e111.z);
  var E: vec3<f32>;
  if (P.dims.z == 1u) {
    // 2D: E_x on the y faces (-F_y(B_z)), E_y on the x faces (F_x(B_z)).
    E = vec3<f32>(-fc(1, t).x, fc(0, t).y, Ez);
  } else {
    let ez0 = eref(t - ez);
    // E_x: a = y, b = z, cells t - ey - ez, t - ez, t - ey, t.
    let Ex = corner(0, t, eref(t - ey - ez).x, ez0.x, e100.x, e111.x);
    // E_y: a = z, b = x, cells t - ez - ex, t - ex, t - ez, t.
    let Ey = corner(1, t, eref(t - ez - ex).y, e010.y, ez0.y, e111.y);
    E = vec3<f32>(Ex, Ey, Ez);
  }
  EMF[emf_index(t)] = vec4<f32>(E, 0.0);
}

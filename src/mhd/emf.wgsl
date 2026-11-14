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

// (F_a(B_t1), F_a(B_t2), upwind weight) of the lower face along a of cell t.
fn fc(a: i32, t: vec3<i32>) -> vec3<f32> {
  let i = 3u * flux_index(t) + 2u;
  if (a == 0) { return FX[i].xyz; }
  if (a == 1) { return FY[i].xyz; }
  return FZ[i].xyz;
}

// What the edges need of a cell: its reference field -v x B, its velocity and sound speed.
struct Ref { e: vec3<f32>, v: vec3<f32>, cs: f32 }

fn eref(c: vec3<i32>) -> Ref {
  let q = mq_load(c);
  return Ref(efield(q.a.yzw, q.B), q.a.yzw, q.b.z);
}

// Edge fields: 0 the mean of the four extrapolations (Gardiner & Stone 2005, as Athena++), 1
// only the upwind ones (Lee 2013's upwind-MEC averaging).
override EDGE_UPWIND: bool = false;

// E along axis c at the edge on the lower a and b faces of cell t (a = c + 1, b = c + 2),
// given the four cells around it: r00 at t - ea - eb, r10 at t - eb, r01 at t - ea, r11 at t.
//
// Each of the four faces around the edge gives an estimate of the edge's field: its own value
// carried half a cell along the face to the edge, with the gradient taken between the face and
// the reference field of the cell on that side, from whichever cell is upwind across the face
// (the mass flux's direction). Gardiner & Stone take the mean of the four. Lee (2013) points
// out that for flow mostly along one axis the faces downstream of the edge carry no information
// about it, and the mean then lets oscillations grow (the field loop at small angles); his
// upwind-MEC uses only the estimates from the faces upwind of the edge: for flow toward +b, the
// a face below the edge, toward +a, the b face to its left, both (weighted a half each) when
// the flow has both components. A component under a tenth of the in-plane speed counts as none,
// so noise does not flip the choice, and where the flow is slower than 1e-4 of the sound speed
// the mean is used. (Lee also adds second-derivative terms to each estimate, from Lee & Deane
// 2009; here the estimates are Gardiner & Stone's.)
fn corner(c: i32, t: vec3<i32>, r00: Ref, r10: Ref, r01: Ref, r11: Ref) -> f32 {
  let a = (c + 1) % 3;
  let b = (c + 2) % 3;
  var ea = vec3<i32>(0);
  ea[a] = 1;
  var eb = vec3<i32>(0);
  eb[b] = 1;
  let e00 = r00.e[c];
  let e10 = r10.e[c];
  let e01 = r01.e[c];
  let e11 = r11.e[c];
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
  // The four estimates: from the a face below the edge (along b), the a face above it, the b
  // face left of it (along a) and the b face right of it.
  let below = ea0 + ((1.0 - fa0.z) * (eb1 - e10) + fa0.z * (eb0 - e00));
  let above = ea1 + ((1.0 - fa1.z) * (eb1 - e11) + fa1.z * (eb0 - e01));
  let left = eb0 + ((1.0 - fb0.z) * (ea1 - e01) + fb0.z * (ea0 - e00));
  let right = eb1 + ((1.0 - fb1.z) * (ea1 - e11) + fb1.z * (ea0 - e10));
  let mean = 0.25 * ((left + right) + (below + above));
  if (!EDGE_UPWIND) { return mean; }
  let v = 0.25 * ((r00.v + r11.v) + (r10.v + r01.v));
  let cs = 0.25 * ((r00.cs + r11.cs) + (r10.cs + r01.cs));
  var u = v[a];
  var w = v[b];
  let speed = sqrt(u * u + w * w);
  if (!(speed > 1.0e-4 * cs)) { return mean; }
  if (abs(u) <= 0.1 * speed) { u = 0.0; }
  if (abs(w) <= 0.1 * speed) { w = 0.0; }
  let along_a = select(right, left, u > 0.0);
  let along_b = select(above, below, w > 0.0);
  if (u == 0.0) { return along_b; }
  if (w == 0.0) { return along_a; }
  return 0.5 * (along_a + along_b);
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

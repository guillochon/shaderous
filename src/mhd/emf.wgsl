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
#include <diffusion>

@group(0) @binding(2) var U1: texture_3d<f32>;
// Declared for mhd/common.wgsl's mq_load, which this pass does not call: not bound (mhd/mhd.ts).
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

// The face field the moving walls carry (mhd/corner.wgsl).
fn wall_bn(a: i32, t: vec3<i32>) -> f32 {
  return bface(a, t);
}

// (F_a(B_t1), F_a(B_t2), upwind weight) of the lower face along a of cell t.
fn fc(a: i32, t: vec3<i32>) -> vec3<f32> {
  let i = 3u * flux_index(t) + 2u;
  if (a == 0) { return FX[i].xyz; }
  if (a == 1) { return FY[i].xyz; }
  return FZ[i].xyz;
}

// The edges need only -v x B of the cells, not the equation of state that mq_load runs. The
// velocity is the same either way (hydro/prim.wgsl).
fn eref(c: vec3<i32>) -> Ref {
  let m = bc_map(c);
  let u = textureLoad(U1, m.c, 0);
  var v = bc_velocity(u.yzw / max(u.x, P.rho_floor), m);
  var B = bcell(m.c) * m.bflip;
  // A shearing box's ghost beyond an x face: between the two sheared cells.
  if (shear_ghost(c)) {
    let c1 = shear_next(m.c);
    let u1 = textureLoad(U1, c1, 0);
    let f = shear_frac(c);
    v = mix(v, bc_velocity(u1.yzw / max(u1.x, P.rho_floor), m), f);
    B = mix(B, bcell(c1) * m.bflip, f);
  }
  return Ref(efield(v, B), B);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ed = emf_dims();
  if (any(vec3<i32>(gid) >= ed)) { return; }
  g_dx = P.dx;
  if (shear_on()) { g_shear = select(clock.shear, shear_half(clock.shear, clock.dt), STAGE == 2u) / P.dx; }
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
  // Resistivity (mhd/ohm.wgsl): eta J, as dx J.
  var J = vec3<f32>(0.0);
  if (P.dims.z == 1u) {
    // 2D: E_x on the y faces (-F_y(B_z)), E_y on the x faces (F_x(B_z)).
    E = vec3<f32>(-fc(1, t).x, fc(0, t).y, Ez);
    if (RESISTIVITY) { J = vec3<f32>(r111.b.z - r100.b.z, r010.b.z - r111.b.z, edge_current(2, r000, r100, r010, r111)); }
  } else {
    let rz0 = eref(t - ez);
    let ryz = eref(t - ey - ez);
    let rzx = eref(t - ez - ex);
    // E_x: a = y, b = z, cells t - ey - ez, t - ez, t - ey, t.
    let Ex = corner(0, t, ryz, rz0, r100, r111);
    // E_y: a = z, b = x, cells t - ez - ex, t - ex, t - ez, t.
    let Ey = corner(1, t, rzx, r010, rz0, r111);
    E = vec3<f32>(Ex, Ey, Ez);
    if (RESISTIVITY) { J = vec3<f32>(edge_current(0, ryz, rz0, r100, r111), edge_current(1, rzx, r010, rz0, r111), edge_current(2, r000, r100, r010, r111)); }
  }
  if (RESISTIVITY) { E += (P.ohm.x / P.dx) * J; }
  // Line-tied and moving walls (hydro/boundary.wgsl): no field along a wall at rest, so the flux
  // through it stays; -v_wall x B_n along a moving one.
  let n = mhd_dims();
  var on = vec3<bool>(false);
  var kinds = vec3<u32>(0u);
  var upper = vec3<bool>(false);
  for (var a = 0; a < 3; a++) {
    if (a == 2 && P.dims.z == 1u) { continue; }
    if (t[a] == 0 && bc_tied(P.bc_lo[a])) {
      on[a] = true;
      kinds[a] = P.bc_lo[a];
    } else if (t[a] == n[a] && bc_tied(P.bc_hi[a])) {
      on[a] = true;
      kinds[a] = P.bc_hi[a];
      upper[a] = true;
    }
  }
  E = E * bc_tied_keep(on) + moving_wall_e(t, on, kinds, upper);
  // The axis of an axisymmetric run: E_phi (code z) vanishes on it, and no B_phi crosses it
  // (2D: E_z, code y, on the x faces is the flux of B_phi through them).
  if (cylindrical() && t.x == 0) { E = vec3<f32>(E.x, 0.0, 0.0); }
  let dt = select(clock.dt, 0.5 * clock.dt, STAGE == 1u);
  EMF[emf_index(t)] = vec4<f32>((dt / P.dx) * E, 0.0);
}

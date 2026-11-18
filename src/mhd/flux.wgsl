// MHD face fluxes on the uniform grid, for one stage of the van Leer integrator (Stone &
// Gardiner 2009; docs/numerics.md section 21). Each invocation computes the fluxes through the lower
// faces of one cell, so every face is computed once and the two cells it separates read the
// same bits.
//
//   STAGE 1: first order, from the cells' own states at the start of the step.
//   STAGE 2: limited linear reconstruction of the primitives from the half-step state, and the
//            shock flags stage 1 left in it.
//
// The faces stored are those of cells t in [-1, n] along each axis (along z, one layer beyond
// each side of the slab: mhd/flux_index.wgsl) whose index along the face's own normal is in
// [0, n] (in 2D, the single layer along z, and no z faces): the edge electric fields on the
// grid's boundary need the faces of one row of ghost cells beyond it.
// Ghost cells take their state through the boundary mapping, as everywhere.
//
// Viscosity and conduction (hydro/diffusion_core.wgsl) add their fluxes to every face, in both
// stages, from the stage's own state: the start's in stage 1, the half step's in stage 2, so
// that they too are integrated to second order in time. The MHD update adds the heat the
// viscosity leaves to the internal energy (mhd/update.wgsl).
//
// Flux buffer of axis a, three vec4 per face (mhd/flux_index.wgsl):
//   (mass, momentum)   (total energy, internal energy, scalar, face velocity)
//   (F_a(B_t1), F_a(B_t2), upwind weight, 0)
// with t1 = a + 1 and t2 = a + 2 (mod 3), and the upwind weight of Athena++ for the edge
// fields: 1 where the gas flows toward +a, 0 toward -a, smooth in between.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> FX: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> FY: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> FZ: array<vec4<f32>>;
@group(0) @binding(14) var<uniform> SLAB: Slab;
// The state and faces at the start of the step, for stage 2's first-order flux correction
// (mhd/face_flux.wgsl); stage 1 reads them as U1, U2, BF.
@group(0) @binding(9) var S1: texture_3d<f32>;
@group(0) @binding(10) var S2: texture_3d<f32>;
@group(0) @binding(11) var SF: texture_3d<f32>;
// The first-order flux correction after the update (mhd/fofc.wgsl): the cells this step's stage 2
// listed, whose count stage 2 resets.
@group(0) @binding(34) var<storage, read_write> FOFC: Fofc;

override STAGE: u32 = 1u;
// The fix's round (mhd/fofc.wgsl): round 1 takes every level of the list, round 0 level 0.
override FOFC_ALL_LEVELS: bool = false;

#include "mhd/common.wgsl"
#include "mhd/flux_index.wgsl"
#include "mhd/riemann/common.wgsl"
#include <mhd_riemann>
#include "mhd/face_flux.wgsl"
#include "mhd/fofc.wgsl"

// For the diffusive fluxes: a cell's velocity and composition, ghosts through the boundary
// mapping, and every cell has its transverse neighbors.
fn load_vel(c: vec3<i32>) -> vec3<f32> {
  let m = bc_map(c);
  let u = textureLoad(U1, m.c, 0);
  return bc_velocity(u.yzw / max(u.x, P.rho_floor), m);
}
fn load_comp(c: vec3<i32>) -> vec4<f32> {
  return comp_tex(bc_map(c).c);
}
fn face_rim(a: vec3<i32>, axis: i32, t: i32) -> i32 {
  return 0;
}
fn load_b(c: vec3<i32>) -> vec3<f32> {
  let m = bc_map(c);
  return bcell(m.c) * m.bflip;
}
#include "hydro/diffusion_core.wgsl"
#include "mhd/diffusion.wgsl"

// The state of a cell at the start of the step, as mq_load reads the stage's.
fn mq_start_at(m: Mapped, c: vec3<i32>) -> MQ {
  let w = prim_from_cons(textureLoad(S1, c, 0), textureLoad(S2, c, 0));
  let n = mhd_dims();
  let up = c + 1;
  let wrap = vec3<bool>(up.x >= n.x && periodic_axis(0), up.y >= n.y && P.bc_lo[1] == 0u, up.z >= n.z && P.bc_lo[2] == 0u);
  let hi = select(up, vec3<i32>(0), wrap);
  let lo = textureLoad(SF, c, 0);
  let bxhi = textureLoad(SF, vec3<i32>(hi.x, c.y, c.z), 0).x;
  let B = vec3<f32>(
    select(0.5 * (lo.x + bxhi), cyl_bcell_r(c.x, lo.x, bxhi), cylindrical()),
    0.5 * (lo.y + textureLoad(SF, vec3<i32>(c.x, hi.y, c.z), 0).y),
    0.5 * (lo.z + textureLoad(SF, vec3<i32>(c.x, c.y, hi.z), 0).z),
  );
  return MQ(vec4<f32>(w.w1.x, bc_velocity(w.w1.yzw, m)), vec4<f32>(w.w2.x, w.w2.z, w.w2.y, w.w2.w), B * m.bflip);
}
fn mq_start(c: vec3<i32>) -> MQ {
  let m = bc_map(c);
  comp_use(comp_tex(m.c));
  let q = mq_start_at(m, m.c);
  if (shear_ghost(c)) { return mq_mix(q, mq_start_at(m, shear_next(m.c)), shear_frac(c)); }
  return q;
}

// The flux through the lower face along `axis` of cell t.
fn face(t: vec3<i32>, axis: i32) -> FaceOut {
  var e = vec3<i32>(0);
  e[axis] = 1;
  if (STAGE == 2u) {
    // First-order flux correction (mhd/face_flux.wgsl).
    let ml = bc_map(t - e).c;
    let mr = bc_map(t).c;
    if (fofc_suspect(textureLoad(S1, ml, 0).x, textureLoad(U1, ml, 0).x) || fofc_suspect(textureLoad(S1, mr, 0).x, textureLoad(U1, mr, 0).x)) {
      return face_first_order(t, axis, ml, mr);
    }
  }
  return face_second_order(t, axis);
}

// The first-order HLLE flux of the state at the start of the step (stage 2; mhd/face_flux.wgsl).
fn face_first_order(t: vec3<i32>, axis: i32, ml: vec3<i32>, mr: vec3<i32>) -> FaceOut {
  var e = vec3<i32>(0);
  e[axis] = 1;
  let a = mq_start(t - e);
  let b = mq_start(t);
  let bn = textureLoad(SF, face_texel(axis, t), 0)[axis];
  return face_of(a, a, b, b, bn, axis, false, 1.0, comp_tex(ml), comp_tex(mr), clock.dt);
}

// The stage's own flux.
fn face_second_order(t: vec3<i32>, axis: i32) -> FaceOut {
  var e = vec3<i32>(0);
  e[axis] = 1;
  let ql = mq_load(t - e);
  let qr = mq_load(t);
  var qll = ql;
  var qrr = qr;
  var w = 0.0;
  if (STAGE == 2u) {
    qll = mq_load(t - 2 * e);
    qrr = mq_load(t + e);
    // The shock flags stage 1 left in the half-step state (U2.w).
    w = max(textureLoad(U2, bc_map(t - e).c, 0).w, textureLoad(U2, bc_map(t).c, 0).w);
  }
  let dt = select(clock.dt, 0.5 * clock.dt, STAGE == 1u);
  var o = face_of(qll, ql, qr, qrr, bface(axis, t), axis, STAGE == 2u, w, comp_tex(bc_map(t - e).c), comp_tex(bc_map(t).c), dt);
  let n = mhd_dims();
  g_tied_face = (t[axis] == 0 && bc_tied(P.bc_lo[axis])) || (t[axis] == n[axis] && bc_tied(P.bc_hi[axis]));
  o.f = mhd_add_diffusion(o.f, t - e, e, axis, ql, qr);
  return o;
}

// The third entry's last channel, free otherwise, carries what an axisymmetric run's radial
// source needs of the stage's field (mhd/update.wgsl): on an x face its normal field B_R, on a
// y entry the cell's own B_phi.
fn put(axis: i32, i: u32, o: FaceOut, extra: f32) {
  let a = vec4<f32>(o.f.m, o.f.mom);
  let b = vec4<f32>(o.f.e, o.f.ei, o.f.s, o.f.vn);
  let c = vec4<f32>(o.f.bt, o.w, extra);
  if (axis == 0) { FX[3u * i] = a; FX[3u * i + 1u] = b; FX[3u * i + 2u] = c; }
  if (axis == 1) { FY[3u * i] = a; FY[3u * i + 1u] = b; FY[3u * i + 2u] = c; }
  if (axis == 2) { FZ[3u * i] = a; FZ[3u * i + 1u] = b; FZ[3u * i + 2u] = c; }
}

// The fix of mhd/fofc.wgsl: face k = j % 6 of listed cell j / 6, its mass, momentum and energy
// entries rewritten with the first-order flux, wherever this slab's buffers hold it: a face on the
// grid's boundary of a periodic axis is held twice (indices 0 and n), and both are written. A
// shearing box's x boundary faces are left as the shear pass made them.
fn fix(j: u32) {
  let c = fofc_listed(j / 6u);
  let a = i32((j % 6u) / 2u);
  let up = (j % 2u) == 1u;
  let n = mhd_dims();
  if (a >= select(3, 2, P.dims.z == 1u)) { return; }
  if (shear_on() && a == 0 && ((!up && c.x == 0) || (up && c.x == n.x - 1))) { return; }
  var t = c;
  if (up) { t[a] += 1; }
  var e = vec3<i32>(0);
  e[a] = 1;
  let o = face_first_order(t, a, bc_map(t - e).c, bc_map(t).c);
  let fa = vec4<f32>(o.f.m, o.f.mom);
  let fb = vec4<f32>(o.f.e, o.f.ei, o.f.s, o.f.vn);
  let fd = flux_dims();
  let z0 = flux_first().z;
  let flat = P.dims.z == 1u;
  // Each copy: the face itself or its twin across its own periodic axis, and on a grid periodic
  // along z, either of those a period up or down, where the last slab's buffers hold the first
  // layers again beyond the top (and the first slab's the last ones below the bottom).
  for (var copy = 0; copy < 2; copy++) {
    var tc = t;
    if (copy == 1) {
      if (!periodic_axis(a)) { break; }
      if (t[a] == 0) { tc[a] = n[a]; } else if (t[a] == n[a]) { tc[a] = 0; } else { break; }
    }
    for (var shift = -1; shift <= 1; shift++) {
      if (shift != 0 && (flat || !periodic_axis(2))) { continue; }
      var tt = tc;
      tt.z += shift * n.z;
      if (!flat && (tt.z < z0 || tt.z >= z0 + fd.z)) { continue; }
      let i = 3u * flux_index(tt);
      if (a == 0) { FX[i] = fa; FX[i + 1u] = fb; }
      if (a == 1) { FY[i] = fa; FY[i + 1u] = fb; }
      if (a == 2) { FZ[i] = fa; FZ[i + 1u] = fb; }
    }
  }
}

// The fix of mhd/fofc.wgsl, after the stage-2 update of a slab: FOFC_GROUPS workgroups through the
// list's faces (an entry point of its own, so that the flux pass carries none of its code).
@compute @workgroup_size(256)
fn fix_main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  g_dx = P.dx;
  g_dt = clock.dt;
  if (shear_on()) { g_shear = shear_half(clock.shear, clock.dt) / P.dx; }
  let count = fofc_count();
  for (var j = wg.x * 256u + li; j < 6u * count; j += FOFC_THREADS) {
    if (fofc_in_round(j / 6u, FOFC_ALL_LEVELS)) { fix(j); }
  }
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  g_dx = P.dx;
  g_dt = clock.dt;
  let fd = flux_dims();
  if (any(vec3<i32>(gid) >= fd)) { return; }
  // A new step's list (mhd/fofc.wgsl), which then runs through all its slabs.
  if (STAGE == 2u && SLAB.z0 == 0 && all(gid == vec3<u32>(0u))) { atomicStore(&FOFC.count, 0u); }
  // A shearing box: the offset of the x faces for the stage's state (the start's, or the half
  // step's in stage 2).
  if (shear_on()) { g_shear = select(clock.shear, shear_half(clock.shear, clock.dt), STAGE == 2u) / P.dx; }
  let t = vec3<i32>(gid) + flux_first();
  let n = mhd_dims();
  let axes = select(3, 2, P.dims.z == 1u);
  let i = flux_index(t);
  for (var a = 0; a < axes; a++) {
    // On a grid periodic along z, the last slab also holds z face n + 1, the twin of face 1: the
    // redo of mhd/fofc.wgsl updates cell 0 there again, as layer n (mhd/update.wgsl redo).
    if (t[a] >= 0 && (t[a] <= n[a] || (a == 2 && t[a] == n[a] + 1 && periodic_axis(2)))) {
      var extra = 0.0;
      if (cylindrical()) {
        if (a == 0) { extra = bface(0, t); }
        if (a == 1 && all(t >= vec3<i32>(0)) && all(t < n)) { extra = bcell(t).z; }
      }
      put(a, i, face(t, a), extra);
    }
  }
}

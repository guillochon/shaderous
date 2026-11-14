// MHD face fluxes on the uniform grid, for one stage of the van Leer integrator (Stone &
// Gardiner 2009; docs/mhd-plan.md). Each invocation computes the fluxes through the lower
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

override STAGE: u32 = 1u;

#include "mhd/common.wgsl"
#include "mhd/flux_index.wgsl"
#include "mhd/riemann/common.wgsl"
#include <mhd_riemann>
#include "mhd/face_flux.wgsl"

// The flux through the lower face along `axis` of cell t.
fn face(t: vec3<i32>, axis: i32) -> FaceOut {
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
  return face_of(qll, ql, qr, qrr, bface(axis, t), axis, STAGE == 2u, w, comp_tex(bc_map(t - e).c), comp_tex(bc_map(t).c), dt);
}

fn put(axis: i32, i: u32, o: FaceOut) {
  let a = vec4<f32>(o.f.m, o.f.mom);
  let b = vec4<f32>(o.f.e, o.f.ei, o.f.s, o.f.vn);
  let c = vec4<f32>(o.f.bt, o.w, 0.0);
  if (axis == 0) { FX[3u * i] = a; FX[3u * i + 1u] = b; FX[3u * i + 2u] = c; }
  if (axis == 1) { FY[3u * i] = a; FY[3u * i + 1u] = b; FY[3u * i + 2u] = c; }
  if (axis == 2) { FZ[3u * i] = a; FZ[3u * i + 1u] = b; FZ[3u * i + 2u] = c; }
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fd = flux_dims();
  if (any(vec3<i32>(gid) >= fd)) { return; }
  g_dx = P.dx;
  g_dt = clock.dt;
  let t = vec3<i32>(gid) + flux_first();
  let n = mhd_dims();
  let axes = select(3, 2, P.dims.z == 1u);
  let i = flux_index(t);
  for (var a = 0; a < axes; a++) {
    if (t[a] >= 0 && t[a] <= n[a]) { put(a, i, face(t, a)); }
  }
}

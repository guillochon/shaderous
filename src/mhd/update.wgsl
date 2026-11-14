// MHD update on the uniform grid, for one stage of the van Leer integrator: the conservative
// update of each cell from the face fluxes of mhd/flux.wgsl, and the constrained-transport
// update of its lower faces from the edge electric fields of mhd/emf.wgsl,
//   b_a -= dt / dx [(E_c(t + e_b) - E_c(t)) - (E_b(t + e_c) - E_b(t))],  b = a + 1, c = a + 2,
// the discrete form of dB/dt = -curl E. Every face is updated from the same four edge values by
// whichever cell owns it, and each edge value enters the faces around it with opposite signs,
// so the divergence of every cell is kept to round-off.
//
//   STAGE 1: half a step from the state at the start of the step: the half-step state.
//   STAGE 2: a full step from the state at the start, with fluxes of the half-step state.
//
// One invocation per texel t in [0, n] along each axis (the single layer along z in 2D): the
// cell t if it is in the grid, and the faces texel t holds that exist. The internal energy's
// p div v takes the pressure at the start of the step in stage 1 and at the half step in
// stage 2.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"
#include "hydro/fluff.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;   // state at the start of the step
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> FX: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> FY: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> FZ: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> EMF: array<vec4<f32>>;
@group(0) @binding(9) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(10) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(11) var OB: texture_storage_3d<rgba32float, write>;
@group(0) @binding(12) var H1: texture_3d<f32>;   // half-step state (stage 2)
@group(0) @binding(13) var H2: texture_3d<f32>;

override STAGE: u32 = 1u;

#include "mhd/common.wgsl"
#include "mhd/flux_index.wgsl"

fn emf(t: vec3<i32>) -> vec3<f32> {
  return EMF[emf_index(t)].xyz;
}

// The face along a of texel t after the stage, from its value at the start of the step.
fn new_face(a: i32, t: vec3<i32>, dtdx: f32) -> f32 {
  let b = (a + 1) % 3;
  let c = (a + 2) % 3;
  var eb = vec3<i32>(0);
  eb[b] = 1;
  var ec = vec3<i32>(0);
  ec[c] = 1;
  let e0 = emf(t);
  let curl = (emf(t + eb)[c] - e0[c]) - (emf(t + ec)[b] - e0[b]);
  return bface(a, t) - dtdx * curl;
}

struct Fl { a: vec4<f32>, b: vec4<f32> }

fn flux_at(axis: i32, t: vec3<i32>) -> Fl {
  let i = 3u * flux_index(t);
  if (axis == 0) { return Fl(FX[i], FX[i + 1u]); }
  if (axis == 1) { return Fl(FY[i], FY[i + 1u]); }
  return Fl(FZ[i], FZ[i + 1u]);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ed = emf_dims();
  if (any(vec3<i32>(gid) >= ed)) { return; }
  g_dx = P.dx;
  let dt = select(clock.dt, 0.5 * clock.dt, STAGE == 1u);
  g_dt = dt;
  let dtdx = dt / P.dx;
  let t = vec3<i32>(gid);
  let n = mhd_dims();
  let axes = select(3, 2, P.dims.z == 1u);

  // Faces held by this texel: along a, those with an index along a up to n (up to n - 1 on a
  // periodic axis, whose upper face is the lower face of cell 0), and in the grid along the
  // other two.
  var bout = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    var ok = t[a] < n[a] || (t[a] == n[a] && !periodic_axis(a));
    for (var b = 0; b < 3; b++) {
      if (b != a && t[b] >= n[b]) { ok = false; }
    }
    if (ok) { bout[a] = new_face(a, t, dtdx); }
  }
  textureStore(OB, t, vec4<f32>(bout, 0.0));

  if (any(t >= n)) { return; }
  // The cell.
  var net_a = vec4<f32>(0.0);   // (mass, momentum)
  var net_b = vec4<f32>(0.0);   // (energy, internal energy, scalar, face velocity)
  var bc = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    if (a < axes) {
      let lo = flux_at(a, t);
      let hi = flux_at(a, t + e);
      net_a += lo.a - hi.a;
      net_b += lo.b - hi.b;
    }
    bc[a] = 0.5 * (bout[a] + new_face(a, t + e, dtdx));
  }
  let u1 = textureLoad(U1, t, 0);
  let u2 = textureLoad(U2, t, 0);
  comp_use(comp_tex(t));
  // Pressure for p div v.
  var ph: f32;
  if (STAGE == 1u) {
    ph = prim_from_cons(u1, u2).w2.x;
  } else {
    ph = prim_from_cons(textureLoad(H1, t, 0), textureLoad(H2, t, 0)).w2.x;
  }
  let rho = max(u1.x + dtdx * net_a.x, P.rho_floor);
  let mom = u1.yzw + dtdx * net_a.yzw;
  let E = u2.x + dtdx * net_b.x;
  let eint = u2.y + dtdx * (net_b.y + max(ph, P.p_floor) * net_b.w);
  let sc = u2.z + dtdx * net_b.z;
  let eb = 0.5 * dot(bc, bc);
  var en = dual_energy(rho, 0.5 * dot(mom, mom) / rho + eb, E, eint);
  var mout = mom;
  let tame = tame_fluff(rho, mom, en.y);
  if (tame.changed) {
    mout = tame.mom;
    en = vec2<f32>(0.5 * dot(mout, mout) / rho + eb + tame.eint, tame.eint);
  }
  textureStore(O1, t, vec4<f32>(rho, mout));
  textureStore(O2, t, vec4<f32>(en.x, en.y, sc, 0.0));
}

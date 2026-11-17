// MHD face fluxes on the block grid, for one stage (mhd/flux.wgsl does the same on the uniform
// grid). Each invocation takes the lower faces of one cell t of a leaf block, t in [-1, BS]
// along each axis: those whose index along their own normal is in [0, BS], and which are at
// most one cell outside the block across it (the edge fields beside the block's edges need
// the ghost rows next to it; nothing needs the ghost corners). The states come from the
// block's assembled storage, ghosts included (mhd/amr_assemble.wgsl).
//
// Writes the chunk's flux scratch, and the flux register for the faces on the block's faces.
//
// Stage 2's first-order flux correction (mhd/face_flux.wgsl) applies to every face computed
// here, the ghost rows' included: a face both blocks beside it compute must come out the same
// from both, and the edge fields of either take the ghost rows' faces. The suspect test and the
// first-order flux read what stage 1's assembly kept, ghosts included.
//
// Viscosity and conduction (hydro/diffusion_core.wgsl) are added to the faces the block's own
// cells use, from the stage's assembled state: its edge ghosts give every such face the
// transverse neighbors of both its cells. (The ghost rows' faces feed only the edge fields.)
// Being in the register, they are matched at refinement boundaries as the rest of the flux is.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;
@group(0) @binding(3) var W2: texture_3d<f32>;
@group(0) @binding(4) var W3: texture_3d<f32>;
@group(0) @binding(5) var FA: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> FL: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> REG: array<vec4<f32>>;
@group(0) @binding(14) var<uniform> CHUNK: Chunk;
// What stage 1's assembly kept for stage 2's first-order flux correction (mhd/amr_keep.wgsl):
// (rho, v), (p, scalar, b_x, b_y), (b_z, cell-centered field), the faces each texel's lower ones.
@group(0) @binding(8) var K0: texture_3d<f32>;
@group(0) @binding(9) var K1: texture_3d<f32>;
@group(0) @binding(10) var K2: texture_3d<f32>;

override STAGE: u32 = 1u;

#include "mhd/state.wgsl"
#include "mhd/amr_block.wgsl"
#include "mhd/riemann/common.wgsl"
#include <mhd_riemann>
#include "mhd/face_flux.wgsl"

// Stage 1's lower face along a at texel c, as kept.
fn kept_face(c: vec3<i32>, a: i32) -> f32 {
  if (a == 2) { return textureLoad(K2, c, 0).x; }
  return textureLoad(K1, c, 0)[2 + a];
}
// A texel of stage 1's assembly, as mq_block reads the stage's. The sound speed and internal
// energy, which the first-order flux does not read, are left out.
fn mq_start(c: vec3<i32>) -> MQ {
  let k1 = textureLoad(K1, c, 0);
  return MQ(textureLoad(K0, c, 0), vec4<f32>(k1.x, k1.y, 0.0, 0.0), textureLoad(K2, c, 0).yzw);
}

// For the diffusive fluxes, at atlas texel c: the assembled velocity. One composition (MHD on
// blocks has no mixtures), and every face that takes them has its transverse neighbors.
fn load_vel(c: vec3<i32>) -> vec3<f32> {
  return textureLoad(W1, c, 0).yzw;
}
fn load_comp(c: vec3<i32>) -> vec4<f32> {
  return vec4<f32>(0.0);
}
fn face_rim(a: vec3<i32>, axis: i32, t: i32) -> i32 {
  return 0;
}
fn load_b(c: vec3<i32>) -> vec3<f32> {
  return textureLoad(W3, c, 0).xyz;
}
#include "hydro/diffusion_core.wgsl"
#include "mhd/diffusion.wgsl"

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let inv = inv_box(wid, lix, -1, FBOX);
  if (!inv.ok) { return; }
  let slot = inv.slot;
  let t = inv.l;
  g_dx = blk_dx(slot);
  let o = blk_origin(slot);
  let dt = stage_dt(slot, STAGE);
  let outside = vec3<i32>(t < vec3<i32>(0)) + vec3<i32>(t >= vec3<i32>(BS));
  for (var a = 0; a < naxes(); a++) {
    if (t[a] < 0 || t[a] > BS) { continue; }
    var across = outside;
    across[a] = 0;
    if (is2d()) { across.z = 0; }
    if (across.x + across.y + across.z > 1) { continue; }
    var e = vec3<i32>(0);
    e[a] = 1;
    let ql = mq_block(o, t - e);
    let qr = mq_block(o, t);
    var qll = ql;
    var qrr = qr;
    var w = 0.0;
    if (STAGE == 2u) {
      qll = mq_block(o, t - 2 * e);
      qrr = mq_block(o, t + e);
      w = max(textureLoad(W3, atl(o, t - e), 0).w, textureLoad(W3, atl(o, t), 0).w);
    }
    var F: FaceOut;
    let cl = atl(o, t - e);
    let cr = atl(o, t);
    if (STAGE == 2u && (fofc_suspect(textureLoad(K0, cl, 0).x, ql.a.x) || fofc_suspect(textureLoad(K0, cr, 0).x, qr.a.x))) {
      // First-order flux correction: the first-order HLLE flux of the state at the start.
      let sl = mq_start(cl);
      let sr = mq_start(cr);
      F = face_of(sl, sl, sr, sr, kept_face(cr, a), a, false, 1.0, vec4<f32>(0.0), vec4<f32>(0.0), dt);
    } else {
      F = face_of(qll, ql, qr, qrr, textureLoad(FA, cr, 0)[a], a, STAGE == 2u, w, vec4<f32>(0.0), vec4<f32>(0.0), dt);
    }
    if (across.x + across.y + across.z == 0) { F.f = mhd_add_diffusion(F.f, atl(o, t - e), e, a, ql, qr); }
    let fa = vec4<f32>(F.f.m, F.f.mom);
    let fb = vec4<f32>(F.f.e, F.f.ei, F.f.s, F.f.vn);
    let i = flux_entry(inv.row, a, t);
    FL[i] = fa;
    FL[i + 1u] = fb;
    FL[i + 2u] = vec4<f32>(F.f.bt, F.w, 0.0);
    // A face on the block's own lower or upper face: into the register.
    if ((t[a] == 0 || t[a] == BS) && across.x + across.y + across.z == 0) {
      let t1 = t[(a + 1) % 3];
      let t2 = t[(a + 2) % 3];
      let r = reg_entry(slot, a, t[a] == BS, select(t1, 0, is2d() && (a + 1) % 3 == 2), select(t2, 0, is2d() && (a + 2) % 3 == 2));
      REG[r] = fa;
      REG[r + 1u] = fb;
    }
  }
}

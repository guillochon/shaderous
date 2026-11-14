// MHD face fluxes on the block grid, for one stage (mhd/flux.wgsl does the same on the uniform
// grid). Each invocation takes the lower faces of one cell t of a leaf block, t in [-1, BS]
// along each axis: those whose index along their own normal is in [0, BS], and which are at
// most one cell outside the block across it (the edge fields beside the block's edges need
// the ghost rows next to it; nothing needs the ghost corners). The states come from the
// block's assembled storage, ghosts included (mhd/amr_assemble.wgsl).
//
// Writes the chunk's flux scratch, and the flux register for the faces on the block's faces.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/pfloor.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;
@group(0) @binding(3) var W2: texture_3d<f32>;
@group(0) @binding(4) var W3: texture_3d<f32>;
@group(0) @binding(5) var FA: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> FL: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> REG: array<vec4<f32>>;
@group(0) @binding(14) var<uniform> CHUNK: Chunk;

override STAGE: u32 = 1u;

#include "mhd/state.wgsl"
#include "mhd/amr_block.wgsl"
#include "mhd/riemann/common.wgsl"
#include <mhd_riemann>
#include "mhd/face_flux.wgsl"

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let inv = inv_box(wid, lix, -1, FBOX);
  if (!inv.ok) { return; }
  let slot = inv.slot;
  let t = inv.l;
  g_dx = blk_dx(slot);
  let o = blk_origin(slot);
  let dt = select(clock.dt, 0.5 * clock.dt, STAGE == 1u);
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
    let bn = textureLoad(FA, atl(o, t), 0)[a];
    let F = face_of(qll, ql, qr, qrr, bn, a, STAGE == 2u, w, vec4<f32>(0.0), vec4<f32>(0.0), dt);
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

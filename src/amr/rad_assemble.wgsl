// Radiation on the block grid, before the advection and before every sweep (amr/radiation.ts;
// docs/numerics.md section 24): the radiation (E, F) and kappa_R rho of every leaf's cells and
// two layers of ghosts, RA and RK, from wherever they live, as amr/assemble.wgsl builds the
// primitives: the cell itself, a same-level neighbor's cell or its image at a boundary (the flux
// reversed across a wall), the mean of a finer block's cells, or the coarser cell beyond the
// face, interpolated linearly with minmod slopes (which keep E positive) and its flux held to
// c E. kappa_R rho is taken as it is from a coarser cell and averaged from finer ones.
//
// The gas's temperature, for the opacity, is the state's on the first sweep of a step (and for
// the advection), and the sweeps' own (G) after it.
#include "hydro/common.wgsl"
#include <eos>
#include <opacity>
#include "radiation/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var U1: texture_3d<f32>;
@group(0) @binding(2) var U2: texture_3d<f32>;
@group(0) @binding(3) var RI: texture_3d<f32>;    // radiation (E, F) of the leaves' cells
@group(0) @binding(4) var GI: texture_3d<f32>;    // the gas's side of the sweeps: (rho e, momentum gained)
@group(0) @binding(5) var RA: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var RK: texture_storage_3d<r32float, write>;
struct SweepInfo { first: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(8) var<uniform> SW: SweepInfo;

struct RC { R: vec4<f32>, chi: f32 }
struct MaybeRC { x: RC, ok: bool }

fn leaf_cell(slot: i32, l: vec3<i32>) -> RC {
  let c = atl(blk_origin(slot), l);
  let rho = max(textureLoad(U1, c, 0).x, P.rho_floor);
  var eint = textureLoad(U2, c, 0).y;
  if (SW.first == 0u) { eint = textureLoad(GI, c, 0).x; }
  comp_use(comp_tex(c));
  let T = eos_temperature(rho, eint);
  return RC(textureLoad(RI, c, 0), kappa_rosseland(rho, T) * rho);
}

// Interior cell l of any block: a leaf stores it, a parent's is the mean of its children's.
fn rc_at(slot: i32, l: vec3<i32>) -> RC {
  if (blk_leaf(slot)) { return leaf_cell(slot, l); }
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  let ch = bt(slot, 12u + u32(h.x | (h.y << 1u) | (h.z << 2u)));
  var x = RC(vec4<f32>(0.0), 0.0);
  // Refined further: 2:1 balance keeps every leaf's stencil from reaching here.
  if (ch < 0 || !blk_leaf(ch)) { return x; }
  let f = 2 * (l - (BS / 2) * h);
  let nz = select(2, 1, is2d());
  for (var k = 0; k < nz; k++) {
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        let c = leaf_cell(ch, f + vec3<i32>(i, j, k));
        x.R += c.R;
        x.chi += c.chi;
      }
    }
  }
  let w = select(0.125, 0.25, is2d());
  return RC(x.R * w, x.chi * w);
}

// A cell next to the interior of block `slot` (inside it or beyond a face), if a block at its
// level holds it.
fn rc_near(slot: i32, l: vec3<i32>) -> MaybeRC {
  let s = ghost_src(slot, l);
  if (s.kind != 0) { return MaybeRC(RC(vec4<f32>(0.0), 0.0), false); }
  var x = rc_at(s.slot, s.l);
  x.R = vec4<f32>(x.R.x, x.R.yzw * s.flip);
  return MaybeRC(x, true);
}

fn minmod4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return select(vec4<f32>(0.0), sign(a) * min(abs(a), abs(b)), sign(a) * sign(b) > vec4<f32>(0.0));
}

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  let inv = inv_flat(wid, lix, NGH);
  if (!inv.ok) { return; }
  let s = ghost_src(inv.slot, inv.l);
  if (s.kind == 2) { return; }
  var x: RC;
  if (s.kind == 0) {
    x = rc_at(s.slot, s.l);
    x.R = vec4<f32>(x.R.x, x.R.yzw * s.flip);
  } else {
    // The coarser leaf's cell that covers this ghost (amr/assemble.wgsl from_coarse).
    let gf = blk_coord(inv.slot) * BS + inv.l;
    let nc = i32(P.amr.y) * BS * (1 << u32(bt(s.slot, 0u)));
    var gc = gf >> vec3<u32>(1u);
    gc = ((gc % nc) + nc) % nc;
    var lc = gc - blk_coord(s.slot) * BS;
    if (is2d()) { lc.z = 0; }
    x = rc_at(s.slot, lc);
    let off = vec3<f32>(gf & vec3<i32>(1)) - 0.5;
    let axes = select(3, 2, is2d());
    var R = x.R;
    for (var a = 0; a < axes; a++) {
      var e = vec3<i32>(0);
      e[a] = 1;
      let m = rc_near(s.slot, lc - e);
      let p = rc_near(s.slot, lc + e);
      if (!m.ok || !p.ok) { continue; }
      R += 0.5 * off[a] * minmod4(x.R - m.x.R, p.x.R - x.R);
    }
    x.R = R;
  }
  let E = max(x.R.x, 0.0);
  var F = x.R.yzw;
  let fm = length(F);
  if (fm > rad_c() * E) { F *= rad_c() * E / max(fm, 1.0e-30); }
  let c = atl(blk_origin(inv.slot), inv.l);
  textureStore(RA, c, vec4<f32>(E, F));
  textureStore(RK, c, vec4<f32>(x.chi, 0.0, 0.0, 0.0));
}

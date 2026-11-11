// Unsplit MUSCL-Hancock update on the block grid: the same gather as hydro/update.wgsl, over
// the cells of leaf blocks, with ghost cells read straight from the block's own storage.
//
// Flux matching at refinement boundaries. Where a leaf's face neighbour is refined, the coarse
// cell does not use its own flux through that face. It recomputes the fine fluxes instead,
// from the fine blocks' stored states with the same function the fine cells use, and takes
// their average. Both sides then agree on what crossed the face, so mass, momentum and energy
// are conserved across levels without storing any fluxes.
//
// Per-block timesteps (amr/bins_on.wgsl) extend the same idea in time. Across a face whose
// other side steps twice as often, a block does not use its own flux either: it takes the
// faster side's, recomputed from that side's stored states, for the half of its step that
// the faster side is now starting. The first half goes into the block's own update; the
// second is added when the faster side starts its second step, in a pass over the block that
// changes nothing else (`deposit` below). What crossed the face is then the same on both
// sides, again without storing fluxes. The bins are such that the faster side is never the
// coarser one, so at a refinement boundary it is still the fine fluxes that count.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"
#include "hydro/fluff.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>
#include "amr/blk_dt.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var W1: texture_3d<f32>;
@group(0) @binding(5) var W2: texture_3d<f32>;
@group(0) @binding(6) var G: texture_3d<f32>;
@group(0) @binding(7) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var PA: texture_3d<f32>;
@group(0) @binding(10) var PB: texture_3d<f32>;

#include "hydro/q.wgsl"

// Ghost cells are stored, so loads are plain texel reads at atlas coordinates.
fn load_q(c: vec3<i32>) -> Q {
  let w1 = textureLoad(W1, c, 0);
  let w2 = textureLoad(W2, c, 0);
  // A negative density marks a solid cell (amr/assemble.wgsl).
  return Q(vec4<f32>(abs(w1.x), w1.yzw), vec4<f32>(w2.x, w2.z, w2.y, select(0.0, 1.0, w1.x < 0.0)));
}
fn load_pred(c: vec3<i32>) -> Q {
  return Q(textureLoad(PA, c, 0), textureLoad(PB, c, 0));
}


// For the well-balanced pressure reconstruction (hydro/q.wgsl): the potential difference
// between the adjacent cells l + k e and l + (k + 1) e of block `slot` along an axis, as
// (seen, face), from the field atlas (which holds the potential on two ghost layers) plus the
// uniform field, which is exact. `seen` is what the deviations from balance are measured
// against: beyond a face of the domain the potential is that of the cell the state there is
// taken from, like the state itself. `face` is what takes the pressure from a cell's centre
// to its face: the same, except across a reflecting face of the domain, where it is the true
// difference (the atlas's ghost cells there continue the potential in a straight line).
fn dphi_blk(slot: i32, l: vec3<i32>, axis: i32, k: i32) -> vec2<f32> {
  let ra = l[axis] + k;
  let ia = row_src(slot, axis, ra);
  let ib = row_src(slot, axis, ra + 1);
  let o = blk_origin(slot);
  var a = l;
  var b = l;
  a[axis] = ia;
  b[axis] = ib;
  var seen = -P.gext[axis] * f32(ib - ia) * g_dx;
  if (P.grav_on != 0u) { seen += textureLoad(G, atl(o, b), 0).w - textureLoad(G, atl(o, a), 0).w; }
  var face = seen;
  let wall = (ra == -1 && bt(slot, 6u + u32(2 * axis)) == -3) || (ra == BS - 1 && bt(slot, 7u + u32(2 * axis)) == -3);
  if (wall) {
    a[axis] = ra;
    b[axis] = ra + 1;
    face = -P.gext[axis] * g_dx;
    if (P.grav_on != 0u) { face += textureLoad(G, atl(o, b), 0).w - textureLoad(G, atl(o, a), 0).w; }
  }
  return vec2<f32>(seen, face);
}

// The cell being updated, for dphi_step.
var<private> g_slot: i32 = 0;
var<private> g_cell: vec3<i32> = vec3<i32>(0);

fn dphi_step(c: vec3<i32>, axis: i32, k: i32) -> vec2<f32> {
  return dphi_blk(g_slot, g_cell, axis, k);
}
// Within another block's stored cells and ghosts, across a face it shares with a block: no
// boundary of the domain is involved.
fn dphi_face(a: vec3<i32>, axis: i32, k: i32) -> f32 {
  var e = vec3<i32>(0);
  e[axis] = 1;
  var d = -P.gext[axis] * g_dx;
  if (P.grav_on != 0u) { d += textureLoad(G, a + (k + 1) * e, 0).w - textureLoad(G, a + k * e, 0).w; }
  return d;
}

#include "hydro/flux.wgsl"
#include "hydro/advance.wgsl"
#include "hydro/sources.wgsl"
#include <cooling>
#include "hydro/cooling.wgsl"

fn flux_add(a: Flux, b: Flux) -> Flux {
  return Flux(a.m + b.m, a.mom + b.mom, a.e + b.e, a.ei + b.ei, a.s + b.s, a.vn + b.vn);
}
fn flux_scale(f: Flux, w: f32) -> Flux {
  return Flux(f.m * w, f.mom * w, f.e * w, f.ei * w, f.s * w, f.vn * w);
}

// The block being worked on: its bin, and whether this pass only deposits the second half of
// its faster neighbours' fluxes (the block is half way through its step).
var<private> g_bin: u32 = 0u;
var<private> g_deposit: bool = false;
var<private> g_touched: bool = false;

// Share of this block's step for which the flux of the neighbouring leaf `nb` is to be used
// now: all of it for a neighbour in step with the block, half for a faster one, and in a
// deposit pass nothing for the former.
fn share(nb: i32) -> f32 {
  if (!BINS_ON) { return 1.0; }
  if (blk_bin(nb) < g_bin) { return 0.5; }
  return select(1.0, 0.0, g_deposit);
}

// Average of the fine fluxes through the face that coarse cell l shares with the refined
// neighbour block `np`. `upper` says the fine blocks lie on the upper side of `np` along `axis`
// (the coarse cell is beyond them), in which case the face is the fine cells' upper face.
fn fine_flux(np: i32, axis: i32, upper: bool, l: vec3<i32>) -> Flux {
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  var bits = h;
  bits[axis] = i32(upper);
  if (is2d()) { bits.z = 0; }
  let ch = bt(np, 12u + u32(bits.x | (bits.y << 1u) | (bits.z << 2u)));
  let part = share(ch);
  if (part == 0.0) { return flux_zero(); }
  g_touched = true;
  let o = blk_origin(ch);
  var e = vec3<i32>(0);
  e[axis] = 1;
  // The fine cell on the lower side of the face: the last interior cell, or the ghost at -1.
  var f = 2 * (l - (BS / 2) * h);
  f[axis] = select(-1, BS - 1, upper);
  let t1 = (axis + 1) % 3;
  let t2 = (axis + 2) % 3;
  // In 2D the faces are edges: two fine ones per coarse one, both in the plane.
  let n1 = select(2, 1, is2d() && t1 == 2);
  let n2 = select(2, 1, is2d() && t2 == 2);

  // The fine cells computed these fluxes with their own cell size.
  let dx_coarse = g_dx;
  g_dx = 0.5 * dx_coarse;
  var sum = flux_zero();
  for (var j = 0; j < n2; j++) {
    for (var i = 0; i < n1; i++) {
      var a = f;
      a[t1] += i;
      a[t2] += j;
      sum = flux_add(sum, flux_between(atl(o, a), e, axis));
    }
  }
  g_dx = dx_coarse;
  var w = 1.0 / f32(n1 * n2);
  if (BINS_ON) { w *= part; }
  return flux_scale(sum, w);
}

// The flux through the face shared with the same-level leaf `np`, as `np` computes it, for a
// neighbour that steps twice as often; otherwise the block's own flux `own`. `upper` as in
// fine_flux: `np` lies below the block, and the face is the upper face of its last cells.
fn leaf_flux(own: Flux, np: i32, axis: i32, upper: bool, l: vec3<i32>) -> Flux {
  if (blk_bin(np) >= g_bin) { return own; }
  g_touched = true;
  var e = vec3<i32>(0);
  e[axis] = 1;
  var a = l;
  a[axis] = select(-1, BS - 1, upper);
  return flux_scale(flux_between(atl(blk_origin(np), a), e, axis), 0.5);
}

// Replace the fluxes through faces shared with a refined neighbour.
fn match_faces(fl: Faces, slot: i32, axis: i32, l: vec3<i32>) -> Faces {
  var out = fl;
  if (l[axis] == 0) {
    let np = bt(slot, 6u + u32(2 * axis));
    if (np >= 0 && !blk_leaf(np)) { out.lo = fine_flux(np, axis, true, l); }
    else if (BINS_ON && np >= 0 && np != slot) { out.lo = leaf_flux(out.lo, np, axis, true, l); }
  }
  if (l[axis] == BS - 1) {
    let np = bt(slot, 7u + u32(2 * axis));
    if (np >= 0 && !blk_leaf(np)) { out.hi = fine_flux(np, axis, false, l); }
    else if (BINS_ON && np >= 0 && np != slot) { out.hi = leaf_flux(out.hi, np, axis, false, l); }
  }
  return out;
}

// Validation hook: when set, the pass writes the face mass fluxes instead of the new state.
override DEBUG_FLUX: bool = false;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let slot = inv.slot;
  let l = inv.l;
  g_dx = blk_dx(slot);
  g_dt = blk_dt(slot);
  g_bin = blk_bin(slot);
  g_deposit = BINS_ON && !blk_active(slot);
  g_slot = slot;
  g_cell = l;
  let c = atl(blk_origin(slot), l);
  if (has_solids()) {
    // Solid and fixed cells keep their state.
    let held = textureLoad(U2, c, 0);
    if (abs(held.w) > 0.5) {
      textureStore(O1, c, textureLoad(U1, c, 0));
      textureStore(O2, c, held);
      return;
    }
  }
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);

  let w0 = load_q(c);
  let p0 = load_pred(c);
  let none = Faces(flux_zero(), flux_zero());
  var fx = none;
  var fy = none;
  var fz = none;
  if (g_deposit) {
    // Half way through the block's step: only what its faster neighbours carry across the
    // shared faces during their second step. A cell on no such face is copied as it is.
    fx = match_faces(none, slot, 0, l);
    fy = match_faces(none, slot, 1, l);
    if (!is2d()) { fz = match_faces(none, slot, 2, l); }
    if (!g_touched) {
      textureStore(O1, c, textureLoad(U1, c, 0));
      textureStore(O2, c, textureLoad(U2, c, 0));
      return;
    }
  } else {
    fx = match_faces(axis_fluxes(c, ex, 0, w0, p0), slot, 0, l);
    fy = match_faces(axis_fluxes(c, ey, 1, w0, p0), slot, 1, l);
    if (!is2d()) { fz = match_faces(axis_fluxes(c, ez, 2, w0, p0), slot, 2, l); }
  }

  if (DEBUG_FLUX) {
    textureStore(O1, c, vec4<f32>(fx.lo.m, fx.hi.m, fy.lo.m, fy.hi.m));
    textureStore(O2, c, vec4<f32>(fz.lo.m, fz.hi.m, 0.0, 0.0));
    return;
  }

  // Gravity: a uniform external field, self-gravity from the field atlas, or both. The step is
  // kick-drift-kick. With self-gravity the second half kick needs the new potential and runs
  // as its own pass (amr/correct.wgsl); with only the uniform field it is applied here.
  let self_gravity = P.grav_on != 0u;
  var g = P.gext.xyz;
  if (self_gravity) { g += textureLoad(G, c, 0).xyz; }
  let gravity = self_gravity || any(P.gext.xyz != vec3<f32>(0.0));
  let flux_work = gravity && (P.flags & 1u) != 0u;
  var dw = 0.0;
  if (flux_work) {
    // Flux-form work: -1/2 (F_hi dphi(+e) - F_lo dphi(-e)). For the uniform field dphi(+-e) is
    // -+ g . e dx; the self-gravity part differences the half-step potential, ghosts included.
    let ge = P.gext.xyz;
    dw = 0.5 * g_dx * (ge.x * (fx.hi.m + fx.lo.m) + ge.y * (fy.hi.m + fy.lo.m) + ge.z * (fz.hi.m + fz.lo.m));
    if (self_gravity) {
      let pc = textureLoad(G, c, 0).w;
      dw -= 0.5 * (fx.hi.m * (textureLoad(G, c + ex, 0).w - pc) - fx.lo.m * (textureLoad(G, c - ex, 0).w - pc)
                 + fy.hi.m * (textureLoad(G, c + ey, 0).w - pc) - fy.lo.m * (textureLoad(G, c - ey, 0).w - pc)
                 + fz.hi.m * (textureLoad(G, c + ez, 0).w - pc) - fz.lo.m * (textureLoad(G, c - ez, 0).w - pc));
    }
  }

  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  var kick = vec3<f32>(0.0);
  var ekick = 0.0;
  if (gravity && !g_deposit) {
    kick = 0.5 * g_dt * u1.x * g;
    if (!flux_work) { ekick = dot(kick, u1.yzw + 0.5 * kick) / max(u1.x, P.rho_floor); }
  }

  var nw = advance(u1, u2, net_inflow(fx, fy, fz), dw, p0.b.x, kick, ekick);
  if (g_deposit) {
    // The kicks, sources and cooling of this step were applied when it started.
    textureStore(O1, c, nw.u1);
    textureStore(O2, c, nw.u2);
    return;
  }

  if (gravity && !self_gravity) {
    // Second half kick, with the updated density.
    let rho = nw.u1.x;
    let k2 = 0.5 * g_dt * rho * g;
    let mom = nw.u1.yzw + k2;
    var E = nw.u2.x;
    var eint = nw.u2.y;
    if (flux_work) {
      let en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);
      E = en.x;
      eint = en.y;
    } else {
      E += dot(k2, nw.u1.yzw + 0.5 * k2) / rho;
    }
    nw = NewState(vec4<f32>(rho, mom), vec4<f32>(E, eint, nw.u2.z, 0.0));
  }
  if (has_sources()) { nw = apply_sources(nw, blk_pos(slot, l) / P.box + 0.5, g_dt); }
  nw = apply_cooling(nw, g_dt);
  textureStore(O1, c, nw.u1);
  textureStore(O2, c, nw.u2);
}

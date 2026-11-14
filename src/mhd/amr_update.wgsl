// MHD update on the block grid, for one stage (mhd/update.wgsl on the uniform grid): the
// conservative update of each leaf cell and the constrained-transport update of the block's
// faces (with compensated sums), over the texels [0, BS] of each block.
//
// Refinement boundaries (docs/mhd-plan.md, section 4b):
//  - a cell beside a refined neighbor takes the mean of the fine fluxes through that face, from
//    the fine blocks' flux register, in place of its own;
//  - every edge has one value, set by the finest blocks around it: for an edge on the block's
//    surface the leaves of the cells around it are looked up (IDX), and where one is finer the
//    mean of the fine block's two impulses along the edge (one at a 2D vertex) is used, scaled
//    to this block's cell size. That covers the faces in the refinement boundary and the faces
//    beside it, and edges a block shares with a finer one only diagonally.
// Blocks at the same level compute shared faces and edges from identical inputs, so their
// copies agree.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"
#include "hydro/fluff.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;   // state at the start of the step
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;   // its faces
@group(0) @binding(5) var BC: texture_3d<f32>;   // and their compensations
@group(0) @binding(6) var<storage, read> FL: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> REG: array<vec4<f32>>;
@group(0) @binding(8) var EI: texture_3d<f32>;    // edge impulses of every leaf
@group(0) @binding(9) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(10) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(11) var OB: texture_storage_3d<rgba32float, write>;
@group(0) @binding(13) var OC: texture_storage_3d<rgba32float, write>;
@group(0) @binding(14) var<uniform> CHUNK: Chunk;
@group(0) @binding(15) var W1: texture_3d<f32>;   // the stage's assembled state
@group(0) @binding(16) var W2: texture_3d<f32>;
@group(0) @binding(17) var W3: texture_3d<f32>;
@group(0) @binding(28) var IDX: texture_3d<u32>;

override STAGE: u32 = 1u;

#include "mhd/state.wgsl"
#include "mhd/amr_block.wgsl"

// For the lookups of amr_common's that this pass uses (map_cell, holder, local_of).
fn n_at(level: u32) -> vec3<i32> {
  let n = i32(P.amr.y) * BS * (1 << level);
  return select(vec3<i32>(n), vec3<i32>(n, n, 1), is2d());
}
struct GCell { g: vec3<i32>, flip: vec3<f32>, side: vec3<f32> }
fn map_cell(level: u32, g0: vec3<i32>) -> GCell {
  let n = n_at(level);
  var o = GCell(g0, vec3<f32>(1.0), vec3<f32>(0.0));
  for (var a = 0; a < 3; a++) {
    let i = g0[a];
    if (i >= 0 && i < n[a]) { continue; }
    let up = i >= n[a];
    let bc = select(P.bc_lo[a], P.bc_hi[a], up);
    if (bc == 0u) {
      o.g[a] = ((i % n[a]) + n[a]) % n[a];
    } else if (bc == 2u) {
      o.g[a] = select(-1 - i, 2 * n[a] - 1 - i, up);
      o.flip[a] = -1.0;
    } else {
      o.g[a] = select(0, n[a] - 1, up);
    }
  }
  return o;
}
struct Holder { slot: i32, rel: i32 }
fn holder(level: u32, g: vec3<i32>) -> Holder {
  let shift = P.amr.z - level;
  var fb = (g << vec3<u32>(shift)) / BS;
  if (is2d()) { fb.z = 0; }
  var s = i32(textureLoad(IDX, fb, 0).x);
  let ls = u32(bt(s, 0u));
  if (ls == level) { return Holder(s, 0); }
  if (ls < level) { return Holder(s, -1); }
  for (var k = ls; k > level; k--) { s = bt(s, 5u); }
  return Holder(s, 1);
}
fn local_of(slot: i32, g: vec3<i32>) -> vec3<i32> {
  var l = g - blk_coord(slot) * BS;
  if (is2d()) { l.z = 0; }
  return l;
}

var<private> g_slot: i32;
var<private> g_o: vec3<i32>;
var<private> g_level: u32;

// The block's bin, and whether this is its deposit (time bins: it is at its midpoint, and takes
// its faster neighbors' second substep on the faces and edges it shares with them).
var<private> g_bin: u32 = 0u;
var<private> g_deposit: bool = false;
var<private> g_touched: bool = false;

// What this block's update adds along edge c at its edge texel t (docs/mhd-plan.md, 4b and 5b).
// Each edge has one definer, the finest leaf around it and among those the fastest; this block
// integrates the definer's impulses over its own step:
//  - itself (or a leaf in step with it at its level): its own impulse;
//  - a finer leaf: the mean of its two impulses along the edge (one at a 2D vertex), times
//    dx_fine / dx = 1/2;
//  - a leaf one bin faster: its impulse for its first substep now (stage 2), and the one for its
//    second at this block's midpoint (the deposit). Stage 1, which only makes the half-step
//    state, keeps this block's own.
// In a deposit only the edges with a faster definer contribute.
fn beyond_wall(level: u32, g: vec3<i32>) -> bool {
  let n = n_at(level);
  for (var a = 0; a < naxes(); a++) {
    if (g[a] < 0 && P.bc_lo[a] != 0u) { return true; }
    if (g[a] >= n[a] && P.bc_hi[a] != 0u) { return true; }
  }
  return false;
}

fn impulse(c: i32, t: vec3<i32>) -> f32 {
  let a = (c + 1) % 3;
  let b = (c + 2) % 3;
  let flat_a = is2d() && a == 2;
  let flat_b = is2d() && b == 2;
  let on_a = !flat_a && (t[a] == 0 || t[a] == BS);
  let on_b = !flat_b && (t[b] == 0 || t[b] == BS);
  let own = textureLoad(EI, atl(g_o, t), 0)[c];
  if (!on_a && !on_b) { return select(own, 0.0, g_deposit); }
  // The cells around the edge: offsets 0 or -1 along a and b (only along the axis in use in 2D).
  let g = blk_coord(g_slot) * BS + t;
  let na = select(2, 1, flat_a);
  let nb = select(2, 1, flat_b);
  var best = own;
  var best_level = g_level;
  var best_bin = g_bin;
  for (var j = 0; j < nb; j++) {
    for (var i = 0; i < na; i++) {
      var cell = t;
      cell[a] -= i;
      cell[b] -= j;
      let inside = all(cell >= vec3<i32>(0)) && all(cell < vec3<i32>(BS, BS, select(BS, 1, is2d())));
      if (inside) { continue; }
      // Beyond a wall or open boundary no block defines the edge (the cell would map back
      // into this one or a neighbor, off by a cell); across a periodic one the wrap is exact.
      let raw = blk_coord(g_slot) * BS + cell;
      if (beyond_wall(g_level, raw)) { continue; }
      let gm = map_cell(g_level, raw).g;
      let h = holder(g_level, gm);
      if (h.rel == 1) {
        // A finer leaf: its edge segments along this one, from the fine cell beside the edge.
        var fcell = 2 * g;
        fcell[a] -= i;
        fcell[b] -= j;
        if (is2d()) { fcell.z = 0; }
        let fm = map_cell(g_level + 1u, fcell).g;
        let f = holder(g_level + 1u, fm);
        let fb = blk_bin(f.slot);
        if (best_level == g_level + 1u && fb >= best_bin) { continue; }
        var le = local_of(f.slot, fm);
        le[a] += i;
        le[b] += j;
        if (is2d()) { le.z = 0; }
        let fo = blk_origin(f.slot);
        if (is2d() && c == 2) {
          best = 0.5 * textureLoad(EI, atl(fo, le), 0)[c];
        } else {
          var le2 = le;
          le2[c] += 1;
          best = 0.25 * (textureLoad(EI, atl(fo, le), 0)[c] + textureLoad(EI, atl(fo, le2), 0)[c]);
        }
        best_level = g_level + 1u;
        best_bin = fb;
      } else if (BINS_ON && h.rel == 0 && h.slot != g_slot && best_level == g_level) {
        // A leaf at this level, faster than the best so far: its own impulse along the edge.
        let qb = blk_bin(h.slot);
        if (qb >= best_bin) { continue; }
        var le = local_of(h.slot, gm);
        le[a] += i;
        le[b] += j;
        if (is2d()) { le.z = 0; }
        best = textureLoad(EI, atl(blk_origin(h.slot), le), 0)[c];
        best_bin = qb;
      }
    }
  }
  let faster = best_bin < g_bin;
  if (g_deposit) {
    if (faster) { g_touched = true; }
    return select(0.0, best, faster);
  }
  if (faster && STAGE == 1u) { return own; }
  return best;
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = opaque(a + b);
  let bb = opaque(s - a);
  // Each error term through opaque() too: (a - x) + (b - bb) is otherwise free to be regrouped
  // as (a + b) - (x + bb), which loses the very rounding this recovers. Firefox's shader compiler
  // did (div B grew as an uncompensated sum's, 4e-6 after 751 steps of Orszag-Tang).
  return vec2<f32>(s, opaque(a - opaque(s - bb)) + opaque(b - bb));
}

// Face along a of texel t after the stage, as (b, c), from the pair at the start.
fn new_face(a: i32, t: vec3<i32>) -> vec2<f32> {
  let b = (a + 1) % 3;
  let c = (a + 2) % 3;
  var eb = vec3<i32>(0);
  eb[b] = 1;
  var ec = vec3<i32>(0);
  ec[c] = 1;
  let x = atl(g_o, t);
  var s = textureLoad(BF, x, 0)[a];
  var comp = textureLoad(BC, x, 0)[a];
  // The four edges, in this order: -E_c at t + eb, +E_c at t, +E_b at t + ec, -E_b at t. In one
  // loop, so impulse() is compiled into it once (loop_count, amr/grid.wgsl).
  for (var k = 0; k < loop_count(4); k++) {
    let along_c = k < 2;
    if (is2d() && select(c, b, along_c) == 2) { continue; }
    var off = vec3<i32>(0);
    if (k == 0) { off = eb; }
    if (k == 2) { off = ec; }
    let sgn = select(1.0, -1.0, k == 0 || k == 3);
    let r = two_sum(s, sgn * impulse(select(b, c, along_c), t + off));
    s = r.x;
    comp += r.y;
  }
  let hi = opaque(s + comp);
  return vec2<f32>(hi, comp - opaque(hi - s));
}

struct Fl { a: vec4<f32>, b: vec4<f32> }

// Flux through the lower face along `axis` of cell t of this block (t[axis] in [0, BS]), per
// unit of this block's step:
//  - on a face of the block whose neighbor is refined, the mean of the fine fluxes (register);
//  - with time bins, across a face whose other side is one bin faster (a same-level leaf, or
//    the fine children), that side's flux for its first substep times 1/2 now (stage 2), the
//    other half at this block's midpoint (the deposit); stage 1 keeps its own;
//  - otherwise its own (scratch). A deposit takes only the faster sides' faces.
fn flux_at(row: u32, axis: i32, t: vec3<i32>) -> Fl {
  let hi = t[axis] == BS;
  let zero = Fl(vec4<f32>(0.0), vec4<f32>(0.0));
  if (t[axis] == 0 || hi) {
    let np = bt(g_slot, 6u + u32(2 * axis) + select(0u, 1u, hi));
    if (np >= 0 && !blk_leaf(np)) {
      let l = t;
      let h = vec3<i32>(l >= vec3<i32>(BS / 2));
      var bits = h;
      bits[axis] = select(1, 0, hi);
      if (is2d()) { bits.z = 0; }
      let ch = bt(np, 12u + u32(bits.x | (bits.y << 1u) | (bits.z << 2u)));
      let faster = BINS_ON && blk_bin(ch) < g_bin;
      if (g_deposit && !faster) { return zero; }
      if (!(faster && STAGE == 1u)) {
        if (g_deposit) { g_touched = true; }
        let t1 = (axis + 1) % 3;
        let t2 = (axis + 2) % 3;
        let m1 = select(2, 1, is2d() && t1 == 2);
        let m2 = select(2, 1, is2d() && t2 == 2);
        var fa = vec4<f32>(0.0);
        var fb = vec4<f32>(0.0);
        for (var j = 0; j < m2; j++) {
          for (var i = 0; i < m1; i++) {
            let f1 = select(2 * (l[t1] - (BS / 2) * h[t1]) + i, 0, m1 == 1);
            let f2 = select(2 * (l[t2] - (BS / 2) * h[t2]) + j, 0, m2 == 1);
            let r = reg_entry(ch, axis, !hi, f1, f2);
            fa += REG[r];
            fb += REG[r + 1u];
          }
        }
        let w = select(1.0, 0.5, faster) / f32(m1 * m2);
        return Fl(fa * w, fb * w);
      }
    } else if (BINS_ON && np >= 0 && np != g_slot && blk_leaf(np) && blk_bin(np) < g_bin) {
      // A faster leaf at this level: its flux through the face, from its side of the register.
      if (STAGE == 2u) {
        if (g_deposit) { g_touched = true; }
        let t1 = (axis + 1) % 3;
        let t2 = (axis + 2) % 3;
        let r = reg_entry(np, axis, !hi, select(t[t1], 0, is2d() && t1 == 2), select(t[t2], 0, is2d() && t2 == 2));
        return Fl(0.5 * REG[r], 0.5 * REG[r + 1u]);
      }
    }
  }
  if (g_deposit) { return zero; }
  let i = flux_entry(row, axis, t);
  return Fl(FL[i], FL[i + 1u]);
}

// Shock flag of cell l at the start of the step, from the assembled start state.
fn shock_flag(l: vec3<i32>) -> f32 {
  var conv = 0.0;
  var jump = 0.0;
  for (var a = 0; a < naxes(); a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let m = mq_block(g_o, l - e);
    let p = mq_block(g_o, l + e);
    conv += p.a[a + 1] - m.a[a + 1];
    let pm = m.b.x + 0.5 * dot(m.B, m.B);
    let pp = p.b.x + 0.5 * dot(p.B, p.B);
    jump = max(jump, abs(pp - pm) / min(pp, pm));
  }
  return select(0.0, 1.0, conv < 0.0 && jump > P.shock_thresh);
}

@compute @workgroup_size(FLAT_WG)
fn main(@builtin(local_invocation_index) lix: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let inv = inv_box(wid, lix, 0, BS + 1);
  if (!inv.ok) { return; }
  let t = inv.l;
  g_slot = inv.slot;
  g_o = blk_origin(inv.slot);
  g_level = u32(bt(inv.slot, 0u));
  g_bin = blk_bin(inv.slot);
  // Time bins: a block dispatched while not starting a step takes its deposit.
  g_deposit = BINS_ON && !blk_active(inv.slot);
  g_dx = blk_dx(inv.slot);
  let dt = stage_dt(inv.slot, STAGE);
  g_dt = dt;
  let dtdx = dt / g_dx;
  let depth = select(BS, 1, is2d());
  let top = vec3<i32>(BS, BS, depth);

  // The block's faces at this texel (i < 3: along a, index up to BS; across it, inside the
  // block), and for a cell, its upper faces (i >= 3), for its field. In one loop, so new_face is
  // compiled into it once (loop_count, amr/grid.wgsl). What the faces at this texel touch does
  // not count toward a deposit's (g_touched below); what the cell's upper faces touch does.
  let inside = all(t < top);
  var bout = vec3<f32>(0.0);
  var cout = vec3<f32>(0.0);
  var bup = vec3<f32>(0.0);
  g_touched = false;
  for (var i = 0; i < loop_count(6); i++) {
    let a = i % 3;
    let upper = i >= 3;
    var e = vec3<i32>(0);
    var ok = inside && a < naxes();
    if (upper) {
      e[a] = 1;
    } else {
      ok = t[a] <= top[a] && !(is2d() && a == 2 && t.z != 0);
      for (var b = 0; b < 3; b++) {
        if (b != a && t[b] >= top[b]) { ok = false; }
      }
    }
    if (!ok) { continue; }
    let touched = g_touched;
    let f = new_face(a, t + e);
    if (upper) {
      bup[a] = f.x;
    } else {
      bout[a] = f.x;
      cout[a] = f.y;
      g_touched = touched;
    }
  }
  let x = atl(g_o, t);
  textureStore(OB, x, vec4<f32>(bout, 0.0));
  if (STAGE == 2u) { textureStore(OC, x, vec4<f32>(cout, 0.0)); }

  if (!inside) { return; }
  var net_a = vec4<f32>(0.0);
  var net_b = vec4<f32>(0.0);
  var bc = vec3<f32>(0.0);
  // Each axis's lower and upper fluxes, in one loop (flux_at compiled into it once); the pair's
  // difference is taken before it is added, as it always was.
  var lo = Fl(vec4<f32>(0.0), vec4<f32>(0.0));
  for (var j = 0; j < loop_count(2 * naxes()); j++) {
    let a = j / 2;
    var e = vec3<i32>(0);
    if ((j & 1) == 1) { e[a] = 1; }
    let f = flux_at(inv.row, a, t + e);
    if ((j & 1) == 0) {
      lo = f;
    } else {
      net_a += lo.a - f.a;
      net_b += lo.b - f.b;
    }
  }
  for (var a = 0; a < 3; a++) {
    // (2D: B_z's face is the cell's only one along z.)
    if (a < naxes()) { bc[a] = 0.5 * (bout[a] + bup[a]); } else { bc[a] = bout[a]; }
  }
  let u1 = textureLoad(U1, x, 0);
  let u2 = textureLoad(U2, x, 0);
  // A deposit that touches neither the cell's fluxes nor its faces leaves it as it is.
  if (g_deposit && !g_touched) {
    textureStore(O1, x, u1);
    textureStore(O2, x, u2);
    return;
  }
  // Pressure for p div v: the stage's assembled state (the start in stage 1, the half step in
  // 2); a deposit, which is not assembled, its stored state.
  var ph = textureLoad(W2, x, 0).x;
  if (g_deposit) { ph = eos_pressure(max(u1.x, P.rho_floor), u2.y); }
  var flag = 0.0;
  if (STAGE == 1u) { flag = shock_flag(t); }
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
  textureStore(O1, x, vec4<f32>(rho, mout));
  textureStore(O2, x, vec4<f32>(en.x, en.y, sc, flag));
}

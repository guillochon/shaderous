// Unsplit MUSCL-Hancock update on the block grid: the same gather as hydro/update.wgsl, over
// the cells of leaf blocks, with ghost cells read straight from the block's own storage.
//
// Flux matching at refinement boundaries. Where a leaf's face neighbor is refined, the coarse
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
#include "hydro/drive.wgsl"
#include "gravity/ext_bind.wgsl"
#include <eos>
#include <limiter>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"
#include "hydro/fluff.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>
#include "amr/blk_dt.wgsl"
#include <frame>
#include <psplit>
#include <ap_store>

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
// Composition moments of a cell (species/mixture.ts), ghosts included: amr/assemble.wgsl
// stores them beside the primitives.
fn load_comp(c: vec3<i32>) -> vec4<f32> {
  if (!COMP) { return vec4<f32>(0.0); }
  return comp_tex(c);
}
// For viscosity (hydro/diffusion.wgsl): the velocity alone. A solid cell is at rest.
fn load_vel(c: vec3<i32>) -> vec3<f32> {
  let w1 = textureLoad(W1, c, 0);
  return select(w1.yzw, vec3<f32>(0.0), w1.x < 0.0);
}
// Also for viscosity. The face between the cell at atlas texel `a` and the next one along
// `axis` lies on the boundary of a block, in the first (+1) or last (-1) row of cells along
// the transverse axis `t`: beyond that row, on the far side of the face, are edge ghosts,
// which are never filled. Both blocks that share the face see it so, whichever one asks.
fn face_rim(a: vec3<i32>, axis: i32, t: i32) -> i32 {
  let la = a[axis] % PADW - NGH;
  if (la != -1 && la != BS - 1) { return 0; }
  let lt = a[t] % PADW - NGH;
  if (lt == 0) { return 1; }
  if (lt == BS - 1) { return -1; }
  return 0;
}


// For the well-balanced pressure reconstruction (hydro/q.wgsl): the potential difference
// between the adjacent cells l + k e and l + (k + 1) e of block `slot` along an axis, as
// (seen, face), from the field atlas (which holds the potential on two ghost layers) plus the
// uniform field, which is exact. `seen` is what the deviations from balance are measured
// against: beyond a face of the domain the potential is that of the cell the state there is
// taken from, like the state itself. `face` is what takes the pressure from a cell's center
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
  var e = vec3<f32>(0.0);
  e[axis] = g_dx;
  seen += ext_dphi(blk_pos(slot, a), e * f32(ib - ia), 0.0);
  var face = seen;
  // (A plane of mirror symmetry is no wall: the cell beyond it is the image, as for the state.)
  let wall = (ra == -1 && bt(slot, 6u + u32(2 * axis)) == -3 && !sym_lo(axis)) || (ra == BS - 1 && bt(slot, 7u + u32(2 * axis)) == -3);
  if (wall) {
    a[axis] = ra;
    b[axis] = ra + 1;
    face = -P.gext[axis] * g_dx;
    if (P.grav_on != 0u) { face += textureLoad(G, atl(o, b), 0).w - textureLoad(G, atl(o, a), 0).w; }
    face += ext_dphi(blk_pos(slot, a), e, 0.0);
  }
  return vec2<f32>(seen, face);
}

// The cell being updated, for dphi_step.
var<private> g_slot: i32 = 0;
var<private> g_cell: vec3<i32> = vec3<i32>(0);

fn dphi_step(c: vec3<i32>, axis: i32, k: i32) -> vec2<f32> {
  return dphi_blk(g_slot, g_cell, axis, k);
}
// For flux_between (hydro/flux.wgsl). Within another block's stored cells and ghosts, across a
// face it shares with a block, no boundary of the domain is involved: the plain difference.
// A face on the domain's edge, computed by the face-flux tile (amr/tile.wgsl), takes its steps
// from dphi_blk, as axis_fluxes does.
fn wall_step(a: vec3<i32>, axis: i32, across: f32) -> vec3<f32> {
  if (!g_face_wall || !well_balanced()) { return vec3<f32>(across, 1.0, 1.0); }
  let t = dphi_blk(g_face_slot, g_face_l, axis, -1).y;
  let image = select(1.0, -1.0, across != t);
  return vec3<f32>(t, select(1.0, image, g_face_l[axis] == 0), select(1.0, image, g_face_l[axis] == BS));
}
// The cell `a` that flux_between is given, as a block and a cell in it, for the point masses'
// part of dphi_face: its callers set it.
var<private> g_fb_slot: i32 = 0;
var<private> g_fb_l: vec3<i32> = vec3<i32>(0);

fn dphi_face(a: vec3<i32>, axis: i32, k: i32) -> f32 {
  if (g_face_wall) { return dphi_blk(g_face_slot, g_face_l, axis, k - 1).x; }
  var e = vec3<i32>(0);
  e[axis] = 1;
  var d = -P.gext[axis] * g_dx;
  if (P.grav_on != 0u) { d += textureLoad(G, a + (k + 1) * e, 0).w - textureLoad(G, a + k * e, 0).w; }
  return d + ext_dphi(blk_pos(g_fb_slot, g_fb_l + k * e), vec3<f32>(e) * g_dx, 0.0);
}

#include "hydro/flux.wgsl"
#include "amr/tile.wgsl"
#include <amr_walls>
#include <species>
#include "hydro/advance.wgsl"
#include "hydro/sources.wgsl"
#include "hydro/damping.wgsl"
#include <cooling>
#include "hydro/cooling.wgsl"

fn flux_add(a: Flux, b: Flux) -> Flux {
  return Flux(a.m + b.m, a.mom + b.mom, a.e + b.e, a.ei + b.ei, a.s + b.s, a.vn + b.vn, a.d + b.d);
}
fn flux_scale(f: Flux, w: f32) -> Flux {
  return Flux(f.m * w, f.mom * w, f.e * w, f.ei * w, f.s * w, f.vn * w, f.d * w);
}

// The block being worked on: its bin, and whether this pass only deposits the second half of
// its faster neighbors' fluxes (the block is half way through its step).
var<private> g_bin: u32 = 0u;
var<private> g_deposit: bool = false;
var<private> g_touched: bool = false;

// Share of this block's step for which the flux of the neighboring leaf `nb` is to be used
// now: all of it for a neighbor in step with the block, half for a faster one, and in a
// deposit pass nothing for the former.
fn share(nb: i32) -> f32 {
  if (!BINS_ON) { return 1.0; }
  if (blk_bin(nb) < g_bin) { return 0.5; }
  return select(1.0, 0.0, g_deposit);
}

// A spherical run: a refined neighbor's fine face (cell `a` of block `src` below or above it)
// against the mean of the two that cover cell l's face along `axis`: its area over theirs, so
// that the coarse face's flux is their area-weighted mean. A radial face's area goes as its
// dcos(theta), a polar face's as the integral of r dr over it (r_c dr, or r_c^2 sinh D).
fn sph_fine_face(slot: i32, l: vec3<i32>, axis: i32, src: i32, a: vec3<i32>) -> f32 {
  let q = blk_sph_grid(src);
  let fg = blk_coord(src) * BS + a;
  let cg = 2 * (blk_coord(slot) * BS + l);
  var w0 = 0.0;
  var w1 = 0.0;
  var hi = false;
  if (axis == 0) {
    w0 = sph_dcos(cg.y, q);
    w1 = sph_dcos(cg.y + 1, q);
    hi = fg.y == cg.y + 1;
  } else {
    let r0 = sph_cell_g(cg.x, 0, q).r;
    let r1 = sph_cell_g(cg.x + 1, 0, q).r;
    w0 = select(r0, r0 * r0, q.d != 0.0);
    w1 = select(r1, r1 * r1, q.d != 0.0);
    hi = fg.x == cg.x + 1;
  }
  return 2.0 * select(w0, w1, hi) / (w0 + w1);
}

// Replace the fluxes `own` through the cell's faces on its block's faces with its neighbors':
//  - a refined neighbor's: the average of the fine fluxes through the face, as the fine cells
//    computed them with their own cell size;
//  - with time bins, a same-level leaf that steps twice as often: its flux through the face,
//    as it computes it, for half of this block's step.
// Every face goes through one loop with a single call of flux_between. Compilers inline each
// call of it (with the Riemann solver), and unroll small loops: as calls for each face of each
// axis, in the deposit and the full pass, it was copied dozens of times, and the pass took
// 1.5 s to compile, two thirds of it these copies (measured on Chrome, D3D12, cold cache).
fn match_faces(own: array<Faces, 3>, slot: i32, l: vec3<i32>) -> array<Faces, 3> {
  var out = own;
  // Lower and upper face of x, then of y, then of z: entry 6 + k of the block table.
  let sides = select(6, 4, is2d());
  for (var k = 0; k < sides; k++) {
    let axis = k / 2;
    let hi = (k & 1) == 1;
    if (l[axis] != select(0, BS - 1, hi)) { continue; }
    let np = bt(slot, 6u + u32(k));
    if (np < 0) { continue; }
    // `upper`: `np` lies below the block, and the face is the upper face of its cells.
    let upper = !hi;
    var e = vec3<i32>(0);
    e[axis] = 1;
    // The faces to add up: the cells of block `o` below them from `a0` on, n1 x n2 of them
    // across the face (axes t1, t2), with their cell size dx, weighted by w.
    var o: vec3<i32>;
    var a0: vec3<i32>;
    var n1 = 1;
    var n2 = 1;
    var dx = g_dx;
    var w = 0.5;
    var src = np;   // the block that holds those cells
    if (!blk_leaf(np)) {
      // The refined neighbor's child that shares this cell's face.
      let h = vec3<i32>(l >= vec3<i32>(BS / 2));
      var bits = h;
      bits[axis] = i32(upper);
      if (is2d()) { bits.z = 0; }
      let ch = bt(np, 12u + u32(bits.x | (bits.y << 1u) | (bits.z << 2u)));
      let part = share(ch);
      if (part == 0.0) {
        if (hi) { out[axis].hi = flux_zero(); } else { out[axis].lo = flux_zero(); }
        continue;
      }
      o = blk_origin(ch);
      src = ch;
      // The fine cell on the lower side of the face: the last interior cell, or the ghost at -1.
      a0 = 2 * (l - (BS / 2) * h);
      // In 2D the faces are edges: two fine ones per coarse one, both in the plane.
      n1 = select(2, 1, is2d() && (axis + 1) % 3 == 2);
      n2 = select(2, 1, is2d() && (axis + 2) % 3 == 2);
      dx = 0.5 * g_dx;
      w = 1.0 / f32(n1 * n2);
      if (BINS_ON) { w *= part; }
    } else if (BINS_ON && np != slot && blk_bin(np) < g_bin) {
      o = blk_origin(np);
      a0 = l;
    } else {
      continue;
    }
    g_touched = true;
    a0[axis] = select(-1, BS - 1, upper);
    let t1 = (axis + 1) % 3;
    let t2 = (axis + 2) % 3;
    let dx_own = g_dx;
    g_dx = dx;
    var sum = flux_zero();
    for (var j = 0; j < n2; j++) {
      for (var i = 0; i < n1; i++) {
        var a = a0;
        a[t1] += i;
        a[t2] += j;
        g_fb_slot = src;
        g_fb_l = a;
        var fl = flux_between(atl(o, a), e, axis);
        // An axisymmetric run: a refined neighbor's faces across z lie at their own radii, and
        // their areas go as those (the coarse face's flux is their area-weighted mean).
        if (cylindrical() && axis == 1 && !blk_leaf(np)) {
          fl = flux_scale(fl, cyl_child_weight(blk_coord(slot).x * BS + l.x, blk_coord(src).x * BS + a.x));
        }
        // A spherical run: the fine faces' areas differ along both axes.
        if (spherical() && !blk_leaf(np)) {
          fl = flux_scale(fl, sph_fine_face(slot, l, axis, src, a));
        }
        sum = flux_add(sum, fl);
      }
    }
    g_dx = dx_own;
    if (hi) { out[axis].hi = flux_scale(sum, w); } else { out[axis].lo = flux_scale(sum, w); }
  }
  return out;
}

#include <debris>

// Debris energies (debris/debris.wgsl): the gas leaving through a face of the domain this
// step (the block's own step), binned by its orbital energy with the cell's velocity and
// potential at the face. A face of the domain has no neighbor to match fluxes with, so this is
// all that crosses it.
fn escape(slot: i32, l: vec3<i32>, c: vec3<i32>, v: vec3<f32>, fx: Faces, fy: Faces, fz: Faces) {
  let lo = vec3<f32>(fx.lo.m, fy.lo.m, fz.lo.m);
  let hi = vec3<f32>(fx.hi.m, fy.hi.m, fz.hi.m);
  // The passive scalar's flux, for problems that bin only the gas it marks.
  let los = vec3<f32>(fx.lo.s, fy.lo.s, fz.lo.s);
  let his = vec3<f32>(fx.hi.s, fy.hi.s, fz.hi.s);
  let xc = blk_pos(slot, l);
  let adt = select(g_dx * g_dx, g_dx, is2d()) * g_dt;
  let phi = textureLoad(G, c, 0).w;
  for (var a = 0; a < 3; a++) {
    if (a == 2 && is2d()) { break; }
    // Beyond a face of the domain the neighbor entry is -1 - (boundary type), at most -2.
    if (l[a] == 0 && bt(slot, 6u + u32(2 * a)) <= -2) {
      var x = xc;
      x[a] -= 0.5 * g_dx;
      deb_tally(x, v, phi, 0.5 * g_dt, -lo[a] * adt, -los[a] * adt);
    }
    if (l[a] == BS - 1 && bt(slot, 7u + u32(2 * a)) <= -2) {
      var x = xc;
      x[a] += 0.5 * g_dx;
      deb_tally(x, v, phi, 0.5 * g_dt, hi[a] * adt, his[a] * adt);
    }
  }
}

// Validation hook: when set, the pass writes the face mass fluxes instead of the new state.
override DEBUG_FLUX: bool = false;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  // The whole workgroup computes the faces of its cells, before any invocation can leave.
  if (TILE) { tile_fluxes(wid, li); }
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let slot = inv.slot;
  let l = inv.l;
  g_dx = cell_dx(blk_dx(slot), clock.ah);
  // Comoving coordinates: cooling's rate at the proper density (hydro/cooling.wgsl).
  if (cosmo_on()) { g_a3 = clock.ah * clock.ah * clock.ah; }
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
      if (!g_deposit) { store_wall_none(c); }
      return;
    }
  }
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);

  let w0 = load_q(c);
  let p0 = load_pred(c);
  let none = Faces(flux_zero(), flux_zero());
  // Half way through the block's step (deposit), only what its faster neighbors carry across
  // the shared faces during their second step: the cell's own fluxes are left at zero.
  var own = array<Faces, 3>(none, none, none);
  if (!g_deposit) {
    var ox: Faces;
    var oy: Faces;
    var oz = none;
    if (TILE) {
      // This cell's box (one of four in 2D, one layer each) and its place in it.
      let b = select(0u, lid.z, TILE_2D);
      let f = select(vec3<i32>(lid), vec3<i32>(i32(lid.x), i32(lid.y), 0), TILE_2D);
      ox = tile_faces(b, 0, f, ex);
      oy = tile_faces(b, 1, f, ey);
      if (!is2d()) { oz = tile_faces(b, 2, f, ez); }
    } else {
      ox = axis_fluxes(c, ex, 0, w0, p0);
      oy = axis_fluxes(c, ey, 1, w0, p0);
      if (!is2d()) { oz = axis_fluxes(c, ez, 2, w0, p0); }
    }
    // Planes of mirror symmetry: nothing crosses them (hydro/flux.wgsl mirror_flux).
    let gm = blk_coord(slot) * BS + l;
    if (gm.x == 0 && sym_lo(0)) { ox.lo = mirror_flux(ox.lo, 0); }
    if (gm.y == 0 && sym_lo(1)) { oy.lo = mirror_flux(oy.lo, 1); }
    if (gm.z == 0 && sym_lo(2) && !is2d()) { oz.lo = mirror_flux(oz.lo, 2); }
    // With species, their pass needs the mass fluxes as this cell computed them.
    store_mass_flux(slot, l, c, ox, oy, oz);
    // Forces on bodies: their wall faces' fluxes (amr/walls_on.wgsl).
    store_wall_forces(slot, l, c, ox, oy, oz);
    store_face_velocities(c, vec3<f32>(ox.lo.vn, oy.lo.vn, oz.lo.vn), vec3<f32>(ox.hi.vn, oy.hi.vn, oz.hi.vn));
    own = array<Faces, 3>(ox, oy, oz);
  }
  let matched = match_faces(own, slot, l);
  var fx = matched[0];
  var fy = matched[1];
  var fz = matched[2];
  // Planes of mirror symmetry: nothing crosses them (hydro/flux.wgsl mirror_flux).
  let gl = blk_coord(slot) * BS + l;
  if (gl.x == 0 && sym_lo(0)) { fx.lo = mirror_flux(fx.lo, 0); }
  if (gl.y == 0 && sym_lo(1)) { fy.lo = mirror_flux(fy.lo, 1); }
  if (gl.z == 0 && sym_lo(2) && !is2d()) { fz.lo = mirror_flux(fz.lo, 2); }
  // An axisymmetric run: the radial faces weighed by their areas, and the radial momentum's
  // geometric source (hydro/update.wgsl), at the cell's global radial index on its level, before the
  // flux-form gravitational work reads the mass fluxes.
  var src_r = 0.0;
  if (cylindrical() && !g_deposit && !DEBUG_FLUX) {
    let w = cyl_weights(gl.x);
    let ps = 0.5 * (face_pressure(fx.lo, 0) + face_pressure(fx.hi, 0));
    src_r = (ps + p0.a.x * p0.a.w * p0.a.w) / (f32(gl.x) + 0.5);
    fx.lo = cyl_face(fx.lo, w.x);
    fx.hi = cyl_face(fx.hi, w.y);
  }
  // A spherical run: the faces weighed by their areas and lever arms, and the r and theta
  // momenta's geometric sources (hydro/update.wgsl), at the cell's global indices on its level.
  // The sources take the pressures of the cell's own faces: with per-block timesteps a face
  // shared with a faster neighbor carries only half of the step's flux now, the rest in the
  // deposit pass (which weighs its fluxes but adds no source).
  var src_t = 0.0;
  if (spherical() && !DEBUG_FLUX) {
    let g = blk_sph(slot, l);
    if (!g_deposit) {
      let v = p0.a.yzw;
      let rho = p0.a.x;
      var sx = fx;
      var sy = fy;
      if (BINS_ON) {
        sx = own[0];
        sy = own[1];
      }
      let pr = vec2<f32>(face_pressure(sx.lo, 0), face_pressure(sx.hi, 0));
      let pt = vec2<f32>(face_pressure(sy.lo, 1), face_pressure(sy.hi, 1));
      src_r = (g.h.x - g.wr.x) * pr.x + (g.wr.y - g.h.x) * pr.y + rho * (v.y * v.y + v.z * v.z) * g.ir;
      src_t = (g.h.y - g.wt.x) * pt.x + (g.wt.y - g.h.y) * pt.y + rho * v.z * v.z * (g.wt.y - g.wt.x);
    }
    fx.lo = sph_face(fx.lo, g.wr.x, vec3<f32>(1.0, g.lr.x, g.lr.x));
    fx.hi = sph_face(fx.hi, g.wr.y, vec3<f32>(1.0, g.lr.y, g.lr.y));
    fy.lo = sph_face(fy.lo, g.wt.x, vec3<f32>(1.0, 1.0, g.lt.x));
    fy.hi = sph_face(fy.hi, g.wt.y, vec3<f32>(1.0, 1.0, g.lt.y));
  }
  if (g_deposit) {
    // A cell on no face shared with a faster neighbor is copied as it is.
    if (!g_touched) {
      textureStore(O1, c, textureLoad(U1, c, 0));
      textureStore(O2, c, textureLoad(U2, c, 0));
      return;
    }
  } else if (DEBRIS) {
    escape(slot, l, c, w0.a.yzw, fx, fy, fz);
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
  // The point masses' pull and the frame's: kicked with the rest, its energy always in kick
  // form (gravity/ext.wgsl).
  let gx = ext_accel(blk_pos(slot, l));
  // (A rotating frame's centrifugal force is part of gx, with or without point masses.)
  // (With implicit pressure a uniform field's kick is the pressure passes': amr/pressure/faces.wgsl.)
  let gravity = !PSPLIT && (self_gravity || any(P.gext.xyz != vec3<f32>(0.0)) || ext_on());
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

  var u1 = textureLoad(U1, c, 0);
  var u2 = textureLoad(U2, c, 0);
  // A non-inertial frame's velocity-dependent forces, the first half of the block's step of
  // them (gravity/frame_off.wgsl). A deposit applies none, as it applies no kick.
  if (FRAME_ON && !g_deposit) {
    let fs = frame_open(u1, u2, 0.5 * g_dt);
    u1 = fs.u1;
    u2 = fs.u2;
  }
  // From here on the equation of state is this cell's.
  comp_use(comp_tex(c));
  var kick = vec3<f32>(0.0);
  var ekick = 0.0;
  if (gravity && !g_deposit) {
    let kx = 0.5 * g_dt * u1.x * gx;
    kick = 0.5 * g_dt * u1.x * g + kx;
    let ke = select(kx, kick, !flux_work);
    ekick = dot(ke, u1.yzw + 0.5 * kick) / max(u1.x, P.rho_floor);
  }

  // Turbulence driving (hydro/drive.wgsl): a whole step's kick, in kick form.
  if (drive_on() && !g_deposit) {
    let kd = drive_kick(kick, ekick, blk_pos(slot, l), g_dt, u1.x, u1.yzw);
    kick = kd.xyz;
    ekick = kd.w;
  }

  var net = net_inflow_heated(fx, fy, fz, w0.a.yzw);
  net.mom.x += src_r;
  if (spherical()) { net.mom.y += src_t; }
  var nw = advance(u1, u2, net, dw, p0.b.x, kick, ekick);
  if (g_deposit) {
    // The kicks, sources and cooling of this step were applied when it started.
    textureStore(O1, c, nw.u1);
    textureStore(O2, c, nw.u2);
    return;
  }

  if (gravity && !self_gravity) {
    // Second half kick, with the updated density.
    let rho = nw.u1.x;
    var k2 = 0.5 * g_dt * rho * g;
    var E = nw.u2.x;
    // A rotating frame's centrifugal force (without self-gravity or point masses, which would
    // close the step in amr/correct.wgsl): its energy in kick form.
    if (ext_on()) {
      let kx2 = 0.5 * g_dt * rho * gx;
      k2 += kx2;
      if (flux_work) { E += dot(kx2, nw.u1.yzw + 0.5 * k2) / rho; }
    }
    let mom = nw.u1.yzw + k2;
    var eint = nw.u2.y;
    if (flux_work) {
      g_kphi = swirl_ke(rho, mom);
      let en = dual_energy(rho, 0.5 * dot(mom, mom) / rho, E, eint);
      E = en.x;
      eint = en.y;
    } else {
      E += dot(k2, nw.u1.yzw + 0.5 * k2) / rho;
    }
    nw = NewState(vec4<f32>(rho, mom), vec4<f32>(E, eint, nw.u2.z, 0.0));
  }
  if (has_sources()) { nw = apply_sources(nw, box_frac(blk_pos(slot, l)), g_dt); }
  nw = apply_damping(nw, box_frac(blk_pos(slot, l)), g_dt);
  nw = apply_cooling(nw, g_dt);
  // The second half of the frame's forces, where no correction pass follows: last, as the
  // correction pass applies it after all of the update on the uniform grid.
  if (FRAME_ON && !self_gravity) {
    // (Implicit pressure, the liquid: the centrifugal force's whole kick too, before the solve.)
    if (PSPLIT && P.rot.w != 0.0) { nw = NewState(vec4<f32>(nw.u1.x, nw.u1.yzw + g_dt * nw.u1.x * rot_accel(blk_pos(slot, l))), nw.u2); }
    let fs = frame_close(nw.u1, nw.u2, 0.5 * g_dt);
    nw = NewState(fs.u1, fs.u2);
  }
  textureStore(O1, c, nw.u1);
  textureStore(O2, c, nw.u2);
}

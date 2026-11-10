// Unsplit MUSCL-Hancock update on the block grid: the same gather as hydro/update.wgsl, over
// the cells of leaf blocks, with ghost cells read straight from the block's own storage.
//
// Flux matching at refinement boundaries. Where a leaf's face neighbour is refined, the coarse
// cell does not use its own flux through that face. It recomputes the fine fluxes instead,
// from the fine blocks' stored states with the same function the fine cells use, and takes
// their average. Both sides then agree on what crossed the face, so mass, momentum and energy
// are conserved across levels without storing any fluxes. (This relies on every level taking
// the same timestep.)
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"
#include "hydro/fluff.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

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
  return Q(w1, vec4<f32>(w2.x, w2.z, w2.y, 0.0));
}
fn load_pred(c: vec3<i32>) -> Q {
  return Q(textureLoad(PA, c, 0), textureLoad(PB, c, 0));
}

#include "hydro/flux.wgsl"
#include "hydro/advance.wgsl"

fn flux_add(a: Flux, b: Flux) -> Flux {
  return Flux(a.m + b.m, a.mom + b.mom, a.e + b.e, a.ei + b.ei, a.s + b.s, a.vn + b.vn);
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
  let w = 1.0 / f32(n1 * n2);
  return Flux(sum.m * w, sum.mom * w, sum.e * w, sum.ei * w, sum.s * w, sum.vn * w);
}

// Replace the fluxes through faces shared with a refined neighbour.
fn match_faces(fl: Faces, slot: i32, axis: i32, l: vec3<i32>) -> Faces {
  var out = fl;
  if (l[axis] == 0) {
    let np = bt(slot, 6u + u32(2 * axis));
    if (np >= 0 && !blk_leaf(np)) { out.lo = fine_flux(np, axis, true, l); }
  }
  if (l[axis] == BS - 1) {
    let np = bt(slot, 7u + u32(2 * axis));
    if (np >= 0 && !blk_leaf(np)) { out.hi = fine_flux(np, axis, false, l); }
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
  let c = atl(blk_origin(slot), l);
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);

  let w0 = load_q(c);
  let p0 = load_pred(c);
  let fx = match_faces(axis_fluxes(c, ex, 0, w0, p0), slot, 0, l);
  let fy = match_faces(axis_fluxes(c, ey, 1, w0, p0), slot, 1, l);
  var fz = Faces(flux_zero(), flux_zero());
  if (!is2d()) { fz = match_faces(axis_fluxes(c, ez, 2, w0, p0), slot, 2, l); }

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
  if (gravity) {
    kick = 0.5 * clock.dt * u1.x * g;
    if (!flux_work) { ekick = dot(kick, u1.yzw + 0.5 * kick) / max(u1.x, P.rho_floor); }
  }

  var nw = advance(u1, u2, net_inflow(fx, fy, fz), dw, p0.b.x, kick, ekick);

  if (gravity && !self_gravity) {
    // Second half kick, with the updated density.
    let rho = nw.u1.x;
    let k2 = 0.5 * clock.dt * rho * g;
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
  textureStore(O1, c, nw.u1);
  textureStore(O2, c, nw.u2);
}

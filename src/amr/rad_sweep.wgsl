// One radiation sweep on the block grid: radiation/sweep.wgsl's, for every leaf cell, from the
// radiation and opacity with ghosts that amr/rad_assemble.wgsl has just built (RA, RK). All
// blocks sweep over the same step, clock.dt / P.rad2.x.
//
// Through a face shared with a refined neighbor, a coarse cell takes the mean of the fine cells'
// fluxes, each computed in the fine block from the fine block's cells and ghosts as the fine
// cell itself computes it: what leaves one level enters the other. Each cell multiplies the
// divergence by its own reduction factor r (its optical depth is its own cell's), so across
// such a face E / r is kept exactly where both sides have the same r (1 in opaque gas) and E
// itself otherwise.
#include "hydro/common.wgsl"
#include <eos>
#include <opacity>
#include "radiation/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;    // the gas: (rho, rho v), fixed through the sweeps
@group(0) @binding(3) var U2: texture_3d<f32>;    // (E, rho e, rho s, mark)
@group(0) @binding(4) var RA: texture_3d<f32>;    // radiation (E, F) before the sweep, with ghosts
@group(0) @binding(5) var GI: texture_3d<f32>;    // the gas's side: (rho e, momentum gained)
@group(0) @binding(6) var RO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var GO: texture_storage_3d<rgba32float, write>;
struct SweepInfo { first: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(8) var<uniform> SW: SweepInfo;
@group(0) @binding(9) var RK: texture_3d<f32>;    // kappa_R rho, with ghosts

#include "radiation/face_flux.wgsl"
#include "radiation/exchange.wgsl"

fn rc(o: vec3<i32>, l: vec3<i32>) -> RCell {
  let c = atl(o, l);
  let R = textureLoad(RA, c, 0);
  return RCell(R.x, R.yzw, textureLoad(RK, c, 0).x);
}

// The flux through the lower face along `a` of cell l of block `slot` (l may be the ghost at BS).
fn own_flux(slot: i32, l: vec3<i32>, a: i32) -> RFlux {
  let o = blk_origin(slot);
  var e = vec3<i32>(0);
  e[a] = 1;
  return face_flux(a, blk_dx(slot), rc(o, l - 2 * e), rc(o, l - e), rc(o, l), rc(o, l + e));
}

// Mean of the fine fluxes through the face coarse cell l shares with the refined neighbor `np`
// (amr/species.wgsl fine_flux): `upper` says the fine blocks lie on np's upper side.
fn fine_flux(np: i32, a: i32, upper: bool, l: vec3<i32>) -> RFlux {
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  var bits = h;
  bits[a] = i32(upper);
  if (is2d()) { bits.z = 0; }
  let ch = bt(np, 12u + u32(bits.x | (bits.y << 1u) | (bits.z << 2u)));
  var f = 2 * (l - (BS / 2) * h);
  f[a] = select(0, BS, upper);
  let t1 = (a + 1) % 3;
  let t2 = (a + 2) % 3;
  let n1 = select(2, 1, is2d() && t1 == 2);
  let n2 = select(2, 1, is2d() && t2 == 2);
  var sum = RFlux(0.0, vec3<f32>(0.0));
  for (var j = 0; j < loop_count(n2); j++) {
    for (var i = 0; i < n1; i++) {
      var q = f;
      q[t1] += i;
      q[t2] += j;
      let x = own_flux(ch, q, a);
      sum.E += x.E;
      sum.F += x.F;
    }
  }
  let w = 1.0 / f32(n1 * n2);
  return RFlux(sum.E * w, sum.F * w);
}

fn gas_eint(c: vec3<i32>) -> f32 {
  if (SW.first != 0u) { return textureLoad(U2, c, 0).y; }
  return textureLoad(GI, c, 0).x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let slot = inv.slot;
  let l = inv.l;
  let c = atl(blk_origin(slot), l);
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  let R = textureLoad(RA, c, 0);
  let gained = select(textureLoad(GI, c, 0).yzw, vec3<f32>(0.0), SW.first != 0u);
  // Solid and fixed cells keep their radiation (a fixed cell is an inflow), and their gas.
  if (has_solids() && abs(u2.w) > 0.5) {
    textureStore(RO, c, R);
    textureStore(GO, c, vec4<f32>(gas_eint(c), gained));
    return;
  }
  let dx = blk_dx(slot);
  g_dx = dx;
  let dt = clock.dt / P.rad2.x;
  var divE = 0.0;
  var divF = vec3<f32>(0.0);
  let nax = select(3, 2, is2d());
  for (var a = 0; a < loop_count(nax); a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let nlo = select(-1, bt(slot, 6u + u32(2 * a)), l[a] == 0);
    let nhi = select(-1, bt(slot, 7u + u32(2 * a)), l[a] == BS - 1);
    var lo: RFlux;
    var hi: RFlux;
    if (nlo >= 0 && !blk_leaf(nlo)) { lo = fine_flux(nlo, a, true, l); } else { lo = own_flux(slot, l, a); }
    if (nhi >= 0 && !blk_leaf(nhi)) { hi = fine_flux(nhi, a, false, l); } else { hi = own_flux(slot, l + e, a); }
    divE += hi.E - lo.E;
    divF += hi.F - lo.F;
  }
  let rho = max(u1.x, P.rho_floor);
  comp_use(comp_tex(c));
  let eint = gas_eint(c);
  let T = eos_temperature(rho, eint);
  let kR = kappa_rosseland(rho, T);
  let kP = kappa_planck(rho, T);
  let r = rad_reduction(kR * rho * dx);
  let Es = max(R.x - r * dt * divE / dx, 0.0);
  let Fs = R.yzw - r * dt * divF / dx;
  let x = rad_exchange(rho, eint, Es, Fs, kP, kR, r, dt);
  // The work of the radiation's force on the gas, as radiation/sweep.wgsl.
  let v = (u1.yzw + gained + 0.5 * x.dmom) / rho;
  var E = max(x.E - r * dot(v, x.dmom), 0.0);
  var F = x.F;
  let fm = length(F);
  if (fm > rad_c() * E) { F *= rad_c() * E / fm; }
  textureStore(RO, c, vec4<f32>(E, F));
  textureStore(GO, c, vec4<f32>(x.eint, gained + x.dmom));
}

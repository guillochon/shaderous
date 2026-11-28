// Radiation carried with the gas on the block grid: radiation/advect.wgsl's, for every leaf cell,
// with the face mass fluxes the update recorded (amr/species_on.wgsl), the primitives the step
// started from (W1, with ghosts) and the radiation with ghosts (RA, amr/rad_assemble.wgsl).
//
// Through a face shared with a refined neighbor, a coarse cell takes the mean of what the fine
// faces carry, each computed in the fine block as the fine cell computes it, from the fine
// blocks' recorded mass fluxes: as amr/species.wgsl does for the species.
#include "hydro/common.wgsl"
#include <eos>
#include <opacity>
#include "radiation/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "amr/species_index.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var W1: texture_3d<f32>;    // primitives as the step started: (density, negative if solid; velocity)
@group(0) @binding(3) var U2: texture_3d<f32>;    // the state as the step started: marks, rho e
@group(0) @binding(5) var RA: texture_3d<f32>;
@group(0) @binding(6) var RO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> MF: array<f32>;
#include <mf_load>

#include "radiation/carry.wgsl"

fn ac(o: vec3<i32>, l: vec3<i32>, a: i32) -> ACell {
  let c = atl(o, l);
  let w = textureLoad(W1, c, 0);
  let R = textureLoad(RA, c, 0);
  let rho = max(abs(w.x), P.rho_floor);
  var x: ACell;
  x.q = R / rho;
  x.rho = rho;
  x.v = w.yzw;
  x.prow = R.x * rad_eddington_row(rad_reduced(R.x, R.yzw), a);
  return x;
}

// What the lower face along `a` of cell l of block `slot` carries toward +a (l may be the ghost at BS).
fn own_carry(slot: i32, l: vec3<i32>, a: i32) -> vec4<f32> {
  let o = blk_origin(slot);
  var e = vec3<i32>(0);
  e[a] = 1;
  let dtdx = clock.dt / blk_dx(slot);
  return face_carry(a, mf_load(mf_index(slot, l, a)), dtdx, ac(o, l - 2 * e, a), ac(o, l - e, a), ac(o, l, a), ac(o, l + e, a));
}

fn fine_carry(np: i32, a: i32, upper: bool, l: vec3<i32>) -> vec4<f32> {
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  var bits = h;
  bits[a] = i32(upper);
  if (is2d()) { bits.z = 0; }
  let ch = bt(np, 12u + u32(bits.x | (bits.y << 1u) | (bits.z << 2u)));
  var f = 2 * (l - (BS / 2) * h);
  f[a] = select(0, BS, upper);
  let t1 = (a + 1) % 3;
  let t2 = (a + 2) % 3;
  let n1 = select(2, 1, (is2d() && t1 == 2) || (is1d() && t1 == 1));
  let n2 = select(2, 1, (is2d() && t2 == 2) || (is1d() && t2 == 1));
  var sum = vec4<f32>(0.0);
  for (var j = 0; j < loop_count(n2); j++) {
    for (var i = 0; i < n1; i++) {
      var q = f;
      q[t1] += i;
      q[t2] += j;
      sum += own_carry(ch, q, a);
    }
  }
  return sum / f32(n1 * n2);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let slot = inv.slot;
  let l = inv.l;
  let c = atl(blk_origin(slot), l);
  let R = textureLoad(RA, c, 0);
  let u2 = textureLoad(U2, c, 0);
  if (has_solids() && abs(u2.w) > 0.5) {
    textureStore(RO, c, R);
    return;
  }
  var acc = vec4<f32>(0.0);
  let nax = blk_axes();
  for (var a = 0; a < loop_count(nax); a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let nlo = select(-1, bt(slot, 6u + u32(2 * a)), l[a] == 0);
    let nhi = select(-1, bt(slot, 7u + u32(2 * a)), l[a] == BS - 1);
    if (nlo >= 0 && !blk_leaf(nlo)) { acc += fine_carry(nlo, a, true, l); } else { acc += own_carry(slot, l, a); }
    if (nhi >= 0 && !blk_leaf(nhi)) { acc -= fine_carry(nhi, a, false, l); } else { acc -= own_carry(slot, l + e, a); }
  }
  let dx = blk_dx(slot);
  let rho = max(abs(textureLoad(W1, c, 0).x), P.rho_floor);
  comp_use(comp_tex(c));
  let r = rad_reduction(kappa_rosseland(rho, eos_temperature(rho, u2.y)) * rho * dx);
  var out = R + (r * clock.dt / dx) * acc;
  out.x = max(out.x, 0.0);
  let fm = length(out.yzw);
  if (fm > rad_c() * out.x) { out = vec4<f32>(out.x, out.yzw * (rad_c() * out.x / fm)); }
  textureStore(RO, c, out);
}

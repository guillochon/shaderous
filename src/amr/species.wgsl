// Species on the block grid, third pass: move every leaf cell's rho X with the mass fluxes the
// update recorded (amr/species_on.wgsl), reconstructed as on the uniform grid
// (hydro/species.wgsl): the upwind cell's fractions half a step ahead (SH) plus half its
// limited slope (from SX) toward the face, limited as on the uniform grid.
//
// Through a face shared with a refined neighbor, a coarse cell takes the average of the fine
// cells' species fluxes, each reconstructed in the fine block as the fine cell itself did: the
// same matching amr/update.wgsl does for the mass, so species are conserved across levels as
// the mass is. One timestep for all blocks (per-block timesteps are off with species). Where
// the composition sets the equation of state (species/mixture.ts), each cell's new moments are
// written beside its rho X.
#include "amr/species_common.wgsl"
#include "amr/species_index.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(3) var U2: texture_3d<f32>;    // marks of solid and fixed cells
@group(0) @binding(4) var N1: texture_3d<f32>;    // state after the update
@group(0) @binding(5) var S: texture_3d<f32>;     // rho X at the start of the step
@group(0) @binding(6) var SO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> MF: array<f32>;
@group(0) @binding(8) var SX: texture_3d<f32>;
@group(0) @binding(9) var SH: texture_3d<f32>;

fn load_sx(c: vec3<i32>) -> array<vec4<f32>, NG> {
  var x: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { x[g] = textureLoad(SX, sp(c, g), 0); }
  return x;
}

// Species flux F X through the face between local cells a and a + e of block `slot` (either
// may be a first-layer ghost), F being the mass flux through it.
fn face_flux(slot: i32, a: vec3<i32>, axis: i32, F: f32) -> array<vec4<f32>, NG> {
  var e = vec3<i32>(0);
  e[axis] = 1;
  let u = atl(blk_origin(slot), select(a + e, a, F >= 0.0));
  let toward = select(-0.5, 0.5, F >= 0.0);
  let s = slopes(load_sx(u - e), load_sx(u), load_sx(u + e));
  var xf: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { xf[g] = textureLoad(SH, sp(u, g), 0) + toward * s[g]; }
  // Toward first order as far as the hydro is (comp_first_order): the upwind cell as it is.
  if (COMP) {
    let w = comp_first_order(u);
    let x0 = load_sx(u);
    for (var g = 0u; g < NG; g++) { xf[g] = mix(xf[g], x0[g], w); }
  }
  xf = face_fractions(xf);
  for (var g = 0u; g < NG; g++) { xf[g] *= F; }
  return xf;
}

// The cell's own flux through its lower (side 0) or upper (side 1) face along an axis.
fn own_flux(slot: i32, l: vec3<i32>, axis: i32, side: i32) -> array<vec4<f32>, NG> {
  var e = vec3<i32>(0);
  e[axis] = 1;
  let F = MF[mf_index(slot, l + side * e, axis)];
  return face_flux(slot, l + (side - 1) * e, axis, F);
}

// Average of the fine species fluxes through the face coarse cell l shares with the refined
// neighbor `np` (amr/update.wgsl match_faces): `upper` says the fine blocks lie on np's upper side.
fn fine_flux(np: i32, axis: i32, upper: bool, l: vec3<i32>) -> array<vec4<f32>, NG> {
  let h = vec3<i32>(l >= vec3<i32>(BS / 2));
  var bits = h;
  bits[axis] = i32(upper);
  if (is2d()) { bits.z = 0; }
  let ch = bt(np, 12u + u32(bits.x | (bits.y << 1u) | (bits.z << 2u)));
  var e = vec3<i32>(0);
  e[axis] = 1;
  var f = 2 * (l - (BS / 2) * h);
  f[axis] = select(-1, BS - 1, upper);
  let t1 = (axis + 1) % 3;
  let t2 = (axis + 2) % 3;
  let n1 = select(2, 1, is2d() && t1 == 2);
  let n2 = select(2, 1, is2d() && t2 == 2);
  var sum: array<vec4<f32>, NG>;
  for (var j = 0; j < n2; j++) {
    for (var i = 0; i < n1; i++) {
      var a = f;
      a[t1] += i;
      a[t2] += j;
      let F = MF[mf_index(ch, a + e, axis)];
      let ff = face_flux(ch, a, axis, F);
      for (var g = 0u; g < NG; g++) { sum[g] += ff[g]; }
    }
  }
  let w = 1.0 / f32(n1 * n2);
  for (var g = 0u; g < NG; g++) { sum[g] *= w; }
  return sum;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  let slot = inv.slot;
  let l = inv.l;
  let c = atl(blk_origin(slot), l);
  if (has_solids() && abs(textureLoad(U2, c, 0).w) > 0.5) {
    var u: array<vec4<f32>, NG>;
    for (var g = 0u; g < NG; g++) {
      u[g] = textureLoad(S, sp(c, g), 0);
      textureStore(SO, sp(c, g), u[g]);
    }
    if (COMP) { comp_store(c, comp_moments(u)); }
    return;
  }
  var acc: array<vec4<f32>, NG>;
  let nax = select(3, 2, is2d());
  for (var a = 0; a < nax; a++) {
    var lo: array<vec4<f32>, NG>;
    var hi: array<vec4<f32>, NG>;
    let nlo = select(-1, bt(slot, 6u + u32(2 * a)), l[a] == 0);
    let nhi = select(-1, bt(slot, 7u + u32(2 * a)), l[a] == BS - 1);
    if (nlo >= 0 && !blk_leaf(nlo)) { lo = fine_flux(nlo, a, true, l); } else { lo = own_flux(slot, l, a, 0); }
    if (nhi >= 0 && !blk_leaf(nhi)) { hi = fine_flux(nhi, a, false, l); } else { hi = own_flux(slot, l, a, 1); }
    for (var g = 0u; g < NG; g++) { acc[g] += lo[g] - hi[g]; }
  }
  let dtdx = clock.dt / blk_dx(slot);
  var out: array<vec4<f32>, NG>;
  var sum = 0.0;
  for (var g = 0u; g < NG; g++) {
    out[g] = max(textureLoad(S, sp(c, g), 0) + dtdx * acc[g], vec4<f32>(0.0));
    sum += dot(out[g], vec4<f32>(1.0));
  }
  let scale = textureLoad(N1, c, 0).x / max(sum, 1e-30);
  for (var g = 0u; g < NG; g++) { textureStore(SO, sp(c, g), out[g] * scale); }
  if (COMP) { comp_store(c, comp_moments(out)); }
}

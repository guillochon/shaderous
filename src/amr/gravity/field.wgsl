// Gravity field atlas, written once per step after the Poisson solve, for every block:
//   xyz = acceleration: self-gravity from the new potential, plus the perturber's pull minus
//         the frame's own acceleration when a perturber is present (interior cells)
//   w   = potential extrapolated half a step ahead, for the update's time-centred energy
//         source (interior cells and the first ghost layer)
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include "gravity/bh.wgsl"

@group(0) @binding(1) var PHI: texture_3d<f32>;
@group(0) @binding(2) var PHI_OLD: texture_3d<f32>;
@group(0) @binding(3) var GOUT: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var<storage, read> BH: Perturber;
@group(0) @binding(5) var<storage, read> clock: Clock;

fn phi(c: vec3<i32>) -> f32 {
  return textureLoad(PHI, c, 0).x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_padded(gid, 1);
  if (!inv.ok) { return; }
  let l = inv.l;
  let out = vec3<i32>(l < vec3<i32>(0)) + vec3<i32>(l >= vec3<i32>(BS));
  let nout = out.x + out.y + out.z;
  if (nout > 1) { return; }
  let c = blk_origin(inv.slot) + l;
  var g = vec3<f32>(0.0);
  if (nout == 0) {
    let h = blk_dx(inv.slot);
    g = -vec3<f32>(
      phi(c + vec3<i32>(1, 0, 0)) - phi(c - vec3<i32>(1, 0, 0)),
      phi(c + vec3<i32>(0, 1, 0)) - phi(c - vec3<i32>(0, 1, 0)),
      phi(c + vec3<i32>(0, 0, 1)) - phi(c - vec3<i32>(0, 0, 1)),
    ) * (0.5 / h);
  }
  var ph = 1.5 * phi(c) - 0.5 * textureLoad(PHI_OLD, c, 0).x;
  if (P.bh.z != 0.0) {
    let x = blk_pos(inv.slot, l);
    g += bh_accel(x, BH.pos.xyz) - BH.frame.xyz;
    // The frame's acceleration a acts as a uniform field -a, whose potential is a . x.
    ph += bh_phi(x, BH.pos.xyz + 0.5 * clock.dt * BH.vel.xyz) + dot(BH.frame.xyz, x);
  }
  textureStore(GOUT, c, vec4<f32>(g, ph));
}

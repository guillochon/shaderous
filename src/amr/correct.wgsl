// Second half of the gravity kick on the block grid, for leaf cells, with the acceleration of
// the updated state (see gravity/correct.wgsl). With per-block timesteps it closes the step a
// block has just finished, whatever its length.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "hydro/fluff.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <bins>
#include "amr/blk_dt.wgsl"
#include <frame>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var N1: texture_3d<f32>;       // state after the hydro update
@group(0) @binding(3) var N2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;        // gravity field of that state
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> BH: Masses;
// With a magnetic field (MHD): the updated faces, whose B^2 / 2 is in E and stays out of what
// this pass changes (gravity/correct.wgsl). A placeholder texel without one.
@group(0) @binding(8) var BF: texture_3d<f32>;
override MHD: bool = false;

#include "gravity/bh.wgsl"
#include "gravity/ext.wgsl"
#include "gravity/sink.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let inv = inv_interior(gid);
  if (!inv.ok) { return; }
  set_proper_dx(blk_dx(inv.slot), clock.a1);
  if (cosmo_on()) { g_a3 = clock.a1 * clock.a1 * clock.a1; }
  g_dt = blk_dt(inv.slot);
  let c = atl(blk_origin(inv.slot), inv.l);
  comp_use(comp_tex(c));
  var u1 = textureLoad(N1, c, 0);
  var u2 = textureLoad(N2, c, 0);
  var emag = 0.0;
  if (MHD) {
    // The block's own faces: texel BS holds its upper ones.
    let lo = textureLoad(BF, c, 0).xyz;
    var hi = lo;
    for (var a = 0; a < 3; a++) {
      if (a == 2 && is2d()) { break; }
      var e = vec3<i32>(0);
      e[a] = 1;
      hi[a] = textureLoad(BF, c + e, 0)[a];
    }
    let B = 0.5 * (lo + hi);
    emag = 0.5 * dot(B, B);
  }
  if (abs(u2.w) > 0.5) {
    // Solid and fixed cells keep their state.
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    return;
  }
  if (P.bh.z != 0.0) {
    // The point masses' sinks take their gas before the kick, from the state the reduction saw.
    let left = apply_sinks(u1, u2 - vec4<f32>(emag, 0.0, 0.0, 0.0), blk_pos(inv.slot, inv.l), g_dt);
    u1 = left.u1;
    u2 = left.u2 + vec4<f32>(emag, 0.0, 0.0, 0.0);
  }

  let kx = 0.5 * g_dt * u1.x * ext_accel_wb(blk_pos(inv.slot, inv.l), blk_dx(inv.slot));
  let kick = 0.5 * g_dt * u1.x * (P.gext.xyz + textureLoad(G, c, 0).xyz) + kx;
  var mom = u1.yzw + kick;
  var E = u2.x;
  var eint = u2.y;
  if ((P.flags & 1u) == 0u) {
    E += dot(kick, u1.yzw + 0.5 * kick) / u1.x;
  } else {
    // The point masses' and the frame's part of the kick adds its own energy (gravity/ext.wgsl);
    // self-gravity's work for the whole step is already in E.
    E += dot(kx, u1.yzw + 0.5 * kick) / u1.x;
    g_kphi = swirl_ke(u1.x, mom);
    let en = dual_energy(u1.x, 0.5 * dot(mom, mom) / u1.x + emag, E, eint);
    E = en.x;
    eint = en.y;
  }
  // The second half step of a non-inertial frame's velocity-dependent forces.
  if (FRAME_ON) {
    let fs = frame_close(vec4<f32>(u1.x, mom), vec4<f32>(E, eint, u2.zw), 0.5 * g_dt);
    mom = fs.u1.yzw;
    E = fs.u2.x;
    eint = fs.u2.y;
  }
  let tame = tame_fluff(u1.x, mom, eint);
  if (tame.changed) {
    mom = tame.mom;
    eint = tame.eint;
    E = 0.5 * dot(mom, mom) / u1.x + emag + eint;
  }
  textureStore(O1, c, vec4<f32>(u1.x, mom));
  textureStore(O2, c, vec4<f32>(E, eint, u2.z, u2.w));
}

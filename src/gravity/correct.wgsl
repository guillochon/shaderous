// Second half of the gravity kick, applied after the hydro update with the acceleration from
// the potential of the updated density. Together with the half kick in hydro/update.wgsl the
// momentum source is dt/2 (rho^n g^n + rho^{n+1} g^{n+1}), i.e. time-centred.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "hydro/fluff.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var N1: texture_3d<f32>;       // state after the hydro update
@group(0) @binding(3) var N2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;        // gravity field of that state
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;

#include "gravity/gfield.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  g_dx = P.dx;
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let u1 = textureLoad(N1, c, 0);
  let u2 = textureLoad(N2, c, 0);
  if (abs(u2.w) > 0.5) {
    // Solid and fixed cells keep their state.
    textureStore(O1, c, u1);
    textureStore(O2, c, u2);
    return;
  }

  let kick = 0.5 * clock.dt * u1.x * g_load(c);
  var mom = u1.yzw + kick;
  var E = u2.x;
  var eint = u2.y;
  if ((P.flags & 1u) == 0u) {
    // Kick-consistent energy: add exactly the kinetic energy of the kick (no cancellation).
    E += dot(kick, u1.yzw + 0.5 * kick) / u1.x;
  } else {
    // Flux-form energy: E already holds the work for the whole step; re-derive e_int.
    let en = dual_energy(u1.x, 0.5 * dot(mom, mom) / u1.x, E, eint);
    E = en.x;
    eint = en.y;
  }

  let tame = tame_fluff(u1.x, mom, eint);
  if (tame.changed) {
    mom = tame.mom;
    eint = tame.eint;
    E = 0.5 * dot(mom, mom) / u1.x + eint;
  }

  textureStore(O1, c, vec4<f32>(u1.x, mom));
  textureStore(O2, c, vec4<f32>(E, eint, u2.z, u2.w));
}

// Second half of the gravity kick, applied after the hydro update with the acceleration from
// the potential of the updated density. Together with the half kick in hydro/update.wgsl the
// momentum source is dt/2 (rho^n g^n + rho^{n+1} g^{n+1}), i.e. time-centered.
//
// With a magnetic field (MHD; mhd/update.wgsl has the first half) E holds B^2/2 of the updated
// faces (BF). It stays out of what this pass changes: the sinks take a share of the gas's
// energy, not the field's (the field stays where it is: flux cannot be removed without breaking
// constrained transport), and E is rebuilt from the kinetic, internal and magnetic energy.
#include "hydro/common.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "hydro/fluff.wgsl"
#include <frame>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var N1: texture_3d<f32>;       // state after the hydro update
@group(0) @binding(3) var N2: texture_3d<f32>;
@group(0) @binding(4) var G: texture_3d<f32>;        // gravity field of that state
@group(0) @binding(5) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(6) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var<storage, read> BH: Masses;
@group(0) @binding(8) var BF: texture_3d<f32>;       // with a field: the updated faces

override MHD: bool = false;
#include "mhd/bcell.wgsl"

#include "gravity/bh.wgsl"
#include "gravity/sink.wgsl"

#include "gravity/gfield.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  g_dx = cell_dx(P.dx, clock.a1);
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  comp_use(comp_tex(c));
  var u1 = textureLoad(N1, c, 0);
  var u2 = textureLoad(N2, c, 0);
  var emag = 0.0;
  if (MHD) {
    let B = bcell(c);
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
    let left = apply_sinks(u1, u2 - vec4<f32>(emag, 0.0, 0.0, 0.0), cell_pos(c), clock.dt);
    u1 = left.u1;
    u2 = left.u2 + vec4<f32>(emag, 0.0, 0.0, 0.0);
  }

  let kx = 0.5 * clock.dt * u1.x * ext_accel(cell_pos(c));
  let kick = 0.5 * clock.dt * u1.x * g_self(c) + kx;
  var mom = u1.yzw + kick;
  var E = u2.x;
  var eint = u2.y;
  if ((P.flags & 1u) == 0u) {
    // Kick-consistent energy: add exactly the kinetic energy of the kick (no cancellation).
    E += dot(kick, u1.yzw + 0.5 * kick) / u1.x;
  } else {
    // Flux-form energy: E already holds self-gravity's work for the whole step; the point
    // masses' and the frame's part of the kick adds its own (gravity/ext.wgsl). Then re-derive
    // e_int.
    E += dot(kx, u1.yzw + 0.5 * kick) / u1.x;
    let en = dual_energy(u1.x, 0.5 * dot(mom, mom) / u1.x + emag, E, eint);
    E = en.x;
    eint = en.y;
  }

  // The second half step of a non-inertial frame's velocity-dependent forces.
  if (FRAME_ON) {
    let fs = frame_close(vec4<f32>(u1.x, mom), vec4<f32>(E, eint, u2.zw), 0.5 * clock.dt);
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

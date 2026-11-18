// The frame slot for a rotating frame (gravity/rot.wgsl): the Coriolis force turns the momentum
// by -2 |Omega| tau about Omega (tau half the step) before the fluxes and again after the second kick, so a step is
// turn, kick, fluxes, kick, turn: symmetric in time. A turn keeps |m|, so E is unchanged.
#include "gravity/rot.wgsl"
const FRAME_ON = true;

fn frame_open(u1: vec4<f32>, u2: vec4<f32>, tau: f32) -> FrameState {
  return FrameState(vec4<f32>(u1.x, rot_turn(u1.yzw, tau)), u2);
}
fn frame_close(u1: vec4<f32>, u2: vec4<f32>, tau: f32) -> FrameState {
  return FrameState(vec4<f32>(u1.x, rot_turn(u1.yzw, tau)), u2);
}
// -2 Omega x v: first order is enough in the predictor.
fn frame_vaccel(v: vec3<f32>) -> vec3<f32> {
  return -2.0 * cross(P.rot.xyz, v);
}

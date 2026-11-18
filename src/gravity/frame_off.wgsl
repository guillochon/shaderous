// The frame slot (sim.ts hydroSlots): forces that depend on the gas's velocity, applied as an
// exact operator around each step: half before the fluxes (frame_open) and half after the
// second kick (frame_close), each over the time `tau` it is given (half the step). This is the stub for an inertial, static frame, whose constants
// let the compiler drop every call. gravity/frame_rot.wgsl is the rotating frame's.
const FRAME_ON = false;

// The state (U1, U2) after a time tau of the frame's forces, at the start of a step.
fn frame_open(u1: vec4<f32>, u2: vec4<f32>, tau: f32) -> FrameState {
  return FrameState(u1, u2);
}
// The same after the step's second half kick.
fn frame_close(u1: vec4<f32>, u2: vec4<f32>, tau: f32) -> FrameState {
  return FrameState(u1, u2);
}
// For the predictor's half step: the frame's rates of change of the velocity and the pressure.
fn frame_predict(v: vec3<f32>, rho: f32, p: f32) -> vec4<f32> {
  return vec4<f32>(0.0);
}

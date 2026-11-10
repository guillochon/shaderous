// Third-order reconstruction, limited only where the flow is not smooth.
//
// Each face of a cell gets its own increment. Unlimited, the face value is that of the
// parabola through the cell and its two neighbours, c + (far + 2 near) / 6, where `near` is
// the difference towards the face and `far` the difference away from it: third-order accurate
// on the same three-cell stencil a slope limiter uses.
//
// A limiter that only looks at three cells cannot tell a smooth maximum from a jump, and
// flattens both, which is what holds a limited scheme to second order or less. So the limiter
// (Koren 1993, with the bounds of MC) is applied only where a quantity changes by more than a
// set fraction of its natural scale across the cell: the density itself, the pressure itself,
// the sound speed for velocities. Below that the reconstruction is unlimited; jumps smaller
// than the fraction are therefore not limited either, and can ring by a small part of their
// own height. (After Cada & Torrilhon 2009, with a relative measure of smoothness in place
// of their fixed length.)
const LIMITER_SYMMETRIC = false;
const SMOOTH: f32 = 0.01;   // relative change per cell below which nothing is limited

fn edge4(far: vec4<f32>, near: vec4<f32>, scale: vec4<f32>) -> vec4<f32> {
  let free = (far + 2.0 * near) / 3.0;
  let m = min(min(2.0 * abs(far), 2.0 * abs(near)), abs(free));
  let lim = select(vec4<f32>(0.0), sign(near) * m, far * near > vec4<f32>(0.0));
  let s = SMOOTH * scale;
  let eta = (far * far + near * near) / max(s * s, vec4<f32>(1e-30));
  return mix(lim, free, clamp(1.5 - eta, vec4<f32>(0.0), vec4<f32>(1.0)));
}
// Single slope, where one is needed (prolongation on the adaptive mesh): MC.
fn limit4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let m = min(min(2.0 * abs(a), 2.0 * abs(b)), 0.5 * abs(a + b));
  return select(vec4<f32>(0.0), sign(a) * m, a * b > vec4<f32>(0.0));
}

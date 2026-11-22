// Third-order reconstruction, limited only where the flow is not smooth.
//
// Each face of a cell gets its own increment. Unlimited, the face value is that of the
// parabola through the cell and its two neighbors, c + (far + 2 near) / 6, where `near` is
// the difference toward the face and `far` the difference away from it: third-order accurate
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

// Where every component is smooth (eta <= 0.5, so the weight of the unlimited value is 1), the
// limited value is not needed: returning the unlimited one at once gives what the blend gives,
// to its round-off, and skips the limiter in most cells of most flows. Measured on an A100 (Chrome 158):
// a step 7 percent faster on a 2048^2 Kelvin-Helmholtz grid and 9 percent on a 256^3 Sedov
// blast; replacing the divisions below with multiplications gained nothing.
fn edge4(far: vec4<f32>, near: vec4<f32>, scale: vec4<f32>) -> vec4<f32> {
  let free = (far + 2.0 * near) / 3.0;
  let s = SMOOTH * scale;
  let ss = max(s * s, vec4<f32>(1e-30));
  let q = far * far + near * near;
  if (all(q <= 0.5 * ss)) { return free; }
  let m = min(min(2.0 * abs(far), 2.0 * abs(near)), abs(free));
  let lim = select(vec4<f32>(0.0), sign(near) * m, far * near > vec4<f32>(0.0));
  let eta = q / ss;
  return mix(lim, free, clamp(1.5 - eta, vec4<f32>(0.0), vec4<f32>(1.0)));
}
// Single slope, where one is needed (prolongation on the adaptive mesh): MC.
fn limit4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let m = min(min(2.0 * abs(a), 2.0 * abs(b)), 0.5 * abs(a + b));
  return select(vec4<f32>(0.0), sign(a) * m, a * b > vec4<f32>(0.0));
}

// HLLE flux for ideal MHD in the frame where x is the face normal (Davis wave-speed estimates
// from the fast magnetosonic speeds). `bn` is the normal field on the face, shared by both
// sides. Requires mhd/riemann/common.wgsl.
fn mhd_riemann(l: MSide, r: MSide, bn: f32, w_hll: f32) -> MFlux {
  return mhd_hlle(l, r, bn);
}

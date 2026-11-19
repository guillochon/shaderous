// The radiation's face flux (radiation/sweep.wgsl on the uniform grid, amr/rad_sweep.wgsl on
// blocks). Requires radiation/common.wgsl.

// What a face flux needs of a cell: its radiation, and kappa_R rho for the optical depth.
struct RCell { E: f32, F: vec3<f32>, chi: f32 }

struct RFlux { E: f32, F: vec3<f32> }

// The flux of (E, F) along axis a through the face between q1 and q2, from the four cells
// q0, q1 | q2, q3 in the order of increasing position, in cells of size dx: as RefRad1D.faceFlux. Linear
// reconstruction of E and the reduced flux f = F / (c E), limited (rad_slope); a face whose reconstructed
// states are not physical (E <= 0 or |f| > 1) takes the two cells' own.
fn face_flux(a: i32, dx: f32, q0: RCell, q1: RCell, q2: RCell, q3: RCell) -> RFlux {
  let c = rad_c();
  let f0 = rad_reduced(q0.E, q0.F);
  var fL = rad_reduced(q1.E, q1.F);
  var fR = rad_reduced(q2.E, q2.F);
  let f3 = rad_reduced(q3.E, q3.F);
  // Slopes of (E, f) of the two cells beside the face.
  let s1 = rad_slope(vec4<f32>(q1.E - q0.E, fL - f0), vec4<f32>(q2.E - q1.E, fR - fL));
  let s2 = rad_slope(vec4<f32>(q2.E - q1.E, fR - fL), vec4<f32>(q3.E - q2.E, f3 - fR));
  var EL = q1.E;
  var ER = q2.E;
  let eL = q1.E + 0.5 * s1.x;
  let eR = q2.E - 0.5 * s2.x;
  let gL = fL + 0.5 * s1.yzw;
  let gR = fR - 0.5 * s2.yzw;
  if (eL > 0.0 && eR > 0.0 && dot(gL, gL) <= 1.0 && dot(gR, gR) <= 1.0) {
    EL = eL;
    ER = eR;
    fL = gL;
    fR = gR;
  }
  let FL = fL * (c * EL);
  let FR = fR * (c * ER);
  let dL = rad_eddington_row(fL, a);
  let dR = rad_eddington_row(fR, a);
  // The largest wave speed of the M1 system along a, bounded by c (or c itself: GLF).
  var S = c;
  if (!rad_glf()) {
    S = c * min(1.0, max(max(sqrt(max(dL[a], 0.0)), sqrt(max(dR[a], 0.0))), max(abs(fL[a]), abs(fR[a]))));
  }
  let tau = 0.5 * (q1.chi + q2.chi) * dx;
  let eps = min(1.0, 1.0 / max(tau, 1.0e-30));
  var out: RFlux;
  out.E = 0.5 * (FL[a] + FR[a]) - 0.5 * eps * S * (ER - EL);
  out.F = 0.5 * c * c * (dL * EL + dR * ER) - 0.5 * eps * S * (FR - FL);
  return out;
}

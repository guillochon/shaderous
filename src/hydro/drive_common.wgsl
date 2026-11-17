// Turbulence driving (hydro/drive.ts): the forcing modes as the GPU keeps them. The idea of
// driving the box with Ornstein-Uhlenbeck modes in a band of wavenumbers follows the taranis
// WebGPU reduced-MHD page of Mallet & Abraham; the method is that of Eswaran & Pope (1988) with
// the solenoidal / compressive projection of Federrath et al. (2010).
//
// The acceleration is a(x) = sum over modes m of ar_m cos(k_m . x) - ai_m sin(k_m . x): the real
// part of a_m exp(i k_m . x), with one mode of each pair +-k. hydro/drive.ts writes the modes'
// wavevectors and their first amplitudes; hydro/drive_ou.wgsl advances the amplitudes once per
// clock pass.
//
//   n     (modes in use, 0 for none; clock passes so far, the random stream's counter;
//          seed; dimensions)
//   par   (correlation time tau, noise amplitude sigma, solenoidal weight zeta, 0)
//   mode  three vec4 per mode: (k, 0), (ar, 0), (ai, 0)
const DRIVE_MODES = 64u;
struct Drive {
  n: vec4<u32>,
  par: vec4<f32>,
  mode: array<vec4<f32>, 192>,
}

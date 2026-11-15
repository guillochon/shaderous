// One network step for each case (validation: burn/network.ts against its float64 mirror).
// IN: per case 16 floats, log10 rho, log10 T, log10 dt and the 13 mass fractions. OUT: per case NET_TEST_STRIDE floats: the 13 changes, the inputs and the
// network's outputs.
#include "hydro/burn_net_core.wgsl"

@group(0) @binding(1) var<storage, read> IN: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

const NET_TEST_STRIDE: u32 = 128u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= arrayLength(&IN) / 16u) { return; }
  let i0 = k * 16u;
  var X: array<f32, 13>;
  for (var i = 0u; i < 13u; i++) { X[i] = IN[i0 + 3u + i]; }
  let f = net_features(IN[i0], IN[i0 + 1u], IN[i0 + 2u], X);
  let z = net_eval(f);
  let D = net_step(IN[i0], IN[i0 + 1u], IN[i0 + 2u], X);
  let o = k * NET_TEST_STRIDE;
  for (var i = 0u; i < 13u; i++) { OUT[o + i] = D[i]; }
  for (var i = 0u; i < NET_IN; i++) { OUT[o + 13u + i] = f[i]; }
  for (var i = 0u; i < NET_OUT; i++) { OUT[o + 13u + NET_IN + i] = z[i]; }
}

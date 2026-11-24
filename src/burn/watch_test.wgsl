// The watch's energy rate for each case (validation: burn/network.ts netWatchSlot against
// netEnergyRate). IN: per case 5 floats, rho (g/cm^3), T (K), and the mass fractions of he4,
// c12 and o16. OUT: per case the rate (erg/g/s).
#include <burn_watch>
#include "hydro/burn_screen.wgsl"

@group(0) @binding(1) var<storage, read> IN: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= arrayLength(&IN) / 5u) { return; }
  let i0 = k * 5u;
  OUT[k] = net_watch_rate(IN[i0], IN[i0 + 1u], IN[i0 + 2u], IN[i0 + 3u], IN[i0 + 4u]);
}

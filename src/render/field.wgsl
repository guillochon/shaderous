// Extracts the displayed quantity from the conserved state, normalised to [0, 1], into a
// filterable rgba16float volume shared by the ray-marcher and the slice view.
#include "hydro/common.wgsl"
#include <eos>

struct View {
  field: u32,      // 0 density, 1 temperature, 2 passive scalar, 3 pressure, 4 speed
  logscale: u32,
  lo: f32,         // range ends, in log10 units when logscale is set
  hi: f32,
}
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var VOL: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  g_dx = P.dx;
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  let rho = max(u1.x, P.rho_floor);
  var x = rho;
  switch (V.field) {
    case 1u { x = eos_temperature(rho, u2.y); }
    case 2u { x = u2.z / rho; }
    case 3u { x = eos_pressure(rho, u2.y); }
    case 4u { x = length(u1.yzw) / rho; }
    default {}
  }
  if (V.logscale != 0u) { x = log(max(x, 1.0e-30)) * 0.4342944819; }
  var t = clamp((x - V.lo) / max(V.hi - V.lo, 1.0e-20), 0.0, 1.0);
  if (u2.w > 0.5) { t = -1.0; }   // solid: drawn as a body, not as gas
  textureStore(VOL, c, vec4<f32>(t, 0.0, 0.0, 0.0));
}

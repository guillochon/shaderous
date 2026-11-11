// Extracts the displayed quantity from the conserved state, normalized to [0, 1], into a
// filterable rgba16float volume shared by the ray-marcher and the slice view, and records the
// range of what it draws (render/field_value.wgsl).
#include "hydro/common.wgsl"
#include <eos>
#include "render/field_value.wgsl"

struct View {
  field: u32,      // index into FIELDS (src/problems/types.ts)
  logscale: u32,
  lo: f32,         // range ends, in log10 units when logscale is set
  hi: f32,
}
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var VOL: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var<storage, read_write> RANGE: array<atomic<u32>, 132>;

// Velocity and density of a neighboring cell, clamped to the grid.
fn neighbor(c: vec3<i32>) -> vec4<f32> {
  let u = textureLoad(U1, clamp(c, vec3<i32>(0), vec3<i32>(P.dims.xyz) - 1), 0);
  let rho = max(u.x, P.rho_floor);
  return vec4<f32>(u.yzw / rho, rho);
}

fn gradients(c: vec3<i32>) -> Grad {
  var d = array<vec4<f32>, 3>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
  let axes = select(3, 2, P.dims.z == 1u);
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    d[a] = (neighbor(c + e) - neighbor(c - e)) / (2.0 * P.dx);
  }
  // d[a] = derivative along axis a of (vx, vy, vz, rho).
  return Grad(vec3<f32>(d[0].x, d[1].x, d[2].x), vec3<f32>(d[0].y, d[1].y, d[2].y), vec3<f32>(d[0].z, d[1].z, d[2].z), vec3<f32>(d[0].w, d[1].w, d[2].w));
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  g_dx = P.dx;
  range_begin(lix);
  if (all(gid < P.dims.xyz)) {
    let c = vec3<i32>(gid);
    let u1 = textureLoad(U1, c, 0);
    let u2 = textureLoad(U2, c, 0);
    var g: Grad;
    if (field_needs_grad(V.field)) { g = gradients(c); }
    var x = field_value(V.field, u1, u2, g, 0.0);
    let solid = u2.w > 0.5;
    if (!solid) { range_add(x); }
    if (V.logscale != 0u) { x = log(max(x, 1.0e-30)) * 0.4342944819; }
    var t = clamp((x - V.lo) / max(V.hi - V.lo, 1.0e-20), 0.0, 1.0);
    if (solid) { t = -1.0; }   // solid: drawn as a body, not as gas
    textureStore(VOL, c, vec4<f32>(t, 0.0, 0.0, 0.0));
  }
  range_end(lix);
}

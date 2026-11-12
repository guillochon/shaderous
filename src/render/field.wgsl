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
  nd: vec4<u32>,   // display volume size (used by the block-grid version)
  cut: vec4<f32>,  // volume view: log10 of the minimum density shown, and 1 (0 in the slice view)
  sp: vec4<u32>,   // mass fraction: species, texels along z between groups of four, groups
}

// Volume view: 1 where the gas is dense enough to be shown, stored in g. The slice view keeps g
// for the block outlines of an adaptive mesh.
fn shown(rho: f32) -> f32 {
  return select(0.0, 1.0, log(max(rho, 1.0e-30)) * 0.4342944819 >= V.cut.x);
}
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var VOL: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var GF: texture_3d<f32>;   // gravity field: acceleration, potential in w
@group(0) @binding(8) var<storage, read_write> RANGE: array<atomic<u32>, 132>;
@group(0) @binding(9) var S: texture_3d<f32>;     // rho X of the species, groups stacked along z

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
    // The uniform field's potential, measured from the box center, is not in the texture.
    let pos = (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
    let phi = textureLoad(GF, c, 0).w - dot(P.gext.xyz, select(pos, vec3<f32>(pos.xy, 0.0), P.dims.z == 1u));
    g_acc = P.gext.xyz;
    if (P.grav_on != 0u) { g_acc += textureLoad(GF, c, 0).xyz; }
    var x = field_value(V.field, u1, u2, g, 0.0, phi);
    if (V.field == F_MASS_FRACTION) { x = mass_fraction(c); }
    let solid = u2.w > 0.5;
    if (!solid) { range_add(x); }
    if (V.logscale != 0u) { x = log(max(x, 1.0e-30)) * 0.4342944819; }
    var t = clamp((x - V.lo) / max(V.hi - V.lo, 1.0e-20), 0.0, 1.0);
    if (solid) { t = -1.0; }   // solid: drawn as a body, not as gas
    textureStore(VOL, c, vec4<f32>(t, select(0.0, shown(u1.x), V.cut.y != 0.0), 0.0, 0.0));
  }
  range_end(lix);
}

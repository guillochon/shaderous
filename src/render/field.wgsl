// Extracts the displayed quantity from the conserved state, normalized to [0, 1], into a
// filterable rgba16float volume the ray-marcher reads, and records the range of what it draws
// (render/field_value.wgsl). The slice and projection views read the cells themselves
// (plane() and column() below, render/project.wgsl).
#include "hydro/common.wgsl"
#include <eos>
#include "render/field_value.wgsl"
#include "render/project.wgsl"

struct View {
  field: u32,      // index into FIELDS (src/problems/types.ts)
  logscale: u32,
  lo: f32,         // range ends, in log10 units when logscale is set
  hi: f32,
  nd: vec4<u32>,   // display volume size (used by the block-grid version); slice and projection map size in w
  cut: vec4<f32>,  // volume view: log10 of the minimum density shown, and 1 (0 in the slice view); slice position in z; 1 with isosurfaces on
  sp: vec4<u32>,   // mass fraction: species, texels along z between groups of four, groups; slice or projection axis in w
}

// Volume view: 1 where the gas is dense enough to be shown, stored in g.
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
@group(0) @binding(10) var PB: texture_3d<f32>;   // predictor's (p, scalar, shock flag, first order), as of the last step

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

// The field at cell c (display voxels are the grid's cells).
fn voxel(c: vec3<i32>) -> Cell {
  g_dx = P.dx;
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  var g: Grad;
  if (field_needs_grad(V.field)) { g = gradients(c); }
  // The uniform field's potential, measured from the box center, is not in the texture.
  let pos = (vec3<f32>(c) + 0.5) * P.dx - 0.5 * P.box;
  let phi = textureLoad(GF, c, 0).w - dot(P.gext.xyz, select(pos, vec3<f32>(pos.xy, 0.0), P.dims.z == 1u));
  g_acc = P.gext.xyz;
  if (P.grav_on != 0u) { g_acc += textureLoad(GF, c, 0).xyz; }
  comp_use(comp_tex(c));
  var x = field_value(V.field, u1, u2, g, 0.0, phi);
  if (V.field == F_MASS_FRACTION) { x = mass_fraction(c); }
  if (V.field == F_SHOCK) { x = textureLoad(PB, c, 0).z; }
  return Cell(x, u1.x, u2.w > 0.5);
}

// Slice: the cell under texel `ij` of the map (one texel per cell), no outlines.
fn plane(ij: vec2<i32>, ab: vec2<i32>, axis: i32) -> Plane {
  let depth = f32(P.dims[axis]);
  var c = vec3<i32>(0);
  c[ab.x] = ij.x;
  c[ab.y] = ij.y;
  c[axis] = i32(clamp(floor(V.cut.z * depth), 0.0, depth - 1.0));
  return Plane(voxel(c), 0.0, 1.0);
}

// Projection: the cells along `axis` through cell `ij` of the map (one map texel per cell).
fn column(ij: vec2<i32>, ab: vec2<i32>, axis: i32) -> Column {
  var s = Column(0.0, 0.0, 0.0, false);
  var c = vec3<i32>(0);
  c[ab.x] = ij.x;
  c[ab.y] = ij.y;
  for (var k = 0; k < i32(P.dims[axis]); k++) {
    c[axis] = k;
    column_add(&s, voxel(c), P.dx, 1.0);
  }
  return s;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  range_begin(lix);
  if (all(gid < P.dims.xyz)) {
    let c = vec3<i32>(gid);
    let q = voxel(c);
    if (!q.solid) { range_add(q.x); }
    // Solid: drawn as a body, not as gas.
    // b: log10 |value|, for the isosurfaces.
    textureStore(VOL, c, vec4<f32>(display_t(q.x, q.solid), select(0.0, shown(q.rho), V.cut.y != 0.0), log10_abs(q.x), 0.0));
  }
  range_end(lix);
}

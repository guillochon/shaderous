// Shared by the display passes of the uniform grid (render/field.wgsl) and the block grid
// (amr/field.wgsl): the normalization of a value for the transfer function, and the projection
// view's pass, which integrates along the axis V.sp.w through the center of each texel of a
// V.nd.w x V.nd.w map (the simulation's resolution). Density becomes the column density (the
// integral of rho dl); any other field the mass-weighted mean along the line of sight (that of
// rho x dl over that of rho dl). Solid cells are left out; a column with nothing but solid is
// drawn as a body.
// Requires V, P, RANGE and the field shader's column(), which walks the cells the line crosses.

struct Cell {
  x: f32,       // the field's value
  rho: f32,     // density, as stored
  solid: bool,
}

// Integrals along a line of sight: rho dl, rho x dl, and whether any gas was crossed.
struct Column {
  m: f32,
  mx: f32,
  gas: bool,
}

fn column_add(s: ptr<function, Column>, q: Cell, dl: f32) {
  if (q.solid) { return; }
  let w = max(q.rho, P.rho_floor) * dl;
  (*s).m += w;
  (*s).mx += w * q.x;
  (*s).gas = true;
}

@group(0) @binding(11) var PROJ: texture_storage_2d<rgba16float, write>;

// A value as the transfer function reads it: [0, 1] over the displayed range, -1 for a solid.
fn display_t(x0: f32, solid: bool) -> f32 {
  var x = x0;
  if (V.logscale != 0u) { x = log(max(x, 1.0e-30)) * 0.4342944819; }
  if (solid) { return -1.0; }
  return clamp((x - V.lo) / max(V.hi - V.lo, 1.0e-20), 0.0, 1.0);
}

// The axes of the box along the map's columns and rows, as in the slice view: x right and y up
// looking along z, y right and z up along x, x right and z up along y.
fn map_axes(axis: i32) -> vec2<i32> {
  if (axis == 0) { return vec2<i32>(1, 2); }
  if (axis == 1) { return vec2<i32>(0, 2); }
  return vec2<i32>(0, 1);
}

// One thread per texel of the map. 128 threads, one per bin of the range's histogram.
@compute @workgroup_size(16, 8, 1)
fn project(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  range_begin(lix);
  if (all(gid.xy < vec2<u32>(V.nd.w))) {
    let axis = i32(V.sp.w);
    let s = column(vec2<i32>(gid.xy), map_axes(axis), axis);
    let x = select(s.mx / max(s.m, 1.0e-30), s.m, V.field == F_DENSITY);
    if (s.gas) { range_add(x); }
    textureStore(PROJ, vec2<i32>(gid.xy), vec4<f32>(display_t(x, !s.gas), 0.0, 0.0, 0.0));
  }
  range_end(lix);
}

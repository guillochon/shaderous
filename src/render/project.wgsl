// Shared by the display passes of the uniform grid (render/field.wgsl) and the block grid
// (amr/field.wgsl): the normalization of a value for the transfer function, and the passes that
// fill the 2D map the slice and projection views draw (render/slice.wgsl), one texel per cell of
// the finest level: r the value normalized as display_t() makes it, g 1 on a block's outline,
// b log10 |value| (for the contour lines, whatever the color scale), a the side of the cell the
// texel shows, in texels (1 on a uniform grid; a power of two on an adaptive mesh), so the contour
// lines can be interpolated between the centers of the cells themselves.
//
// The map is V.nd.w texels on a side (the simulation's resolution), across the two axes other
// than V.sp.w. The slice pass takes the cell under each texel's center, in the layer of cells
// V.cut.z of the way along V.sp.w. The projection pass integrates along V.sp.w through each
// texel's center. Density becomes the column density (the
// integral of rho dl); any other field the mass-weighted mean along the line of sight (that of
// rho x dl over that of rho dl). Solid cells are left out; a column with nothing but solid is
// drawn as a body.
// Requires V, P, RANGE and the field shader's column(), which walks the cells the line crosses,
// and plane(), the cell under a texel of the slice.

struct Cell {
  x: f32,       // the field's value
  rho: f32,     // density, as stored
  solid: bool,
}

// Integrals along a line of sight: rho dl, rho x dl, rho dl log2 of the cell's side in map
// texels, and whether any gas was crossed.
struct Column {
  m: f32,
  mx: f32,
  ml: f32,
  gas: bool,
}

// Adds a cell crossed for a length dl, `texels` map texels on a side.
fn column_add(s: ptr<function, Column>, q: Cell, dl: f32, texels: f32) {
  if (q.solid) { return; }
  let w = max(q.rho, P.rho_floor) * dl;
  (*s).m += w;
  (*s).mx += w * q.x;
  (*s).ml += w * log2(texels);
  (*s).gas = true;
}

// A projection's cell side for its contours: the mass-weighted typical cell along the column.
fn column_texels(s: Column) -> f32 {
  return exp2(round(s.ml / max(s.m, 1.0e-30)));
}

@group(0) @binding(11) var PROJ: texture_storage_2d<rgba16float, write>;

// A value as the transfer function reads it: [0, 1] over the displayed range, -1 for a solid.
fn display_t(x0: f32, solid: bool) -> f32 {
  var x = x0;
  if (V.logscale != 0u) { x = log(max(x, 1.0e-30)) * 0.4342944819; }
  if (solid) { return -1.0; }
  return clamp((x - V.lo) / max(V.hi - V.lo, 1.0e-20), 0.0, 1.0);
}

fn log10_abs(x: f32) -> f32 {
  return log(max(abs(x), 1.0e-30)) * 0.4342944819;
}

// The axes of the box along the map's columns and rows, as in the slice view: x right and y up
// looking along z, y right and z up along x, x right and z up along y.
fn map_axes(axis: i32) -> vec2<i32> {
  if (axis == 0) { return vec2<i32>(1, 2); }
  if (axis == 1) { return vec2<i32>(0, 2); }
  return vec2<i32>(0, 1);
}

// The value under a texel of the slice, and whether the texel lies on a block's outline.
struct Plane {
  q: Cell,
  edge: f32,
  texels: f32,   // the side of the cell, in map texels
}

// One thread per texel of the map. The range comes from the display pass over the whole box.
@compute @workgroup_size(16, 8, 1)
fn slice(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (all(gid.xy < vec2<u32>(V.nd.w))) {
    let axis = i32(V.sp.w);
    let s = plane(vec2<i32>(gid.xy), map_axes(axis), axis);
    textureStore(PROJ, vec2<i32>(gid.xy), vec4<f32>(display_t(s.q.x, s.q.solid), s.edge, log10_abs(s.q.x), s.texels));
  }
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
    textureStore(PROJ, vec2<i32>(gid.xy), vec4<f32>(display_t(x, !s.gas), 0.0, log10_abs(x), column_texels(s)));
  }
  range_end(lix);
}

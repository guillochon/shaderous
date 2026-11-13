// Shared by the display passes of the uniform grid (render/field.wgsl) and the block grid
// (amr/field.wgsl): the normalization of a value for the transfer function, and the projection
// view's pass, which sums each column of display voxels along the axis V.sp.w. Density becomes
// the column density (sum of rho dl, V.cut.z being dl); any other field the mass-weighted mean
// along the line of sight (sum of rho x over sum of rho). Solid cells are left out; a column with
// nothing but solid is drawn as a body.
// Requires V, P, RANGE and the field shader's voxel(c), the value at display voxel c.

struct Cell {
  x: f32,       // the field's value
  rho: f32,     // density, as stored
  solid: bool,
}

@group(0) @binding(11) var PROJ: texture_storage_2d<rgba16float, write>;

// A value as the transfer function reads it: [0, 1] over the displayed range, -1 for a solid.
fn display_t(x0: f32, solid: bool) -> f32 {
  var x = x0;
  if (V.logscale != 0u) { x = log(max(x, 1.0e-30)) * 0.4342944819; }
  if (solid) { return -1.0; }
  return clamp((x - V.lo) / max(V.hi - V.lo, 1.0e-20), 0.0, 1.0);
}

// One thread per column; the texel (i, j) is laid out as in the slice view: x right and y up
// along z, y right and z up along x, x right and z up along y. 128 threads, one per bin of the
// range's histogram.
@compute @workgroup_size(16, 8, 1)
fn project(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  range_begin(lix);
  let axis = i32(V.sp.w);
  let nd = vec3<i32>(V.nd.xyz);
  var a = 0;
  var b = 1;
  if (axis == 0) { a = 1; b = 2; }
  if (axis == 1) { b = 2; }
  if (i32(gid.x) < nd[a] && i32(gid.y) < nd[b]) {
    var c = vec3<i32>(0);
    c[a] = i32(gid.x);
    c[b] = i32(gid.y);
    var m = 0.0;
    var mx = 0.0;
    var gas = false;
    for (var k = 0; k < nd[axis]; k++) {
      c[axis] = k;
      let q = voxel(c);
      if (q.solid) { continue; }
      let w = max(q.rho, P.rho_floor);
      m += w;
      mx += w * q.x;
      gas = true;
    }
    let x = select(mx / max(m, 1.0e-30), m * V.cut.z, V.field == F_DENSITY);
    if (gas) { range_add(x); }
    textureStore(PROJ, vec2<i32>(gid.xy), vec4<f32>(display_t(x, !gas), 0.0, 0.0, 0.0));
  }
  range_end(lix);
}

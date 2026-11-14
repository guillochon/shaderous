// Resamples the block grid onto the uniform display volume that the ray-marcher reads. Each
// display voxel takes the value of the leaf cell under its center (render/field_value.wgsl),
// and the pass records the range of what it draws.
//   r = displayed quantity, normalized to [0, 1]
//   g = volume view: 1 where the gas is dense enough to be shown
//   b = log10 |displayed quantity|, for the isosurfaces: with them on (V.cut.w), interpolated
//       trilinearly between the centers of leaf cells the size of the one under the voxel, so a
//       coarse cell does not show as a block of one value (smooth_log() below)
// The slice and projection views read the leaf cells themselves, at the finest level's
// resolution (plane() and column() below, render/project.wgsl). So does the 3D volume view,
// through `atlas` below, which writes the same three channels for every leaf cell into an
// atlas the ray-marcher reads at each block's own resolution (render/volume_atlas.wgsl).
// Derivatives come from the primitive variables, whose ghost cells hold the neighbors across
// block faces at the block's own level (as of the last assembly pass).
#include "hydro/common.wgsl"
#include "gravity/ext_bind.wgsl"
#include <eos>
#include "amr/grid.wgsl"
#include "render/field_value.wgsl"
#include "render/project.wgsl"
#include "amr/list.wgsl"

struct View {
  field: u32,      // index into FIELDS (src/problems/types.ts)
  logscale: u32,
  lo: f32,         // range ends, in log10 units when logscale is set
  hi: f32,
  nd: vec4<u32>,   // display volume size; slice and projection map size in w
  cut: vec4<f32>,  // volume view: log10 of the minimum density shown, and 1 (0 in the slice view); slice position in z; 1 with isosurfaces on
  sp: vec4<u32>,   // mass fraction: species, texels along z between groups of four, groups; slice or projection axis in w
}
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var VOL: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var IDX: texture_3d<u32>;   // leaf slot per deepest-level block
@group(0) @binding(6) var W1: texture_3d<f32>;    // primitives: (density, negative if solid; velocity)
@group(0) @binding(7) var GF: texture_3d<f32>;    // self-gravity's field: acceleration, potential in w
@group(0) @binding(8) var<storage, read_write> RANGE: array<atomic<u32>, 132>;
@group(0) @binding(9) var S: texture_3d<f32>;     // rho X of the species, groups stacked along z
@group(0) @binding(10) var PB: texture_3d<f32>;   // predictor's (p, scalar, shock flag, first order), as of the last step
@group(0) @binding(12) var DA: texture_storage_3d<rgba16float, write>;   // display atlas (`atlas`)
// Per-block timesteps (amr/bins_on.wgsl): BIN[1 + slot] is a block's time bin; all zero when off.
@group(0) @binding(29) var<storage, read> TBIN: array<u32>;

fn neighbor(c: vec3<i32>) -> vec4<f32> {
  let w = textureLoad(W1, c, 0);
  return vec4<f32>(w.yzw, max(abs(w.x), P.rho_floor));
}

fn gradients(c: vec3<i32>, dx: f32) -> Grad {
  var d = array<vec4<f32>, 3>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
  let axes = select(3, 2, is2d());
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    d[a] = (neighbor(c + e) - neighbor(c - e)) / (2.0 * dx);
  }
  return Grad(vec3<f32>(d[0].x, d[1].x, d[2].x), vec3<f32>(d[0].y, d[1].y, d[2].y), vec3<f32>(d[0].z, d[1].z, d[2].z), vec3<f32>(d[0].w, d[1].w, d[2].w));
}

// The leaf cell at a point of the box. Sets g_dx to its size.
struct Leaf {
  slot: i32,
  level: i32,
  nblk: i32,         // blocks across the domain at its level
  l: vec3<i32>,      // cell within the block
  c: vec3<i32>,      // texel in the atlas
  frac: vec3<f32>,   // the point, in fractions of the box
}
fn leaf_at(frac0: vec3<f32>) -> Leaf {
  var frac = frac0;
  if (is2d()) { frac.z = 0.0; }
  let m = i32(P.amr.y << P.amr.z);
  let slot = i32(textureLoad(IDX, vec3<i32>(frac * f32(m)), 0).x);
  g_dx = blk_dx(slot);
  let level = i32(bt(slot, 0u));
  let nblk = i32(P.amr.y) * (1 << u32(level));
  let ncell = nblk * BS;
  var l = clamp(vec3<i32>(frac * f32(ncell)) - blk_coord(slot) * BS, vec3<i32>(0), vec3<i32>(BS - 1));
  if (is2d()) { l.z = 0; }
  return Leaf(slot, level, nblk, l, atl(blk_origin(slot), l), frac);
}

fn leaf_value(f: Leaf) -> Cell {
  let c = f.c;
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  var g: Grad;
  if (field_needs_grad(V.field)) { g = gradients(c, g_dx); }
  // The uniform field's potential, measured from the box center, is not in the texture.
  let xp = blk_pos(f.slot, f.l);
  let phi = textureLoad(GF, c, 0).w + ext_phi(xp) - dot(P.gext.xyz, xp);
  g_acc = P.gext.xyz;
  if (P.grav_on != 0u) { g_acc += textureLoad(GF, c, 0).xyz; }
  comp_use(comp_tex(c));
  var x = field_value(V.field, u1, u2, g, f32(f.level), phi);
  if (V.field == F_MASS_FRACTION) { x = mass_fraction(c); }
  if (V.field == F_SHOCK) { x = textureLoad(PB, c, 0).z; }
  if (V.field == F_TIME_BIN) { x = f32(TBIN[1u + u32(f.slot)]); }
  return Cell(x, u1.x, u2.w > 0.5);
}

// The leaf cell under the center of display voxel v.
fn leaf(v: vec3<i32>) -> Leaf {
  return leaf_at((vec3<f32>(v) + 0.5) / vec3<f32>(V.nd.xyz));
}

// Slice: the leaf cell under the center of texel `ij` of the map, in the middle of the layer of
// finest cells V.cut.z of the way along `axis`, and 1 on the lower edges of its block, one texel
// wide.
fn plane(ij: vec2<i32>, ab: vec2<i32>, axis: i32) -> Plane {
  let finest = f32(i32(P.amr.y << P.amr.z) * BS);
  let side = f32(V.nd.w);
  var p = vec3<f32>(0.0);
  p[ab.x] = (f32(ij.x) + 0.5) / side;
  p[ab.y] = (f32(ij.y) + 0.5) / side;
  p[axis] = (clamp(floor(V.cut.z * finest), 0.0, finest - 1.0) + 0.5) / finest;
  let f = leaf_at(p);
  // Position within the block, in texels of the map.
  let within = (p * f32(f.nblk) - vec3<f32>(blk_coord(f.slot))) * side / f32(f.nblk);
  let edge = select(0.0, 1.0, within[ab.x] < 1.0 || within[ab.y] < 1.0);
  return Plane(leaf_value(f), edge, side / f32(f.nblk * BS));
}

// Projection: the leaf cells along `axis` through the center of texel `ij` of the map, each
// crossed whole, so the integral is exact at every level of refinement. Each step looks up the
// leaf a quarter of a finest cell past the last boundary and moves to its far face.
fn column(ij: vec2<i32>, ab: vec2<i32>, axis: i32) -> Column {
  var s = Column(0.0, 0.0, 0.0, false);
  let finest = f32(i32(P.amr.y << P.amr.z) * BS);
  var p = vec3<f32>(0.0);
  p[ab.x] = (f32(ij.x) + 0.5) / f32(V.nd.w);
  p[ab.y] = (f32(ij.y) + 0.5) / f32(V.nd.w);
  var at = 0.0;   // the boundary reached, in fractions of the box
  for (var i = 0; i < i32(finest) && at < 1.0; i++) {
    p[axis] = at + 0.25 / finest;
    let f = leaf_at(p);
    let ncell = f32(f.nblk * BS);
    column_add(&s, leaf_value(f), g_dx, f32(V.nd.w) / ncell);
    at = (floor(p[axis] * ncell) + 1.0) / ncell;
  }
  return s;
}

// log10 |value| at the center of display voxel `gid`, in leaf `f`, of value `own`: trilinear
// between the centers of the 8 nearest cells of f's size, each taken from the leaf under it
// (finer, coarser or the same), so the ray-marcher's isosurfaces are smooth across coarse cells.
// A leaf no larger than a display voxel keeps its own value; so does a solid's neighbor.
fn smooth_log(f: Leaf, own: f32) -> f32 {
  let ncell = f32(f.nblk * BS);
  if (is2d() || ncell >= f32(V.nd.x)) { return own; }
  let g = f.frac * ncell - 0.5;
  let i0 = floor(g);
  let fr = g - i0;
  var acc = 0.0;
  for (var c = 0u; c < 8u; c++) {
    let o = vec3<f32>(f32(c & 1u), f32((c >> 1u) & 1u), f32((c >> 2u) & 1u));
    // Cell centers past the box are taken from the edge cells, as the boundary does.
    let at = clamp((i0 + o + 0.5) / ncell, vec3<f32>(0.5 / ncell), vec3<f32>(1.0 - 0.5 / ncell));
    let q = leaf_value(leaf_at(at));
    let w = mix(1.0 - fr, fr, o);
    acc += w.x * w.y * w.z * select(log10_abs(q.x), own, q.solid);
  }
  return acc;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  range_begin(lix);
  if (all(gid < V.nd.xyz)) {
    let f = leaf(vec3<i32>(gid));
    let q = leaf_value(f);
    if (!q.solid) { range_add(q.x); }
    // Solid: drawn as a body, not as gas.
    // The volume view draws block outlines as lines.
    let shown = select(0.0, 1.0, log(max(q.rho, 1.0e-30)) * 0.4342944819 >= V.cut.x);
    // b: log10 |value|, for the isosurfaces (smoothed across coarse cells when they are on).
    var lg = log10_abs(q.x);
    if (V.cut.w != 0.0 && !q.solid) { lg = smooth_log(f, lg); }
    textureStore(VOL, vec3<i32>(gid), vec4<f32>(display_t(q.x, q.solid), select(0.0, shown, V.cut.y != 0.0), lg, 0.0));
  }
  range_end(lix);
}

// The display atlas of a 3D adaptive mesh (render/volume_atlas.wgsl): each leaf block's cells,
// and one layer of border cells all round holding the leaf there (of any level; at the faces of
// the domain, the edge cells), so the ray-marcher's trilinear filter is continuous across
// blocks. Dispatched over the leaf list like the padded passes. Records the range as the uniform
// pass does, each cell counted as the number of finest cells it covers.
@compute @workgroup_size(8, 8, 4)
fn atlas(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lix: u32) {
  range_begin(lix);
  let inv = inv_padded(gid, 1);
  if (inv.ok) {
    let slot = inv.slot;
    let l = inv.l;
    let level = i32(bt(slot, 0u));
    let nblk = i32(P.amr.y) * (1 << u32(level));
    let ncell = f32(nblk * BS);
    let inside = all(l >= vec3<i32>(0)) && all(l < vec3<i32>(BS));
    let at = (vec3<f32>(blk_coord(slot) * BS + l) + 0.5) / ncell;
    var f: Leaf;
    if (inside) {
      g_dx = blk_dx(slot);
      f = Leaf(slot, level, nblk, l, atl(blk_origin(slot), l), at);
    } else {
      f = leaf_at(clamp(at, vec3<f32>(0.5 / ncell), vec3<f32>(1.0 - 0.5 / ncell)));
    }
    let q = leaf_value(f);
    if (inside && !q.solid) { range_add_w(q.x, 1u << (3u * (P.amr.z - u32(level)))); }
    let shown = select(0.0, 1.0, log(max(q.rho, 1.0e-30)) * 0.4342944819 >= V.cut.x);
    textureStore(DA, atl(blk_origin(slot), l), vec4<f32>(display_t(q.x, q.solid), select(0.0, shown, V.cut.y != 0.0), log10_abs(q.x), 0.0));
  }
  range_end(lix);
}

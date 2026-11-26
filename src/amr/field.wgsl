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

// |x| over the top of the color range (a: for isosurfaces interpolated in the value rather than its log; as render/field.wgsl).
fn lin_scaled(x: f32) -> f32 {
  let top = select(log(max(abs(V.hi), 1.0e-30)) * 0.4342944819, V.hi, V.logscale != 0u);
  return abs(x) * exp2(-top * 3.321928095);
}
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
@group(0) @binding(13) var BF: texture_3d<f32>;   // face-centered magnetic field (mhd/amr_mhd.ts); a placeholder without one
// The magnetic diagnostics' reduction (mhd/mhd.ts, MAG_OPS), as of the frame: [2] is the largest
// |B| anywhere. The divergence is shown over it, as the run's ∇·B diagnostic is, and the plasma
// beta takes at least (1e-4 of it)^2 for B^2: a block the field has not reached holds only
// round-off, whose divergence over its own |B| and whose beta would otherwise set the color range.
// A placeholder without a field.
@group(0) @binding(14) var<storage, read> MAG: array<f32, 4>;
// Burning's heating rate over the last step, per unit volume (burn/burn.ts); one texel of 0 without.
@group(0) @binding(15) var HEAT: texture_3d<f32>;
#include "mhd/mri.wgsl"

// Radiation (amr/radiation.ts): (E, F) of the leaves' cells as of the frame; a placeholder texel without.
@group(0) @binding(16) var RAD: texture_3d<f32>;
#include <opacity>
#include "radiation/common.wgsl"

// The radiation fields, F_RADIATION onward, at the leaf cell at atlas texel c, of density rho and
// internal energy eint (render/field.wgsl radiation_value). g_dx is the cell's size.
fn radiation_value(f: u32, c: vec3<i32>, rho: f32, eint: f32) -> f32 {
  if (P.rad.w == 0.0) { return 0.0; }
  let R = textureLoad(RAD, c, 0);
  let E = max(R.x, 0.0);
  let tau = kappa_rosseland(rho, eos_temperature(rho, eint)) * rho * g_dx;
  switch (f - F_RADIATION) {
    case 0u { return E; }
    case 1u { return sqrt(sqrt(E / rad_a())); }
    case 2u { return length(R.yzw); }
    case 3u { return select(0.0, length(R.yzw) / (rad_c() * E), E > 0.0); }
    case 4u { return tau; }
    default { return rad_reduction(tau); }
  }
}

// The magnetic fields, F_MAGNETIC onward, at the leaf cell at atlas texel c, whose gas pressure
// is p and density rho, at x (relative to the box center): from the block's own faces (texel BS
// holds its upper ones). g_dx is the cell's size.
fn magnetic_value(f: u32, c: vec3<i32>, p: f32, rho: f32, x: vec3<f32>) -> f32 {
  let lo = textureLoad(BF, c, 0).xyz;
  var hi = lo;
  var div = 0.0;
  for (var a = 0; a < 3; a++) {
    if (a == 2 && is2d()) { break; }
    var e = vec3<i32>(0);
    e[a] = 1;
    hi[a] = textureLoad(BF, c + e, 0)[a];
    div += hi[a] - lo[a];
  }
  return magnetic_of(f, 0.5 * (lo + hi), div, p, rho, x);
}

// The same from the cell's field B and the sum of its faces' differences div.
fn magnetic_of(f: u32, B: vec3<f32>, div: f32, p: f32, rho: f32, x: vec3<f32>) -> f32 {
  let b2 = dot(B, B);
  switch (f - F_MAGNETIC) {
    case 0u, 1u, 2u { return B[f - F_MAGNETIC]; }
    case 3u { return sqrt(b2); }
    case 4u {
      let floor = 1.0e-4 * MAG[2];
      return 2.0 * p / max(b2, max(floor * floor, 1.0e-30));
    }
    case 6u, 7u { return mri_quality(B, rho, p, x, g_dx)[f - F_MAGNETIC - 6u]; }
    default { return abs(div) / max(MAG[2], 1.0e-30); }
  }
}
// View only, one field held (AmrSim viewBaked, plans/view-only.md): the field and the density at
// each leaf cell, without ghost cells, in slots of BS cells laid out as the atlas's are. Written
// by `bake` from a checkpoint's primitives, a batch of blocks at a time; read by leaf_value in
// place of the state. A placeholder texel otherwise.
override BAKED: u32 = 0u;
@group(0) @binding(17) var BK: texture_3d<f32>;   // texel bk_texel (amr/grid.wgsl)

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
  // View only, one field held (AmrSim viewBaked): the field and the density as `bake` left them.
  if (BAKED != 0u) {
    let b = textureLoad(BK, bk_texel(f.slot, f.l), 0);
    return Cell(b.x, b.y, false);
  }
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
  if (V.field == F_NUCLEAR_HEATING) { x = textureLoad(HEAT, c, 0).x; }
  if (V.field == F_SHOCK) { x = textureLoad(PB, c, 0).z; }
  if (V.field == F_TIME_BIN) { x = f32(TBIN[1u + u32(f.slot)]); }
  if (V.field >= F_MAGNETIC && V.field < F_MAGNETIC + N_MAGNETIC) { x = magnetic_value(V.field, c, eos_pressure(max(u1.x, P.rho_floor), u2.y), u1.x, xp); }
  if (V.field >= F_RADIATION && V.field < F_RADIATION + N_RADIATION) { x = radiation_value(V.field, c, max(u1.x, P.rho_floor), u2.y); }
  return Cell(x, u1.x, u2.w > 0.5);
}

// The leaf cell under the center of display voxel v.
fn leaf(v: vec3<i32>) -> Leaf {
  return leaf_at((vec3<f32>(v) + 0.5) / vec3<f32>(V.nd.xyz) * cube_scale());
}

// Slice: the leaf cell under the center of texel `ij` of the map, in the middle of the layer of
// finest cells V.cut.z of the way along `axis`, and 1 on the lower edges of its block, one texel
// wide.
fn plane(ij: vec2<i32>, ab: vec2<i32>, axis: i32) -> Plane {
  // (In fractions of the computed box, then of the tree's cube: cube_scale.)
  let s = cube_scale();
  let finest = f32(i32(P.amr.y << P.amr.z) * BS) * s[axis];
  let side = f32(V.nd.w);
  var p = vec3<f32>(0.0);
  p[ab.x] = (f32(ij.x) + 0.5) / side;
  p[ab.y] = (f32(ij.y) + 0.5) / side;
  p[axis] = (clamp(floor(V.cut.z * finest), 0.0, finest - 1.0) + 0.5) / finest;
  p *= s;
  let f = leaf_at(p);
  // Position within the block, in texels of the map.
  let within = (p * f32(f.nblk) - vec3<f32>(blk_coord(f.slot))) * side / (f32(f.nblk) * s);
  let edge = select(0.0, 1.0, within[ab.x] < 1.0 || within[ab.y] < 1.0);
  return Plane(leaf_value(f), edge, side / f32(f.nblk * BS));
}

// Projection: the leaf cells along `axis` through the center of texel `ij` of the map, each
// crossed whole, so the integral is exact at every level of refinement. Each step looks up the
// leaf a quarter of a finest cell past the last boundary and moves to its far face.
fn column(ij: vec2<i32>, ab: vec2<i32>, axis: i32) -> Column {
  var s = Column(0.0, 0.0, 0.0, false);
  let finest = f32(i32(P.amr.y << P.amr.z) * BS);
  let cs = cube_scale();
  var p = vec3<f32>(0.0);
  p[ab.x] = (f32(ij.x) + 0.5) / f32(V.nd.w) * cs[ab.x];
  p[ab.y] = (f32(ij.y) + 0.5) / f32(V.nd.w) * cs[ab.y];
  var at = 0.0;   // the boundary reached, in fractions of the tree's cube (the box ends at cs)
  for (var i = 0; i < i32(finest) && at < cs[axis]; i++) {
    p[axis] = at + 0.25 / finest;
    let f = leaf_at(p);
    let ncell = f32(f.nblk * BS);
    column_add(&s, leaf_value(f), g_dx, f32(V.nd.w) / ncell);
    at = (floor(p[axis] * ncell) + 1.0) / ncell;
  }
  return s;
}

// Profile: the leaf cell under a point, in fractions of the computed box (render/project.wgsl profile).
fn profile_cell(f: vec3<f32>) -> PointCell {
  let s = cube_scale();
  let l = leaf_at(clamp(f, vec3<f32>(0.0), vec3<f32>(0.999999)) * s);
  return PointCell(leaf_value(l), f32(l.level));
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
    let at = clamp((i0 + o + 0.5) / ncell, vec3<f32>(0.5 / ncell), cube_scale() - 0.5 / ncell);
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
    textureStore(VOL, vec3<i32>(gid), vec4<f32>(display_t(q.x, q.solid), select(0.0, shown, V.cut.y != 0.0), lg, lin_scaled(q.x)));
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
      f = leaf_at(clamp(at, vec3<f32>(0.5 / ncell), cube_scale() - 0.5 / ncell));
    }
    let q = leaf_value(f);
    if (inside && !q.solid) { range_add_w(q.x, 1u << (3u * (P.amr.z - u32(level)))); }
    let shown = select(0.0, 1.0, log(max(q.rho, 1.0e-30)) * 0.4342944819 >= V.cut.x);
    textureStore(DA, atl(blk_origin(slot), l), vec4<f32>(display_t(q.x, q.solid), select(0.0, shown, V.cut.y != 0.0), log10_abs(q.x), lin_scaled(q.x)));
  }
  range_end(lix);
}

// `bake`: the field V.field at each cell of a batch of blocks, from a checkpoint's primitives
// (io/view.ts streamForView), into BKW (the BK above). CH holds each variable for the batch's
// blocks, each block as [z][y][x]; BAKE_IN its layout: [0] blocks, [1] cells per block, [2..12)
// where each of rho, p, v_x, v_y, v_z, the passive scalar, B_x, B_y, B_z and the mass fraction
// shown starts in CH (BAKE_NONE where the batch does not have it), then each block's slot. Fields
// that need neighbors (derivatives, the shock flag) or what a checkpoint does not hold (the
// potential, time bins, burning, radiation, the divergence of B) are not offered for baking.
@group(0) @binding(18) var<storage, read> CH: array<f32>;
@group(0) @binding(19) var<storage, read> BAKE_IN: array<u32>;
@group(0) @binding(21) var BKW: texture_storage_3d<rg32float, write>;
const BAKE_NONE = 0xffffffffu;

fn bake_var(k: u32, i: u32) -> f32 {
  let o = BAKE_IN[2u + k];
  return select(CH[o + i], 0.0, o == BAKE_NONE);
}

@compute @workgroup_size(8, 8, 4)
fn bake(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = BAKE_IN[0];
  let cells = BAKE_IN[1];
  let b = gid.x / u32(BS);
  if (b >= n || gid.y >= u32(BS) || (is2d() && gid.z > 0u) || gid.z >= u32(BS)) { return; }
  let l = vec3<i32>(i32(gid.x % u32(BS)), i32(gid.y), i32(gid.z));
  let slot = i32(BAKE_IN[12u + b]);
  let i = b * cells + (gid.z * u32(BS) + gid.y) * u32(BS) + gid.x % u32(BS);
  g_dx = blk_dx(slot);
  g_acc = P.gext.xyz;
  let rho = max(bake_var(0u, i), P.rho_floor);
  let p = bake_var(1u, i);
  let v = vec3<f32>(bake_var(2u, i), bake_var(3u, i), bake_var(4u, i));
  let eint = eos_eint_from_p(rho, p);
  let u1 = vec4<f32>(rho, rho * v);
  let u2 = vec4<f32>(eint + 0.5 * rho * dot(v, v), eint, rho * bake_var(5u, i), 0.0);
  var g: Grad;
  let level = f32(bt(slot, 0u));
  var x = field_value(V.field, u1, u2, g, level, 0.0);
  if (V.field == F_MASS_FRACTION) { x = bake_var(9u, i); }
  if (V.field >= F_MAGNETIC && V.field < F_MAGNETIC + N_MAGNETIC) {
    x = magnetic_of(V.field, vec3<f32>(bake_var(6u, i), bake_var(7u, i), bake_var(8u, i)), 0.0, p, rho, blk_pos(slot, l));
  }
  textureStore(BKW, bk_texel(slot, l), vec4<f32>(x, rho, 0.0, 0.0));
}

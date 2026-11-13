// Volume lookup for the ray-marcher and the depth pick (the <volume> slot) on a 3D adaptive
// mesh: the leaf blocks at their own resolution, rather than a volume resampled at one.
//
// The display pass (amr/field.wgsl, `atlas`) writes each leaf block's display values into an
// atlas laid out like the state's (amr/grid.wgsl), with one layer of border cells all round
// that hold whatever leaf lies there, at any level. A sample finds the leaf under it through
// the leaf index (one texel per block of the finest level) and the block table, and reads the
// atlas with hardware trilinear filtering: within a block it blends cell centers, and at its
// faces with the border, so the image is continuous across blocks. Memory goes with the cells
// the mesh has, not with the finest resolution cubed.
//
// The ray moves on by a fraction of the cell it is in, so fine regions are sampled finely and
// the coarse rest of the box cheaply. Requires render/camera.wgsl.
struct Look {
  s: vec4<f32>,   // the display atlas's texel, filtered
  cell: f32,      // size of the leaf cell there, in units of the box side
}

struct Atlas {
  a: vec4<u32>,    // (atlas slots per side, root blocks per side, finest level, 0)
  size: vec4<f32>, // atlas texels along x, y, z
}
@group(0) @binding(7) var IDX: texture_3d<u32>;   // leaf slot per block of the finest level
@group(0) @binding(8) var<uniform> A: Atlas;
@group(0) @binding(30) var<storage, read> BT: array<i32>;

// As in amr/grid.wgsl and src/amr/tree.ts.
const ABS = 16;
const APAD = 20;
const ANG = 2;
const ABT_STRIDE = 32u;

// At p, in the box [-0.5, 0.5]^3.
fn vol_look(p: vec3<f32>) -> Look {
  let frac = clamp(p + 0.5, vec3<f32>(0.0), vec3<f32>(0.9999999));
  let m = f32(A.a.y << A.a.z);
  let slot = i32(textureLoad(IDX, vec3<i32>(frac * m), 0).x);
  let b = u32(slot) * ABT_STRIDE;
  let level = u32(BT[b]);
  let coord = vec3<f32>(f32(BT[b + 1u]), f32(BT[b + 2u]), f32(BT[b + 3u]));
  let ncell = f32((A.a.y << level) * u32(ABS));
  // Within the block, in cells: cell i spans [i, i + 1), so its center falls on its texel's.
  let local = frac * ncell - coord * f32(ABS);
  let n = i32(A.a.x);
  let origin = vec3<f32>(vec3<i32>(slot % n, (slot / n) % n, slot / (n * n)) * APAD + ANG);
  return Look(textureSampleLevel(VOL, S, (origin + local) / A.size.xyz, 0.0), 1.0 / ncell);
}

// How far the ray moves past a sample: two thirds of the cell it is in (as the uniform view
// takes 1.5 samples per cell), but no less than the finest step the uniform view would take.
fn vol_step(l: Look) -> f32 {
  return max(l.cell / 1.5, 1.0 / 1536.0);
}

// Most samples along a ray: the box crossed diagonally at the finest step, and some to spare.
const VOL_MAX_SAMPLES = 4096;

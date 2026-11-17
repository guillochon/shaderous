// Layout of the MHD face-flux and edge buffers. A step's three passes run slab by slab along z
// (mhd/mhd.ts), so these buffers hold one slab of layers, not the whole grid: SLAB.z0 is the
// slab's first layer of cells and SLAB.nz how many it has. In 2D there is one slab, the single
// layer. Faces and edges on the planes between two slabs are computed by both, with the same
// pipelines from the same inputs, so both get the same bits.
// Requires P and a binding SLAB.

struct Slab { z0: i32, nz: i32, last: u32, unused: u32 }

// The flux grid: cells t in [-2, n + 1] along x and y, and in [z0 - 2, z0 + nz + 1] along z
// (the single layer in 2D), three vec4 per entry (mhd/flux.wgsl). The edge fields need the faces
// of one row beyond the grid; the second row held the faces Lee & Deane's edge estimates read,
// which were removed (docs/numerics.md section 21), and nothing reads it now.
fn flux_dims() -> vec3<i32> {
  let n = vec3<i32>(P.dims.xyz);
  return vec3<i32>(n.xy + 4, select(SLAB.nz + 4, 1, n.z == 1));
}
// Index offset: the flux grid's first entry is cell (-2, -2, z0 - 2), or (-2, -2, 0) in 2D.
fn flux_first() -> vec3<i32> {
  return vec3<i32>(-2, -2, select(SLAB.z0 - 2, 0, P.dims.z == 1u));
}
// Entry of cell t. Along z in 2D every index is the single layer.
fn flux_index(t: vec3<i32>) -> u32 {
  let d = flux_dims();
  var s = t - flux_first();
  if (P.dims.z == 1u) { s.z = 0; }
  return u32((s.z * d.y + s.y) * d.x + s.x);
}

// The edge buffer (mhd/emf.wgsl): one vec4 per texel t in [0, n] along x and y and in
// [z0, z0 + nz] along z (the single layer in 2D), on the cell's lower edges: (dt/dx) E_x at
// (i, j - 1/2, k - 1/2), (dt/dx) E_y at (i - 1/2, j, k - 1/2), (dt/dx) E_z at
// (i - 1/2, j - 1/2, k): the stage's change of the faces around each edge.
fn emf_dims() -> vec3<i32> {
  let n = vec3<i32>(P.dims.xyz);
  return vec3<i32>(n.xy + 1, select(SLAB.nz + 1, 1, n.z == 1));
}
fn emf_first() -> i32 {
  return select(SLAB.z0, 0, P.dims.z == 1u);
}
fn emf_index(t: vec3<i32>) -> u32 {
  let d = emf_dims();
  var s = t;
  s.z = select(t.z - SLAB.z0, 0, P.dims.z == 1u);
  return u32((s.z * d.y + s.y) * d.x + s.x);
}

// Layers of texels the update covers in this slab: its cells, and on the last slab the grid's
// upper z faces too (texel nz).
fn update_layers() -> i32 {
  if (P.dims.z == 1u) { return 1; }
  return SLAB.nz + select(0, 1, SLAB.last != 0u);
}

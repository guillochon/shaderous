// Face-flux tile for the uniform-grid flux update (hydro/update.wgsl). In the gather form every
// face flux is computed twice, once by each cell that shares the face, and the face work
// (reconstruction, Riemann solve) is nearly all of the update's cost. Here the workgroup computes
// each face of its box of cells once, with flux_between, into workgroup memory; each cell then
// reads its six. A box of 8 x 4 x 4 cells has 464 faces where the gather computes 768.
//
// Bit-identical faces. No flux leaves the pass: a face on the edge of the box is computed again
// by the neighboring workgroup. WGSL lets a compiler fuse and reassociate floating-point
// operations (WGSL 15.7.5), and Metal does: two copies of the same arithmetic, such as a cell's
// lower and upper face in the gather, or flux_between and axis_fluxes, can then differ in the
// last bit, and conservation holds only to that. Here every face that two cells share is
// computed by the same code, in the same compiled pipeline, from the same inputs: within a
// workgroup once, and across a workgroup's edge once on each side by identical instructions.
// The pass is therefore not split into interior and shell pipelines (hydro/region_pass.ts): the
// loads map boundaries at run time instead (g_inner skips the mapping where it is not needed).
// flux_between knows the reflecting walls of the grid (wall_step in gravity/gfield.wgsl), so
// it serves for the faces on the grid's edge too.
// The validation suite checks that every face agrees (DEBUG_FLUX in hydro/update.wgsl).
//
// Measured on an Apple M4 Pro in Chrome: caching the stencil's cell states in workgroup memory
// instead was slower than repeating the texture loads, but computing half as many faces made
// the update nearly twice as fast. Faces beyond the grid's edge (a workgroup that sticks out
// of a grid whose side it does not divide) are computed through the boundary mapping and
// never read.
// Requires hydro/region.wgsl, hydro/flux.wgsl, and P, g_dx and g_dt set.

// Set by hydro/region_pass.ts when the tile fits in the device's workgroup memory.
override TILE: bool = false;
override TILE_FLAT: bool = false;

// Faces along each axis: the box's cells plus one more layer, normal to that axis.
override TF_NX: u32 = (WGX + 1u) * WGY * WGZ;
override TF_NY: u32 = WGX * (WGY + 1u) * WGZ;
override TF_NZ: u32 = select(WGX * WGY * (WGZ + 1u), 0u, TILE_FLAT);
override TF_N: u32 = select(1u, TF_NX + TF_NY + TF_NZ, TILE);
override TF_ND: u32 = select(1u, TF_N, VISCOSITY);

// A Flux in three vec4s: (m, mom), (e, ei, s, vn), and the viscous part d where there is one.
var<workgroup> tf_a: array<vec4<f32>, TF_N>;
var<workgroup> tf_b: array<vec4<f32>, TF_N>;
var<workgroup> tf_d: array<vec4<f32>, TF_ND>;
// The first cell of this workgroup's box.
var<private> tile_o: vec3<i32>;

fn tiled() -> bool {
  return TILE;
}

fn tile_w() -> vec3<i32> {
  return vec3<i32>(i32(WGX), i32(WGY), i32(WGZ));
}

// Entry of the face along `axis` below the cell at offset f from the box's first cell.
fn face_index(axis: i32, f: vec3<i32>) -> i32 {
  let w = tile_w();
  if (axis == 0) { return f.x + (w.x + 1) * (f.y + w.y * f.z); }
  if (axis == 1) { return i32(TF_NX) + f.x + w.x * (f.y + (w.y + 1) * f.z); }
  return i32(TF_NX + TF_NY) + f.x + w.x * (f.y + w.y * f.z);
}

// Compute every face of the workgroup's box, whose first cell is `first`. Every invocation of
// the workgroup must call this (it ends with a barrier); the loop runs the same number of times
// in each, so the barrier is in uniform control flow.
fn tile_fluxes(first: vec3<i32>, li: u32) {
  tile_o = first;
  let w = tile_w();
  let n = WGX * WGY * WGZ;
  for (var k = 0u; k < (TF_N + n - 1u) / n; k++) {
    let i = i32(k * n + li);
    if (i < i32(TF_N)) {
      var axis = 0;
      var f: vec3<i32>;
      if (i < i32(TF_NX)) {
        f = vec3<i32>(i % (w.x + 1), (i / (w.x + 1)) % w.y, i / ((w.x + 1) * w.y));
      } else if (i < i32(TF_NX + TF_NY)) {
        let j = i - i32(TF_NX);
        axis = 1;
        f = vec3<i32>(j % w.x, (j / w.x) % (w.y + 1), j / (w.x * (w.y + 1)));
      } else {
        let j = i - i32(TF_NX + TF_NY);
        axis = 2;
        f = vec3<i32>(j % w.x, (j / w.x) % w.y, j / (w.x * w.y));
      }
      var e = vec3<i32>(0);
      e[axis] = 1;
      // The face between cell a and cell a + e.
      let a = first + f - e;
      g_inner = stencil_inside(a, 2);
      let F = flux_between(a, e, axis);
      tf_a[i] = vec4<f32>(F.m, F.mom);
      tf_b[i] = vec4<f32>(F.e, F.ei, F.s, F.vn);
      if (VISCOSITY) { tf_d[i] = F.d; }
    }
  }
  workgroupBarrier();
}

// The flux through the face along `axis` below cell c, which must be in this workgroup's box or
// its upper neighbor along `axis`.
fn tile_flux(axis: i32, c: vec3<i32>) -> Flux {
  let i = face_index(axis, c - tile_o);
  let a = tf_a[i];
  let b = tf_b[i];
  var d = vec4<f32>(0.0);
  if (VISCOSITY) { d = tf_d[i]; }
  return Flux(a.x, a.yzw, b.x, b.y, b.z, b.w, d);
}

fn tile_faces(axis: i32, c: vec3<i32>, e: vec3<i32>) -> Faces {
  return Faces(tile_flux(axis, c), tile_flux(axis, c + e));
}

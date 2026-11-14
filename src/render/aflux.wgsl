// The flux function A_z of a 2D run (B_x = dA/dy, B_y = -dA/dx), for its field lines
// (render/fieldlines.ts, drawn by render/fluxfunction.wgsl), integrated from the face field
// itself: A_z at a corner is the flux through a path from the box's corner, along the bottom
// row (`row`, one invocation) and then up its column (`cols`). The face field is divergence-free
// to round-off, so the flux does not depend on the path, and neighboring columns agree. (Integrated
// from a resampled cell-centered field instead, each column drifts on its own and the contours
// come out jagged.)
//
// A_z is kept at the corners of an M x M grid, each of its edges s cells of the finest level
// long (M s = n, the finest cells per side), in units of the box side. The <aface> slot gives
// the face field at the finest level's faces: render/aface_uniform.wgsl reads the grid's own;
// amr/aface.wgsl a leaf's, between the faces of a coarse cell interpolated linearly (normal
// component along its normal), which keeps the field divergence-free inside it.
#include "hydro/common.wgsl"
#include <aface>

@group(0) @binding(17) var<storage, read_write> A: array<f32>;
// The least and greatest A_z, as ordered keys (render/fluxfunction.wgsl, unkey()).
@group(0) @binding(18) var<storage, read_write> AR: array<atomic<u32>, 2>;
struct Grid { m: u32, s: u32, unused0: u32, unused1: u32 }
@group(0) @binding(19) var<uniform> G: Grid;

fn key(f: f32) -> u32 {
  let b = bitcast<u32>(f);
  return select(~b, b | 0x80000000u, (b & 0x80000000u) == 0u);
}
fn note(a: f32) {
  atomicMin(&AR[0], key(a));
  atomicMax(&AR[1], key(a));
}

@compute @workgroup_size(1)
fn row() {
  let h = 1.0 / f32(G.m * G.s);
  var a = 0.0;
  A[0] = 0.0;
  note(0.0);
  for (var i = 0u; i < G.m; i++) {
    for (var k = 0u; k < G.s; k++) { a -= by_face(i * G.s + k, 0u) * h; }
    A[i + 1u] = a;
    note(a);
  }
}

@compute @workgroup_size(64)
fn cols(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i > G.m) { return; }
  let h = 1.0 / f32(G.m * G.s);
  let w = G.m + 1u;
  var a = A[i];
  for (var j = 0u; j < G.m; j++) {
    for (var k = 0u; k < G.s; k++) { a += bx_face(i * G.s, j * G.s + k) * h; }
    A[(j + 1u) * w + i] = a;
    note(a);
  }
}

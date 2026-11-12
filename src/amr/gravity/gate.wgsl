// Whether the next V-cycle runs: it does if the residual that amr/gravity/residual.wgsl just
// measured is above TOL times the largest source term and the last cycle cut it by at least
// PROGRESS (so it is not at its floor), and then ARGS holds the sizes of all
// its dispatches over the blocks in the list; otherwise every size is zero and the cycle's
// dispatches do nothing. The decision stays on the GPU: nothing is read back.
//
// ARGS, one (x, y, z) triple each, as AmrGravity.cells(), color() and faces() size them:
//   0..4   every cell of a block at sublevels of 16, 8, 4, 2 and 1 cells per side
//   5..9   one color of a red-black sweep at the same sublevels
//   10     the face ghosts of a block
// STATS: [0] cycles run beyond the first, [1] the last relative residual measured (as float
// bits), [2] how many residuals were measured, [3] the one before [1] (for inspection).
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"

@group(0) @binding(1) var<storage, read_write> RES: array<u32, 4>;
@group(0) @binding(2) var<storage, read_write> ARGS: array<u32, 33>;
@group(0) @binding(3) var<storage, read_write> STATS: array<u32, 4>;

override TOL: f32 = 1e-4;
// A cycle runs only if the last one cut the residual at least this much: once it stops falling
// it has reached the floor that the boundary values and rounding set, which another cycle
// does not lower.
override PROGRESS: f32 = 0.5;
// Only record the residual (measured before the first cycle, which always runs).
override RECORD: bool = false;

fn put(i: u32, x: u32, y: u32, z: u32) {
  ARGS[3u * i] = x;
  ARGS[3u * i + 1u] = y;
  ARGS[3u * i + 2u] = z;
}

@compute @workgroup_size(1)
fn main() {
  let r = bitcast<f32>(RES[0]);
  let f = bitcast<f32>(RES[1]);
  // Ready for the next measurement.
  RES[0] = 0u;
  RES[1] = 0u;
  let measured = f > 0.0;
  let rel = select(0.0, r / f, measured);
  let prev = bitcast<f32>(STATS[1]);
  if (measured) {
    STATS[3] = STATS[1];
    STATS[1] = bitcast<u32>(rel);
    STATS[2] += 1u;
  }
  if (RECORD) { return; }
  let run = measured && rel > TOL && rel < PROGRESS * prev;
  let n = select(0u, LIST[0], run);
  if (run) { STATS[0] += 1u; }
  for (var k = 0u; k < 5u; k++) {
    let b = 16u >> k;
    let span = max(b, 8u);
    let gy = (b + 7u) / 8u;
    let gz = (b + 3u) / 4u;
    put(k, n * (span / 8u), select(0u, gy, run), select(0u, gz, run));
    put(5u + k, (n * max(b / 2u, 1u) + 7u) / 8u, select(0u, gy, run), select(0u, gz, run));
  }
  put(10u, select(0u, 48u, run), n, 1u);
}

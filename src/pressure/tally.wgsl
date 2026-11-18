// Where a solve ended: the residual pressure/measure.wgsl just measured, relative to the largest
// right-hand side, against the tolerance. Kept on the GPU and read back only on request.
// TALLY: [0] the tolerance (float bits, written by the CPU), [1] the last relative residual
// (float bits), [2] solves finished, [3] solves that ended above the tolerance.
//
// The number of V-cycles is fixed (pressure/multigrid.ts). Sizing later cycles from this
// measurement by indirect dispatch, as the block gravity solver does, was tried and cost more
// than it saved: a cycle the gate turned down still paid for its passes, and on these grids the
// passes, not the cells, are most of a cycle's cost (a 128^2 cavity ran at half the speed).
@group(0) @binding(0) var<storage, read_write> RES: array<u32, 2>;
@group(0) @binding(1) var<storage, read_write> TALLY: array<u32, 4>;

@compute @workgroup_size(1)
fn main() {
  let r = bitcast<f32>(RES[0]);
  let f = bitcast<f32>(RES[1]);
  RES[0] = 0u;
  RES[1] = 0u;
  let rel = select(0.0, r / f, f > 0.0);
  TALLY[1] = bitcast<u32>(rel);
  TALLY[2] += 1u;
  if (rel > bitcast<f32>(TALLY[0])) { TALLY[3] += 1u; }
}

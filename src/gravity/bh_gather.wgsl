// The point masses are summed over the gas in groups of four (gravity/mass_sums.ts): this
// copies one group's sums, as its reduction left them, to their place among every mass's.
struct Group { base: vec4<u32> }   // x: the group's first mass

@group(0) @binding(0) var<storage, read> IN: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(2) var<uniform> G: Group;

@compute @workgroup_size(1)
fn main() {
  // The gas's mass is the same in every group's sums: the first group's is kept.
  if (G.base.x == 0u) { OUT[0] = IN[0]; }
  for (var c = 0u; c < 40u; c++) {
    OUT[1u + 10u * G.base.x + c] = IN[1u + c];
  }
}

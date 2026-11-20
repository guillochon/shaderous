// Taking point masses out of the run (Simulation.removeMasses): those that stay are moved up
// to close the gaps, in their order. The Masses block (gravity/bh.wgsl) as plain floats: the
// frame's acceleration and a followed frame's state (12), then 40 per mass.
struct Keep { mask: vec4<u32> }   // x: bit i is set if mass i stays

@group(0) @binding(0) var<storage, read_write> BH: array<f32>;
@group(0) @binding(1) var<uniform> K: Keep;

@compute @workgroup_size(1)
fn main() {
  var n = 0u;
  for (var i = 0u; i < 32u; i++) {
    if (((K.mask.x >> i) & 1u) == 0u) { continue; }
    if (n != i) {
      for (var c = 0u; c < 40u; c++) { BH[12u + 40u * n + c] = BH[12u + 40u * i + c]; }
    }
    n++;
  }
}

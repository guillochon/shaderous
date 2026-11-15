// Hierarchical reduction template. Each invocation folds a block of RED_F^3 inputs into one
// output record, so sums are accumulated in a tree rather than one long float32 chain.
//
// <red_def> supplies:    const RED_K: u32, alias Red (RED_K floats), and red_identity(),
//                        red_combine(), red_store() and (later stages) red_load(), written
//                        out channel by channel (gpu/reduce.ts: kept out of local memory)
// <red_source> supplies: fn red_cell(c: vec3<i32>) -> Red, and its own input bindings (< 20)
#include <red_def>

override NINX: u32 = 1u;   // inputs along each axis
override NINY: u32 = 1u;
override NINZ: u32 = 1u;
override NOUTX: u32 = 1u;  // outputs along each axis = ceil(inputs / RED_F)
override NOUTY: u32 = 1u;
override NOUTZ: u32 = 1u;
override RED_F: u32 = 4u;  // block side
// A long x axis needs more workgroups than a dispatch may have along one axis: it is then cut
// into slabs of RED_XG workgroups, stacked along z after the RED_ZG workgroups z itself needs.
override RED_XG: u32 = 65535u;
override RED_ZG: u32 = 1u;

@group(0) @binding(20) var<storage, read_write> RED_OUT: array<f32>;

#include <red_source>

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let gid = vec3<u32>(wid.x + (wid.z / RED_ZG) * RED_XG, wid.y, wid.z % RED_ZG) * 4u + lid;
  if (any(gid >= vec3<u32>(NOUTX, NOUTY, NOUTZ))) { return; }
  var acc = red_identity();
  for (var z = 0u; z < RED_F; z++) {
    for (var y = 0u; y < RED_F; y++) {
      for (var x = 0u; x < RED_F; x++) {
        let c = gid * RED_F + vec3<u32>(x, y, z);
        if (all(c < vec3<u32>(NINX, NINY, NINZ))) {
          acc = red_combine(acc, red_cell(vec3<i32>(c)));
        }
      }
    }
  }
  red_store(((gid.z * NOUTY + gid.y) * NOUTX + gid.x) * RED_K, acc);
}

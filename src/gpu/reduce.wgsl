// Hierarchical reduction template. Each invocation folds a block of RED_F^3 inputs into one
// output record, so sums are accumulated in a tree rather than one long float32 chain.
//
// <red_def> supplies:    const RED_K: u32, alias Red (RED_K floats), and red_identity(),
//                        red_combine(), red_store() and (later stages) red_load(), written
//                        out channel by channel (gpu/reduce.ts: kept out of local memory)
// <red_source> supplies: fn red_cell(c: vec3<i32>) -> Red, and its own input bindings (< 20)
#include <red_def>

// The stage's grid, in a uniform rather than override constants, so the same pipeline serves
// every resolution and block count (a new size used to mean new pipelines, half a second each).
// A long x axis needs more workgroups than a dispatch may have along one axis: it is then cut
// into slabs of `xg` workgroups, stacked along z after the `zg` workgroups z itself needs.
struct RedDims {
  nin: vec3<u32>,  // inputs along each axis
  xg: u32,
  nout: vec3<u32>, // outputs along each axis = ceil(inputs / RED_F)
  zg: u32,
}
@group(0) @binding(40) var<uniform> RD: RedDims;
override RED_F: u32 = 4u;  // block side

@group(0) @binding(20) var<storage, read_write> RED_OUT: array<f32>;

#include <red_source>

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let gid = vec3<u32>(wid.x + (wid.z / RD.zg) * RD.xg, wid.y, wid.z % RD.zg) * 4u + lid;
  if (any(gid >= RD.nout)) { return; }
  var acc = red_identity();
  for (var z = 0u; z < RED_F; z++) {
    for (var y = 0u; y < RED_F; y++) {
      for (var x = 0u; x < RED_F; x++) {
        let c = gid * RED_F + vec3<u32>(x, y, z);
        if (all(c < RD.nin)) {
          acc = red_combine(acc, red_cell(vec3<i32>(c)));
        }
      }
    }
  }
  red_store(((gid.z * RD.nout.y + gid.y) * RD.nout.x + gid.x) * RED_K, acc);
}

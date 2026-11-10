// Hierarchical reduction template. Each invocation folds a block of RED_F^3 inputs into one
// output record, so sums are accumulated in a tree rather than one long float32 chain.
//
// <red_def> supplies:    const RED_K: u32, and RED_MAX, RED_MIN: bit i set if channel i is a
//                        maximum / minimum (otherwise it is a sum)
// <red_source> supplies: fn red_cell(c: vec3<i32>) -> Red, and its own input bindings (< 20)
#include <red_def>

alias Red = array<f32, RED_K>;

// 0 sum, 1 max, 2 min.
fn red_op(i: u32) -> u32 {
  return ((RED_MAX >> i) & 1u) + 2u * ((RED_MIN >> i) & 1u);
}

override NINX: u32 = 1u;   // inputs along each axis
override NINY: u32 = 1u;
override NINZ: u32 = 1u;
override NOUTX: u32 = 1u;  // outputs along each axis = ceil(inputs / RED_F)
override NOUTY: u32 = 1u;
override NOUTZ: u32 = 1u;
override RED_F: u32 = 4u;  // block side

@group(0) @binding(20) var<storage, read_write> RED_OUT: array<f32>;

#include <red_source>

fn red_identity() -> Red {
  var r: Red;
  for (var i = 0u; i < RED_K; i++) {
    r[i] = 0.0;
    if (red_op(i) == 1u) { r[i] = -3.0e38; }
    if (red_op(i) == 2u) { r[i] = 3.0e38; }
  }
  return r;
}

fn red_combine(a: Red, b: Red) -> Red {
  var r: Red;
  var aa = a;
  var bb = b;
  for (var i = 0u; i < RED_K; i++) {
    let op = red_op(i);
    if (op == 0u) { r[i] = aa[i] + bb[i]; }
    else if (op == 1u) { r[i] = max(aa[i], bb[i]); }
    else { r[i] = min(aa[i], bb[i]); }
  }
  return r;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
  let base = ((gid.z * NOUTY + gid.y) * NOUTX + gid.x) * RED_K;
  for (var i = 0u; i < RED_K; i++) {
    RED_OUT[base + i] = acc[i];
  }
}

// Reduction source for stages after the first: records written by the previous stage.
@group(0) @binding(0) var<storage, read> RED_IN: array<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  let base = ((u32(c.z) * NIN + u32(c.y)) * NIN + u32(c.x)) * RED_K;
  var r: Red;
  for (var i = 0u; i < RED_K; i++) {
    r[i] = RED_IN[base + i];
  }
  return r;
}

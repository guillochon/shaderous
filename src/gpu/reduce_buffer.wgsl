// Reduction source for stages after the first: records written by the previous stage.
@group(0) @binding(0) var<storage, read> RED_IN: array<f32>;

fn red_cell(c: vec3<i32>) -> Red {
  return red_load(((u32(c.z) * NINY + u32(c.y)) * NINX + u32(c.x)) * RED_K);
}

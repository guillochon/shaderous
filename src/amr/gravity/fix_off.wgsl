// No uniform acceleration taken off self-gravity's field (amr/gravity/fix_on.wgsl): the point
// masses' field, and runs that do not keep it.
fn fix_accel() -> vec3<f32> {
  return vec3<f32>(0.0);
}

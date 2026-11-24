// The uniform acceleration taken off self-gravity's field, its potential (a . x) put on
// (amr/gravity.ts AmrGravity.fix, kept by amr/amr_sim.ts encodeSelfForce): the net force of the
// gas's gravity on itself, per unit mass, as the last step left it.
@group(0) @binding(8) var<storage, read> FIX: vec4<f32>;

fn fix_accel() -> vec3<f32> {
  return FIX.xyz;
}

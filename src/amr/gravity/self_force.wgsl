// Self-gravity's net force on the gas, kept off its field (amr/amr_sim.ts encodeSelfForce): the
// sums over the leaf cells of the field just written (amr/self_force.wgsl: m, then m g) add
// what is left of the net force per unit mass to the uniform acceleration the field pass takes
// off (FIX, amr/gravity/fix_on.wgsl). The field held g - a, so a becomes the whole net force of
// this step, which the next step's field takes off. Along a plane of mirror symmetry the image's
// pull is real and stays: MASK holds the axes kept free of a net force.
override MASK: u32 = 7u;

@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> FIX: vec4<f32>;

@compute @workgroup_size(1)
fn main() {
  if (!(S[0] > 0.0)) { return; }
  let a = FIX.xyz + vec3<f32>(S[1], S[2], S[3]) / S[0];
  let keep = vec3<bool>((MASK & 1u) != 0u, (MASK & 2u) != 0u, (MASK & 4u) != 0u);
  FIX = vec4<f32>(select(vec3<f32>(0.0), a, keep), 0.0);
}

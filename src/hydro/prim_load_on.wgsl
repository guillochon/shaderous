// Stored primitives on the uniform grid (primStored in src/sim.ts): the timestep-rates pass
// (hydro/rates.wgsl, through hydro/prim_store_on.wgsl) writes what load_q returns for each cell
// of the current state, and load_q (hydro/recon.wgsl) reads it here instead of converting the
// conserved state again. A cell is otherwise converted at every stencil load, some twenty times
// a step in 3D; with an equation of state that solves for its temperature (Helmholtz) that was
// most of the step. The values are the conversion's own, so the scheme does not change.
// (Off, hydro/recon.wgsl gets the stubs of DEFAULT_SLOTS in gpu/shaders.ts.)
const PRIM_STORED = true;
@group(0) @binding(61) var PW1: texture_3d<f32>;
@group(0) @binding(62) var PW2: texture_3d<f32>;
fn prim_a(c: vec3<i32>) -> vec4<f32> { return textureLoad(PW1, c, 0); }
fn prim_b(c: vec3<i32>) -> vec4<f32> { return textureLoad(PW2, c, 0); }

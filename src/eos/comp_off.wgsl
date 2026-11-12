// One composition everywhere (species/mixture.ts): the equation of state takes its constants
// from P alone. Everything here folds away, so a run built with it compiles to what it did
// before the composition could vary. (A run with tracer species gets this with COMP_NG set to
// its number of groups: compSlot() in species/mixture.ts.)
const COMP = false;
const COMP_NG: u32 = 1u;
const COMP_ZH1: f32 = 2.0;
var<private> g_comp: vec4<f32>;
fn comp_use(k: vec4<f32>) {}
fn comp_tex(c: vec3<i32>) -> vec4<f32> { return vec4<f32>(1.0, 1.0, 0.0, 0.0); }
fn comp_store(c: vec3<i32>, k: vec4<f32>) {}
fn comp_moments(u: array<vec4<f32>, COMP_NG>) -> vec4<f32> { return vec4<f32>(1.0, 1.0, 0.0, 0.0); }
fn comp_jump(a: vec4<f32>, b: vec4<f32>) -> f32 { return 0.0; }
fn comp_first_order(c: vec3<i32>) -> f32 { return 0.0; }

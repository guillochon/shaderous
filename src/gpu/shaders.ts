// Registry of every .wgsl file under src/, keyed by path relative to src/ ("hydro/update.wgsl").
import { preprocess, type Slots } from './preprocess';

const raw = import.meta.glob('../**/*.wgsl', { query: '?raw', import: 'default', eager: true }) as Record<
  string,
  string
>;

export const shaderFiles: Record<string, string> = {};
// Glob keys are relative to this file: "../hydro/x.wgsl", but "./x.wgsl" for files in gpu/.
for (const [k, v] of Object.entries(raw)) shaderFiles[k.replace(/^\.\.\//, '').replace(/^\.\//, 'gpu/')] = v;

// Slots most shaders never set: the composition (species/mixture.ts) is uniform unless a run's
// species say otherwise.
const DEFAULT_SLOTS: Slots = {
  comp: 'eos/comp_off.wgsl',
  // A spherical run's geometry is compiled in only where it is one (hydroSlots), so that every
  // other run's shaders are what they were before it, bit for bit (hydro/common.wgsl).
  geometry: 'const SPHERICAL = false;\n',
  comp_src: 'amr/comp_src_off.wgsl',
  debris: 'debris/debris_off.wgsl',
  // An inertial frame (gravity/frame_off.wgsl) unless hydroSlots says otherwise.
  frame: 'gravity/frame_off.wgsl',
  // The pressure in the explicit fluxes unless hydroSlots says implicit (hydro/predictor.wgsl).
  psplit: 'const PSPLIT = false;\nconst PSPLIT_LAG = false;\nconst PSPLIT_CENTER = false;\n',
  psplit_grad: 'pressure/predict_off.wgsl',
  psplit_store: 'pressure/store_off.wgsl',
  // Forces on bodies (plans/wind-tunnel.md 4.2): summed only where a run has bodies (walls_on).
  walls: 'hydro/walls_off.wgsl',
  amr_walls: 'amr/walls_off.wgsl',
  pwalls: 'pressure/pwalls_off.wgsl',
  ap_walls: 'amr/pressure/pwalls_off.wgsl',
  ap_store: 'amr/pressure/store_off.wgsl',
  ap_lag: 'amr/pressure/lag_liquid.wgsl',
  // No limit on the block grid's step from the solved pressure, unless AmrSim says a gas has one.
  ap_acc: 'const AP_ACC = false;\nfn ap_pressure_accel(c: vec3<i32>, rho: f32, dx: f32) -> vec3<f32> { return vec3<f32>(0.0); }\n',
  volume: 'render/volume_uniform.wgsl',
  flag_species: 'fn species_indicator(c: vec3<i32>) -> f32 { return 0.0; }\n',
  flag_burn: 'fn burn_indicator(c: vec3<i32>, dx: f32) -> f32 { return 0.0; }\n',
  // Face mass flux records in one window (amr/species_index.wgsl) unless AmrSim splits them.
  mf_load: 'fn mf_load(i: u32) -> f32 { return MF[i]; }\n',
  mf_store: 'fn mf_store(i: u32, v: f32) { MF[i] = v; }\n',
  mf_add: 'fn mf_add(i: u32, v: f32) { MF[i] = MF[i] + v; }\n',
};

export function assemble(entry: string, slots: Slots = {}): string {
  return preprocess(entry, shaderFiles, { ...DEFAULT_SLOTS, ...slots });
}

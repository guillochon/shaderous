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
  comp_src: 'amr/comp_src_off.wgsl',
  debris: 'debris/debris_off.wgsl',
  volume: 'render/volume_uniform.wgsl',
  flag_species: 'fn species_indicator(c: vec3<i32>) -> f32 { return 0.0; }\n',
  flag_burn: 'fn burn_indicator(c: vec3<i32>, dx: f32) -> f32 { return 0.0; }\n',
};

export function assemble(entry: string, slots: Slots = {}): string {
  return preprocess(entry, shaderFiles, { ...DEFAULT_SLOTS, ...slots });
}

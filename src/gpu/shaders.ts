// Registry of every .wgsl file under src/, keyed by path relative to src/ ("hydro/update.wgsl").
import { preprocess, type Slots } from './preprocess';

const raw = import.meta.glob('../**/*.wgsl', { query: '?raw', import: 'default', eager: true }) as Record<
  string,
  string
>;

export const shaderFiles: Record<string, string> = {};
// Glob keys are relative to this file: "../hydro/x.wgsl", but "./x.wgsl" for files in gpu/.
for (const [k, v] of Object.entries(raw)) shaderFiles[k.replace(/^\.\.\//, '').replace(/^\.\//, 'gpu/')] = v;

export function assemble(entry: string, slots: Slots = {}): string {
  return preprocess(entry, shaderFiles, slots);
}

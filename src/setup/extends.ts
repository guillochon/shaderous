// `extends`: a setup that gives only its differences from a parent. Objects merge key by key;
// arrays (the parts, the featured options) and everything else replace the parent's.
import { cloneSetup, type SetupSpec } from './spec';

const isObject = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);

function merge(parent: unknown, child: unknown): unknown {
  if (!isObject(parent) || !isObject(child)) return child;
  const out: Record<string, unknown> = { ...parent };
  for (const [k, v] of Object.entries(child)) {
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') continue;
    out[k] = k in parent ? merge(parent[k], v) : v;
  }
  return out;
}

/**
 * The whole document a setup stands for: its parents' content with its own laid over it, and
 * no `extends` left. `find` returns a parent by id (already whole or not), or undefined.
 */
export function resolveExtends(spec: SetupSpec, find: (id: string) => SetupSpec | undefined, depth = 0): SetupSpec {
  if (spec.extends === undefined) return spec;
  if (depth > 8) throw new Error('setup: "extends" is nested too deep, or in a loop');
  const parent = find(spec.extends);
  if (!parent) throw new Error(`setup: extends "${spec.extends}", which this build does not have`);
  const whole = resolveExtends(parent, find, depth + 1);
  const { extends: _drop, ...own } = spec;
  return merge(cloneSetup(whole), cloneSetup(own)) as SetupSpec;
}

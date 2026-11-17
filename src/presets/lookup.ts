// A problem by its id, wherever it comes from: a preset's file, or the one custom setup the
// page holds (presets/custom.ts).
import type { Problem } from '../problems/types';
import { CUSTOM_ID, customSetup } from './custom';
import { FAMILIES, PRESET_FILES, preset, type ProblemFamily } from './index';

/** Whether this build can run a problem of this id now. */
export const hasProblem = (id: string | null | undefined): boolean => !!id && (id in PRESET_FILES || (id === CUSTOM_ID && !!customSetup()));

export function problemOf(id: string): Problem {
  const c = id === CUSTOM_ID ? customSetup() : null;
  return c ? c.problem : preset(id).problem;
}

/** The menu entry a problem belongs to: a preset's family, or the custom setup alone. */
export function familyFor(id: string): ProblemFamily {
  const c = id === CUSTOM_ID ? customSetup() : null;
  if (c) return { id: CUSTOM_ID, label: c.problem.label, variants: [{ label: 'Standard', problem: c.problem }] };
  const f = FAMILIES.find((q) => q.variants.some((v) => v.problem.id === id));
  if (!f) throw new Error(`unknown problem: ${id}`);
  return f;
}

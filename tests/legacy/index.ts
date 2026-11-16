// The hand-written problems the preset files replace, by id: what tests/unit/presets.test.ts
// compares every compiled preset with.
import { PROBLEMS } from '../../src/problems/index';
import type { Problem } from '../../src/problems/types';

export const LEGACY: Record<string, Problem> = Object.fromEntries(PROBLEMS.map((p) => [p.id, p]));

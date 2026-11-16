// The hand-written problems the preset files replaced, by id: what tests/unit/presets.test.ts
// compares every compiled preset with.
import { PROBLEMS } from './menu';
import type { Problem } from '../../src/problems/types';

export const LEGACY: Record<string, Problem> = Object.fromEntries(PROBLEMS.map((p) => [p.id, p]));

import { describe, expect, it } from 'vitest';
import { checkpointIndex, nextCheckpoint } from '../../src/io/checkpoints';

describe('checkpoint times', () => {
  it('is the next multiple of the interval after t', () => {
    expect(nextCheckpoint(0, 0.1)).toBeCloseTo(0.1, 12);
    expect(nextCheckpoint(0.25, 0.1)).toBeCloseTo(0.3, 12);
    expect(nextCheckpoint(3, 1)).toBe(4);
  });

  it('moves past a clock held at a checkpoint, either side of it in f32', () => {
    const t = Math.fround(0.3);
    expect(nextCheckpoint(t, 0.1)).toBeCloseTo(0.4, 6);
    expect(nextCheckpoint(0.3 * (1 - 1e-7), 0.1)).toBeCloseTo(0.4, 6);
    expect(checkpointIndex(t, 0.1)).toBe(3);
  });
});

import { beforeAll, describe, expect, it } from 'vitest';
import { defaultEosParams, eosForMoments, makeEos } from '../../src/eos/eos';
import { loadHelmholtz } from '../../src/eos/helmholtz';

beforeAll(async () => {
  await loadHelmholtz();
});

describe('eintFromP remembers its last answers (eos/eos.ts rememberEint)', () => {
  it('gives the same internal energy as a fresh equation of state, whatever the order of the questions', () => {
    const p = defaultEosParams();
    const a = makeEos('helmholtz', p);
    const qs: [number, number][] = [];
    for (let i = 0; i < 40; i++) {
      // Runs of one state (ambient gas) broken by others (a star's cells), and returns to each.
      const r = i % 7 < 3 ? 1e-6 : 0.5 + 0.1 * (i % 5);
      qs.push([r, r < 1e-3 ? 1e-9 : 0.2 * r ** 1.6]);
    }
    for (const [r, pr] of qs) expect(a.eintFromP(r, pr)).toBe(makeEos('helmholtz', p).eintFromP(r, pr));
  });

  it('keeps one equation of state per composition', () => {
    const p = defaultEosParams();
    const of = eosForMoments('helmholtz', p);
    const k1 = [0.5, 1 / 12, 6, 36];
    const k2 = [0.5, 1 / 4, 2, 4];
    expect(of(k1)).toBe(of([...k1]));
    expect(of(k1)).not.toBe(of(k2));
    expect(of(k2).eintFromP(1, 0.3)).toBe(makeEos('helmholtz', p, k2).eintFromP(1, 0.3));
  });
});

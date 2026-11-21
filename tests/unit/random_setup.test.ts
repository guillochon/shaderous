// The random setup generator (tools/random-test/generate.ts): every document it makes passes the
// page's check, the same seed makes the same document, and every feature of the option table can
// be put into a document. An option added to the table that the generator cannot place fails the
// last test: give it a hint, a hook or a fixer there (or list it in EXCLUDED, with why).
// `npm run random-test -- --coverage 3000` reports how often the seeds reach each feature.
import { describe, expect, it } from 'vitest';
import { attempt, errorsOf, features, generate } from '../../tools/random-test/generate';

const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
const SEEDS = Number(env.RANDOM_SETUP_SEEDS ?? 200);

describe('random setups', () => {
  const gens = Array.from({ length: SEEDS }, (_, i) => generate(1000 + i));

  it('makes documents that pass the page’s check', () => {
    const bad = gens.filter((g) => errorsOf(g.doc).length).map((g) => `${g.seed}: ${errorsOf(g.doc)[0].text}`);
    expect(bad).toEqual([]);
  });

  it('makes the same document from the same seed', () => {
    for (const s of [1001, 1002, 1077]) expect(JSON.stringify(generate(s).doc)).toBe(JSON.stringify(gens[s - 1000].doc));
  });

  it('can put every feature of the option table into a document', () => {
    const never: string[] = [];
    for (const f of features()) {
      let placed = false;
      for (const seed of [1, 3, 5, 7, 9, 11, 13, 15]) if ((placed = attempt(f.id, [seed])[0].ok)) break;
      if (!placed) never.push(`${f.id}: ${attempt(f.id, [1])[0].why}`);
    }
    expect(never).toEqual([]);
  }, 120_000);
});

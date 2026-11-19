// The sound speed of the ANEOS tables (src/eos/aneos.ts) against the published tables' own, at
// their nodes. The published files are 40 MB each and are not in the repository, so this runs
// only when ANEOS_PUBLISHED names a folder that holds them:
//
//   <folder>/fe85si15/NEW-SESAME-EXT.TXT   github.com/ststewart/aneos-Fe85Si15-2020 (doi:10.5281/zenodo.3866550)
//   <folder>/pyrolite/NEW-SESAME-EXT.TXT   doi:10.5281/zenodo.6968539 (aneos-pyrolite-2022 v0.2.0)
//
//   ANEOS_PUBLISHED=<folder> npx vitest run tests/unit/aneos_published.test.ts
//
// It writes <folder>/sound_speed.txt. Compared: condensed matter (ANEOS's phase flag solid or
// liquid) at or above the reference density. Below it the published solid is under tension and
// ours is in balance with its vapor (NOTENSION, tools/eos/aneos_table.py), so they differ by design.
import { beforeAll, describe, expect, it } from 'vitest';
import { ANEOS_RHO_MAX, ANEOS_RHO_MIN, ANEOS_T_MAX, ANEOS_T_MIN, aneosFromT, loadAneos, type AneosTable } from '../../src/eos/aneos';

declare const process: { env: Record<string, string | undefined> };
// The project has no Node types (as tests/unit/generated.test.ts): what this file uses of node:fs.
const { existsSync, readFileSync, writeFileSync }: { existsSync(p: string): boolean; readFileSync(p: string, enc: string): string; writeFileSync(p: string, text: string): void } = await import(/* @vite-ignore */ 'node:fs' as string);
const DIR = process.env.ANEOS_PUBLISHED;
const MATERIALS = ['fe85si15', 'pyrolite'] as const;
const SOLID = 4;
const LIQUID = 6;
const q = (a: number[], f: number) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];

describe.skipIf(!DIR || !MATERIALS.every((m) => existsSync(`${DIR}/${m}/NEW-SESAME-EXT.TXT`)))('ANEOS sound speed against the published tables', () => {
  const tab = {} as Record<string, AneosTable>;
  beforeAll(async () => {
    for (const m of MATERIALS) tab[m] = await loadAneos(m);
  }, 120000);

  it('agrees at the nodes of condensed matter', () => {
    const report: string[] = [];
    for (const m of MATERIALS) {
      // SESAME text: a 201 record (the reference density is its third word), then the 301 record:
      // nr, nt, the densities, the temperatures, and four arrays of nr * nt (entropy, sound speed
      // in cm/s, heat capacity, phase flag), the density running fastest.
      const lines = readFileSync(`${DIR}/${m}/NEW-SESAME-EXT.TXT`, 'utf8').split('\n');
      const rho0 = Number(lines[lines.findIndex((l) => l.includes('TYPE =  201')) + 1].trim().split(/\s+/)[2]);
      const w: number[] = [];
      for (let i = lines.findIndex((l) => l.includes('TYPE =  301')) + 1; i < lines.length; i++) for (const t of lines[i].match(/[-+ ]\d\.\d{8}E[-+]\d\d/g) ?? []) w.push(Number(t));
      const [nr, nt] = w;
      const rho = w.slice(2, 2 + nr);
      const T = w.slice(2 + nr, 2 + nr + nt);
      const o = 2 + nr + nt;
      const n = nr * nt;
      const err: number[] = [];
      for (let j = 0; j < nt; j++) {
        if (T[j] < ANEOS_T_MIN * 1.01 || T[j] > ANEOS_T_MAX * 0.99) continue;
        for (let i = 0; i < nr; i++) {
          if (rho[i] < Math.max(rho0, ANEOS_RHO_MIN) || rho[i] > ANEOS_RHO_MAX * 0.99) continue;
          const c = w[o + n + j * nr + i];
          const phase = Math.round(w[o + 3 * n + j * nr + i]);
          if (!(c > 0) || (phase !== SOLID && phase !== LIQUID)) continue;
          err.push(Math.abs(Math.sqrt(aneosFromT(tab[m], rho[i], T[j]).c2) - c) / c);
        }
      }
      const over = err.filter((x) => x > 0.01).length / err.length;
      report.push(`${m}: ${err.length} nodes at rho >= ${rho0} g/cm^3; relative difference: median ${q(err, 0.5).toExponential(2)}, 90th percentile ${q(err, 0.9).toExponential(2)}, 99th ${q(err, 0.99).toExponential(2)}; over 1% at ${(100 * over).toFixed(1)}% of them`);
      expect(q(err, 0.5)).toBeLessThan(1e-5);
      expect(q(err, 0.9)).toBeLessThan(1e-3);
      expect(over).toBeLessThan(0.05);
    }
    writeFileSync(`${DIR}/sound_speed.txt`, report.join('\n') + '\n');
  }, 600000);
});

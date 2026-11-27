import { describe, expect, it } from 'vitest';
import { NET_DATA } from '../../src/burn/aprox13_net';
import { NET_REACTIONS } from '../../src/burn/aprox13_reactions';
import { netEnergyRate, netReactionRates, netUnits, NET_SPECIES } from '../../src/burn/network';
import { cellReactions, reactionEdges } from '../../src/ui/burn_graph';

const terms = NET_DATA.terms as readonly { powers: readonly (readonly number[])[]; nrho: number; coef: number }[];

describe('the network reactions table', () => {
  it('lines up with the network terms', () => {
    expect(NET_REACTIONS.length).toBe(terms.length);
    NET_REACTIONS.forEach((r, t) => {
      // The reactants are the term's powers.
      const used = r.nu.filter(([, n]) => n < 0).map(([i, n]) => [i, -n]);
      expect(used).toEqual(terms[t].powers.map(([i, n]) => [i, n]));
      // Nucleons and charge are conserved.
      expect(r.nu.reduce((s, [i, n]) => s + NET_DATA.A[i] * n, 0)).toBe(0);
      expect(r.nu.reduce((s, [i, n]) => s + NET_DATA.Z[i] * n, 0)).toBe(0);
      expect(r.pair).toBe(t >> 1);
      expect(r.dir).toBe(t % 2 ? 'rev' : 'fwd');
    });
    // Each pair's backward reaction undoes the forward one.
    for (let k = 0; k < terms.length; k += 2) {
      const fwd = Object.fromEntries(NET_REACTIONS[k].nu);
      const rev = Object.fromEntries(NET_REACTIONS[k + 1].nu.map(([i, n]) => [i, -n]));
      expect(rev).toEqual(fwd);
    }
  });
});

// Independent of the table: aprox13's dY/dt for he4, c12 and o16 written out, term by term
// (train.FULL_TERMS: 0, 1 C12(a,g)O16; 2, 3 O16(a,g)Ne20; 4, 5 Ne20(a,g)Mg24; 6, 7 C+C;
// 8, 9 C+O; 10, 11 O+O; 12, 13 C+Ne; 14, 15 triple alpha; 16 on: the captures from Mg24).
function hecoRhs(r: number[]): number[] {
  const he = -r[0] + r[1] - r[2] + r[3] - r[4] + r[5] + r[6] - r[7] + r[8] - r[9] + r[10] - r[11] + r[12] - r[13] - 3 * r[14] + 3 * r[15];
  let captures = 0;
  for (let t = 16; t < 32; t += 2) captures += r[t + 1] - r[t];
  const heAll = he + captures;
  const c = -r[0] + r[1] - 2 * r[6] + 2 * r[7] - r[8] + r[9] - r[12] + r[13] + r[14] - r[15];
  const o = r[0] - r[1] - r[2] + r[3] - r[8] + r[9] - 2 * r[10] + 2 * r[11];
  return [heAll, c, o];
}

describe('reaction rates', () => {
  const states: [number, number, number[]][] = [
    [7, 9.4, [0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [6, 8.5, [0.3, 0.35, 0.35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [8, 9.7, [0.05, 0.1, 0.2, 0.1, 0.15, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002]],
  ];

  it('sum to the reference dY/dt of He, C and O', () => {
    for (const [lr, lT, X] of states) {
      const r = netReactionRates(lr, lT, X);
      const { dYdt } = reactionEdges(r);
      const want = hecoRhs(r);
      for (let i = 0; i < 3; i++) expect(dYdt[i]).toBeCloseTo(want[i], 10 - Math.max(0, Math.ceil(Math.log10(Math.abs(want[i]) + 1e-300))));
      // Mass is conserved: sum A_i dY_i/dt = 0.
      const mass = dYdt.reduce((s, y, i) => s + NET_DATA.A[i] * y, 0);
      const scale = dYdt.reduce((s, y, i) => s + NET_DATA.A[i] * Math.abs(y), 0);
      expect(Math.abs(mass)).toBeLessThan(1e-12 * scale + 1e-300);
    }
  });

  it('are the network inputs, unclamped', () => {
    const r = netReactionRates(7, 9.4, states[0][2]);
    expect(r.length).toBe(32);
    // C+C at 2.5e9 K and 1e7 g/cm^3 burns fast, and the triple alpha has no helium to burn.
    expect(r[6]).toBeGreaterThan(1e-3);
    expect(r[14]).toBeLessThan(1e-60);
  });

  it('keep the watch energy rate', () => {
    // The energy of the eight He/C/O reactions, written against the rates directly.
    const X = [0.2, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const eps = netEnergyRate(6.5, 9, X);
    expect(eps).toBeGreaterThan(0);
    const { edges } = reactionEdges(netReactionRates(6.5, 9, X), X);
    // The graph's energy from the C+C, O+O, 3a and captures agrees within the C+O branching the watch takes (half to Si28).
    const total = edges.reduce((s, e) => s + e.energy, 0);
    expect(total / eps).toBeGreaterThan(0.5);
    expect(total / eps).toBeLessThan(2);
  });
});

describe('reaction edges', () => {
  const rates = (pairs: Record<number, [number, number]>) => {
    const r = new Array(32).fill(0);
    for (const [k, [f, b]] of Object.entries(pairs)) [r[2 * +k], r[2 * +k + 1]] = [f, b];
    return r;
  };

  it('net each pair and point the way the flux goes', () => {
    const { edges } = reactionEdges(rates({ 0: [2, 0.5], 3: [1, 3], 9: [1, 0.995] }));
    expect(edges.length).toBe(16);
    expect(edges[0]).toMatchObject({ label: 'C12(a,g)O16', revLabel: 'O16(g,a)C12', signed: 1.5, net: 1.5, dir: 1, eq: false });
    expect(edges[3]).toMatchObject({ label: 'C12(C12,a)Ne20', signed: -2, net: 2, dir: -1, eq: false });
    expect(edges[9]).toMatchObject({ label: 'Si28(a,g)S32', dir: 1, eq: true });
    expect(edges[5]).toMatchObject({ net: 0, dir: 0, eq: false });
    // Captures release energy going up the chain.
    expect(edges[0].Q).toBeGreaterThan(0);
    expect(edges[3].energy).toBeLessThan(0);
  });

  it('drop the terms of a species the cell does not hold', () => {
    const X = [0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const { edges } = reactionEdges(rates({ 0: [1e-40, 1e-3], 15: [1e-40, 1e-40] }), X);
    expect(edges[0]).toMatchObject({ fwd: 0, rev: 1e-3, dir: -1 });
    expect(edges[15]).toMatchObject({ fwd: 0, rev: 0, eq: false, dir: 0 });
  });
});

describe('cell reactions', () => {
  // cgs code units: density in g/cm^3, time in seconds.
  const units = { name: 'cgs', length: 0.01, mass: 1e-3, time: 1 };
  const u = netUnits(units);
  const full = { species: NET_SPECIES, units };

  it('are none without the network', () => {
    expect(cellReactions(null, full, 1, 1, [])).toBeNull();
    expect(cellReactions({ fuel: 'a', ash: 'b', k: 1, ea: 1, q: 1 } as never, full, 1, 1, [])).toBeNull();
    expect(cellReactions({ network: 'aprox13' }, { species: ['a', 'b'], units }, 1, 1, [])).toBeNull();
  });

  it('say why a cell does not burn, and when the run only watches', () => {
    const spec = { network: 'aprox13' } as const;
    const cold = cellReactions(spec, full, 1e7 / u.rho, 5e7 / u.T, [0, 0.5, 0.5]);
    expect(cold?.burning).toBe(false);
    expect(cold?.why).toMatch(/T < /);
    const hot = cellReactions(spec, full, 1e7 / u.rho, 3e9 / u.T, [0, 0.5, 0.5]);
    expect(hot?.burning).toBe(true);
    expect(hot?.rho).toBeCloseTo(1e7, -1);
    // Carbon burns: C+C and C+O carry it away from c12.
    expect(hot!.dYdt[1]).toBeLessThan(0);
    expect(hot!.edges[3].dir).toBe(1);
    const watch = cellReactions(spec, { species: ['he4', 'c12', 'o16'], units }, 1e7 / u.rho, 3e9 / u.T, [0.1, 0.45, 0.45]);
    expect(watch?.burning).toBe(false);
    expect(watch?.why).toMatch(/watching/);
    // Only the He/C/O reactions light up.
    expect(watch!.edges.slice(8).every((e) => e.net === 0)).toBe(true);
    expect(watch?.watching).toBe(true);
    expect(watch?.X.length).toBe(13);
    expect(watch?.X.slice(3).every((x) => x === 0)).toBe(true);
  });
});

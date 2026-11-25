import { describe, expect, it } from 'vitest';
import { defaultEosParams, makeEos } from '../../src/eos/eos';
import { defaultCoolingParams } from '../../src/cooling/cooling';
import { cellEquations, gravityAt, importance, rampColor, type TermInput } from '../../src/ui/terms';
import type { CellStencil } from '../../src/sim';

const eos = makeEos('ideal', { ...defaultEosParams(), gamma: 5 / 3 });
const DX = 0.1;

/** A stencil of the cells around the origin, from the primitive fields at each cell center. */
function stencil(rho: (x: number[]) => number, v: (x: number[]) => number[], p: (x: number[]) => number, ndim: 2 | 3 = 3, missing: number[] = []): CellStencil {
  const s: CellStencil = { u1: new Array(27).fill(null), u2: new Array(27).fill(null), g: [0, 0, 0] };
  for (let k = -1; k <= 1; k++) {
    for (let j = -1; j <= 1; j++) {
      for (let i = -1; i <= 1; i++) {
        const n = i + 1 + 3 * (j + 1) + 9 * (k + 1);
        if ((ndim === 2 && k !== 0) || missing.includes(n)) continue;
        const x = [i * DX, j * DX, k * DX];
        const r = rho(x);
        const u = v(x);
        const eint = eos.eintFromP(r, p(x));
        s.u1[n] = [r, r * u[0], r * u[1], r * u[2]];
        s.u2[n] = [eint + 0.5 * r * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2), eint, 0, 0];
      }
    }
  }
  return s;
}

const input = (st: CellStencil, extra: Partial<TermInput> = {}): TermInput => ({
  ndim: 3, dx: DX, stencil: st, pos: [0, 0, 0], eos, gext: [0, 0, 0], masses: [], cooling: 'off', coolParams: defaultCoolingParams(), calm: 0, ...extra,
});

const eq = (x: TermInput, id: string) => cellEquations(x)!.find((e) => e.id === id)!;
const termOf = (x: TermInput, eqId: string, id: string) => eq(x, eqId).terms.find((t) => t.id === id);

describe('equation terms of the query readout', () => {
  it('gas at rest under uniform pressure: every term is zero', () => {
    const x = input(stencil(() => 1, () => [0, 0, 0], () => 1));
    for (const e of cellEquations(x)!) {
      expect(e.largest).toBe(0);
      expect(e.netSize).toBe(0);
    }
  });

  it('a uniform flow carries nothing in or out', () => {
    const x = input(stencil(() => 2, () => [0.3, -0.2, 0.1], () => 1));
    expect(termOf(x, 'mass', 'mass-flux')!.size).toBeLessThan(1e-12);
    expect(termOf(x, 'momentum', 'mom-flux')!.size).toBeLessThan(1e-12);
    expect(termOf(x, 'energy', 'energy-flux')!.size).toBeLessThan(1e-12);
  });

  it('a diverging flow empties the cell at rho div v', () => {
    const x = input(stencil(() => 1, (r) => [r[0], 2 * r[1], 0], () => 1));
    expect(termOf(x, 'mass', 'mass-flux')!.value[0]).toBeCloseTo(-3, 10);
  });

  it('hydrostatic balance: pressure gradient and gravity cancel, both large', () => {
    const g = 2;
    const x = input(stencil(() => 1, () => [0, 0, 0], (r) => 10 - g * r[2]), { gext: [0, 0, -g] });
    const m = eq(x, 'momentum');
    expect(termOf(x, 'momentum', 'pressure')!.value[2]).toBeCloseTo(g, 10);
    expect(termOf(x, 'momentum', 'gravity')!.value[2]).toBeCloseTo(-g, 10);
    expect(m.largest).toBeCloseTo(g, 10);
    expect(m.netSize).toBeLessThan(1e-10);
  });

  it('a missing neighbor gives a one-sided difference', () => {
    // Slot 14 is the neighbor at +x.
    const x = input(stencil(() => 1, () => [0, 0, 0], (r) => 1 + 3 * r[0], 3, [14]));
    expect(termOf(x, 'momentum', 'pressure')!.value[0]).toBeCloseTo(-3, 10);
  });

  it('2D: no z neighbors and no z component', () => {
    const x = input(stencil(() => 1, (r) => [r[0], r[1], 0], () => 1, 2), { ndim: 2 });
    expect(termOf(x, 'mass', 'mass-flux')!.value[0]).toBeCloseTo(-2, 10);
  });

  it('a point mass pulls at G M / (r^2 + soft^2)^(3/2) r', () => {
    const x = input(stencil(() => 1, () => [0, 0, 0], () => 1), { pos: [3, 0, 4], masses: [{ gm: 50, pos: [0, 0, 0], soft: 0 }] });
    const g = gravityAt(x);
    expect(Math.hypot(...g)).toBeCloseTo(2, 10);
    expect(g[0] / g[2]).toBeCloseTo(0.75, 10);
    expect(g[2]).toBeLessThan(0);
  });

  it('terms that are off are left out; damping and cooling appear when on', () => {
    const plain = input(stencil(() => 1, () => [1, 0, 0], () => 1));
    expect(eq(plain, 'momentum').terms.map((t) => t.id)).toEqual(['mom-flux', 'pressure']);
    const on = input(stencil(() => 1, () => [1, 0, 0], () => 1), { calm: 0.5, cooling: 'power', coolParams: { ...defaultCoolingParams(), rate: 1, alpha: 0.5 } });
    expect(termOf(on, 'momentum', 'calm')!.value[0]).toBeCloseTo(-0.5, 12);
    expect(termOf(on, 'energy', 'calm-e')!.value[0]).toBeCloseTo(-0.5, 12);
    expect(termOf(on, 'energy', 'cooling')!.value[0]).toBeLessThan(0);
  });

  it('a barotropic gas has no energy equation', () => {
    const baro = makeEos('barotropic', defaultEosParams());
    const x = input(stencil(() => 1, () => [0, 0, 0], () => 1), { eos: baro });
    expect(cellEquations(x)!.map((e) => e.id)).toEqual(['mass', 'momentum']);
  });

  it('MHD: a uniform field exerts no force; a field-line bend gives tension', () => {
    const uniform = stencil(() => 1, () => [0, 0, 0], () => 1);
    uniform.b = uniform.u1.map((u) => (u ? [1, 0.5, 0] : null));
    const x = input(uniform);
    expect(termOf(x, 'momentum', 'lorentz')!.size).toBeLessThan(1e-12);
    // B = (1, b0 x, 0): curl B x B = (B . grad) B - grad(B^2 / 2) = (-b0^2 x, b0, 0); at x = 0, (0, b0, 0).
    const b0 = 0.3;
    const bent = stencil(() => 1, () => [0, 0, 0], () => 1);
    bent.b = bent.u1.map((u, n) => (u ? [1, b0 * ((n % 3) - 1) * DX, 0] : null));
    const f = termOf(input(bent), 'momentum', 'lorentz')!.value;
    expect(f[0]).toBeCloseTo(0, 12);
    expect(f[1]).toBeCloseTo(b0, 12);
  });

  it('MHD: induction of a field compressed by a converging flow', () => {
    // v = (-x, 0, 0), B = (0, 1, 0): dB_y/dt = -d(v_x B_y)/dx = 1.
    const st = stencil(() => 1, (r) => [-r[0], 0, 0], () => 1);
    st.b = st.u1.map((u) => (u ? [0, 1, 0] : null));
    const x = input(st, { resistivity: 0.1 });
    const ind = eq(x, 'induction');
    expect(ind.terms.find((t) => t.id === 'induction')!.value[1]).toBeCloseTo(1, 10);
    // A uniform field does not diffuse.
    expect(ind.terms.find((t) => t.id === 'resistive')!.size).toBe(0);
  });

  it('MHD: energy has the magnetic tension work; no induction equation without a field', () => {
    const st = stencil(() => 1, () => [0, 0, 0], () => 1);
    st.b = st.u1.map((u) => (u ? [1, 0, 0] : null));
    expect(eq(input(st), 'energy').terms.map((t) => t.id)).toContain('mag-work');
    expect(cellEquations(input(stencil(() => 1, () => [0, 0, 0], () => 1)))!.map((e) => e.id)).not.toContain('induction');
  });

  it('burning heats the gas at the rate it read', () => {
    const st = stencil(() => 1, () => [0, 0, 0], () => 1);
    st.heat = 4.5;
    expect(termOf(input(st), 'energy', 'burn')!.value[0]).toBe(4.5);
  });

  it('radiation: force kappa_R rho F / c, heating c kappa_P rho (E - a T^4)', () => {
    const st = stencil(() => 2, () => [0.5, 0, 0], () => 1);
    st.rad = [3, 0.6, 0, 0];
    const radiation = { c: 10, a: 0.01, opacity: () => ({ planck: 0.2, rosseland: 0.5 }) };
    const x = input(st, { radiation });
    const T = eos.temperature(2, st.u2[13]![1]);
    expect(termOf(x, 'momentum', 'rad-force')!.value[0]).toBeCloseTo((0.5 * 2 * 0.6) / 10, 12);
    expect(termOf(x, 'energy', 'rad-heat')!.value[0]).toBeCloseTo(10 * 0.2 * 2 * (3 - 0.01 * T ** 4), 10);
    expect(termOf(x, 'energy', 'rad-work')!.value[0]).toBeCloseTo(0.5 * 0.06, 12);
  });

  it('float32 round-off counts as zero: radiation in equilibrium, a rounded pressure', () => {
    const st = stencil(() => 2, () => [0, 0, 0], () => 1);
    const T = eos.temperature(2, st.u2[13]![1]);
    st.rad = [0.01 * T ** 4 * (1 + 3e-7), 0, 0, 0];
    const radiation = { c: 1e4, a: 0.01, opacity: () => ({ planck: 1e3, rosseland: 1e3 }) };
    expect(termOf(input(st, { radiation }), 'energy', 'rad-heat')!.value[0]).toBe(0);
    const rounded = stencil(() => 1, () => [0, 0, 0], (r) => Math.fround(1 + 1e-8 * r[0]));
    expect(termOf(input(rounded), 'momentum', 'pressure')!.size).toBe(0);
  });

  it('importance runs over six decades below the largest term', () => {
    expect(importance(1, 1)).toBe(1);
    expect(importance(1e-3, 1)).toBeCloseTo(0.5, 12);
    expect(importance(1e-9, 1)).toBe(0);
    expect(importance(0, 1)).toBe(0);
    expect(importance(1, 0)).toBe(0);
    expect(rampColor(0)).toBe('rgb(74, 81, 98)');
    expect(rampColor(1)).toBe('rgb(255, 224, 102)');
  });
});

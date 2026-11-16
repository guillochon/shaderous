import { describe, expect, it } from 'vitest';
import { laneEmden, polytropeK, polytropeMass, polytropeScales } from '../../src/problems/lane_emden';
import { placedSettling, Settler, type SettleCells } from '../../src/problems/settle';
import { packBrush, STAR_TABLE, starBrush } from '../../src/sim';
import { POLY, POLY_K, polyStar, polytrope, polytropeUnits } from '../../src/problems/polytrope';
import { defaultEosParams, makeEos, physicalEosParams } from '../../src/eos/eos';
import { loadHelmholtz } from '../../src/eos/helmholtz';
import { COMPOSITIONS, meanMolecularWeight } from '../../src/units/composition';
import { ASTRO } from '../../src/units/constants';
import { codeG, fromCode, kelvinPerCodeT } from '../../src/units/units';

describe('star brush', () => {
  it('normalizes as the polytrope preset does', () => {
    // n = 1 has K = 2 G R^2 / pi whatever the central density.
    expect(polytropeK(laneEmden(1), POLY.rhoc, POLY.R, POLY.G)).toBeCloseTo(POLY_K, 4);
    expect(polytropeK(laneEmden(1), 7, POLY.R, POLY.G)).toBeCloseTo(POLY_K, 4);
  });

  it('inverts polytropeScales', () => {
    for (const n of [1, 1.5, 3]) {
      const poly = laneEmden(n);
      const { rhoc, K } = polytropeScales(poly, 2.5, 0.7, 1.3);
      expect(polytropeMass(poly, rhoc, 0.7)).toBeCloseTo(2.5, 6);
      expect(polytropeK(poly, rhoc, 0.7, 1.3)).toBeCloseTo(K, 6);
    }
  });

  it('packs the profile into the brush block', () => {
    const op = starBrush(1.5, 2, 0.1, 4, 1, [0.5, 0.25, 0.75], [1, 0, 0]);
    const t = op.star!.table;
    expect(t.length).toBe(STAR_TABLE);
    expect(t[0]).toBeCloseTo(1, 5);
    expect(t[STAR_TABLE - 1]).toBe(0);
    for (let i = 1; i < STAR_TABLE; i++) expect(t[i]).toBeLessThanOrEqual(t[i - 1]);
    const f = new Float32Array(packBrush(op));
    expect(new Uint32Array(f.buffer)[8]).toBe(6);
    expect(f[9]).toBe(2);
    expect(f[28]).toBeCloseTo(op.star!.K, 5);
    expect(f[29]).toBeCloseTo(1 + 1 / 1.5, 6);
    expect(f[32]).toBeCloseTo(1, 5);
  });
});

describe('polytrope preset', () => {
  it('is the n = 1 star it was by default', () => {
    const st = polyStar();
    expect(st.K).toBe(POLY_K);
    expect(st.rhoCut).toBe(POLY.rhoCut);
    expect(st.mass).toBeCloseTo(polytropeMass(laneEmden(1), POLY.rhoc, POLY.R), 6);
    const w = polytrope.init(64, makeEos('ideal', defaultEosParams()), 3, { m: 1, r: 1, n: 1 })(0.1, 0, 0);
    const xi = (Math.PI * 0.1) / POLY.R;
    expect(w.rho).toBeCloseTo(Math.sin(xi) / xi, 12);
    expect(w.p).toBeCloseTo(POLY_K * w.rho ** 2, 12);
  });

  it('cuts every index off where the ambient gas is as hot as the center', () => {
    for (const n of [1, 1.5, 3]) {
      const st = polyStar(n);
      expect(st.pressure(st.rhoCut) / POLY.rhoAmb).toBeCloseTo(st.pressure(POLY.rhoc) / POLY.rhoc, 10);
      expect(st.density(0)).toBeCloseTo(POLY.rhoc, 6);
      expect(st.density(POLY.R)).toBe(0);
    }
  });

  it('takes its physical scale from the mass and radius', () => {
    for (const n of [1, 1.5, 3]) {
      const u = polytropeUnits({ m: 2, r: 0.5, n });
      expect(codeG(u)).toBeCloseTo(1, 12);
      expect(fromCode(u, 'length', POLY.R) / ASTRO.Rsun).toBeCloseTo(0.5, 12);
      expect(fromCode(u, 'mass', polyStar(n).mass) / ASTRO.Msun).toBeCloseTo(2, 12);
    }
    // Central density of an n = 3/2 polytrope: 5.99 times the mean, 1410 kg/m^3 for the Sun.
    const sun = polytropeUnits({ m: 1, r: 1, n: 1.5 });
    expect(fromCode(sun, 'density', POLY.rhoc) / (5.99 * 1410)).toBeCloseTo(1, 2);
  });

  it('gives the tabulated equation of state a star it can hold up', async () => {
    await loadHelmholtz();
    // A Sun-like n = 3/2 star: its gas is close to ideal, so the center is near
    // T = mu m_u p / (k rho), about 7.5e6 K, and the pressure comes back from the energy.
    const u = polytropeUnits({ m: 1, r: 1, n: 1.5 });
    const eos = makeEos('helmholtz', { ...defaultEosParams(), ...physicalEosParams(u, COMPOSITIONS.solar) });
    const st = polyStar(1.5);
    const kelvin = kelvinPerCodeT(u, meanMolecularWeight(COMPOSITIONS.solar));
    for (const rho of [POLY.rhoc, 0.1, st.rhoCut]) {
      const p = st.pressure(rho);
      const eint = eos.eintFromP(rho, p);
      expect(eos.pressure(rho, eint) / p).toBeCloseTo(1, 4);
      const T = eos.temperature(rho, eint) * kelvin;
      expect(T).toBeGreaterThan(1e4);
      if (rho === POLY.rhoc) {
        expect(T).toBeGreaterThan(6e6);
        expect(T).toBeLessThan(8e6);
      }
    }
  });
});

describe('settling a star the tool placed', () => {
  // An n = 1 star of radius 0.2 off the center of a box of side 1, in gas of density 1e-3 with
  // a denser lump far from it, on 32^3 cells.
  const n = 32;
  const dx = 1 / n;
  const R = 0.2;
  const center = [0.1, -0.05, 0];
  const K = (2 * R * R) / Math.PI;
  const star = { n: 1, K, rhoc: 1, center, vel: [0.3, 0, 0], R };
  const at = (i: number) => (i + 0.5) * dx - 0.5;
  const pos = (q: number): [number, number, number] => [at(q % n), at(Math.floor(q / n) % n), at(Math.floor(q / (n * n)))];
  const profile = (r: number) => (r >= R ? 0 : r < 1e-9 ? 1 : Math.sin((Math.PI * r) / R) / ((Math.PI * r) / R));
  const build = () => {
    const u1 = new Float32Array(n ** 3 * 4);
    const u2 = new Float32Array(n ** 3 * 4);
    const phi = new Float32Array(n ** 3);
    for (let q = 0; q < n ** 3; q++) {
      const [x, y, z] = pos(q);
      const r = Math.hypot(x - center[0], y - center[1], z - center[2]);
      const lump = Math.hypot(x + 0.35, y - 0.35, z) < 0.08;
      const bg = lump ? 0.5 : 1e-3;
      const rho = Math.max(profile(r), bg);
      u1[q * 4] = rho;
      u1[q * 4 + 1] = lump ? 0.1 : 0;
      u2[q * 4 + 1] = lump ? 2 : 1e-3; // e_int: pressure with gamma = 2
      u2[q * 4] = u2[q * 4 + 1] + (0.5 * u1[q * 4 + 1] ** 2) / rho;
      // The potential the analytic star balances in: h = 2 K rho inside, rising outside.
      phi[q] = r < R ? -2 * K * profile(r) : (r - R) / R;
    }
    const cells: SettleCells = { count: n ** 3, pos, vol: () => dx ** 3, u1, u2 };
    return { cells, phi };
  };

  it('takes its surroundings from the gas just outside the star', () => {
    const { cells } = build();
    const set = placedSettling(star, 1, cells, (_rho, eint) => eint);
    expect(typeof set).toBe('object');
    if (typeof set === 'string') return;
    expect(set.keep).toBe(true);
    expect(set.rhoCut).toBeCloseTo(1e-3, 9);
    expect(set.rhoAmb).toBeCloseTo(1e-3, 9);
    expect(set.pAmb).toBeCloseTo(1e-3, 9);
    // Among gas as dense as itself there is nothing to settle against.
    expect(typeof placedSettling({ ...star, rhoc: 5e-3 }, 1, cells, (_rho, eint) => eint)).toBe('string');
  });

  it('reshapes the star and leaves the gas around it alone', () => {
    const { cells, phi } = build();
    const before = { u1: cells.u1.slice(), u2: cells.u2.slice() };
    const set = placedSettling(star, 1, cells, (_rho, eint) => eint);
    if (typeof set === 'string') throw new Error(set);
    // The gas beyond the star is what it falls toward: all of it is among the lumps.
    let outside = 0;
    for (let q = 0; q < cells.count; q++) {
      const [x, y, z] = pos(q);
      if (Math.hypot(x - center[0], y - center[1], z - center[2]) > 1.3 * R) outside += cells.u1[q * 4] * dx ** 3;
    }
    expect(set.others!.reduce((m, o) => m + o.gm, 0)).toBeCloseTo(outside, 8);
    // This potential is the star's alone: nothing else's pull is in it to leave out.
    set.others = [];
    const settler = new Settler(set, cells);
    for (let i = 0; i < 3; i++) settler.round(phi);
    settler.finish((_rho, p) => p);
    let starCells = 0;
    for (let q = 0; q < cells.count; q++) {
      const o = q * 4;
      if (before.u1[o] > set.rhoCut && settler.owner[q] === 0) {
        // The star: in balance with this potential it is the analytic one, moving as placed.
        starCells++;
        expect(settler.rewrote[q]).toBe(1);
        if (cells.u1[o] > set.rhoCut) {
          expect(cells.u1[o] / before.u1[o]).toBeCloseTo(1, 2);
          expect(cells.u1[o + 1] / cells.u1[o]).toBeCloseTo(0.3, 5);
          expect(cells.u2[o + 1]).toBeCloseTo(K * cells.u1[o] ** 2, 6);
        }
      } else {
        // Everything else, the lump included, is as it was.
        for (let c = 0; c < 4; c++) {
          expect(cells.u1[o + c]).toBe(before.u1[o + c]);
          expect(cells.u2[o + c]).toBe(before.u2[o + c]);
        }
      }
    }
    expect(starCells).toBeGreaterThan(500);
  });
});

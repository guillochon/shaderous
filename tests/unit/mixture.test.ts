import { describe, expect, it } from 'vitest';
import { defaultEosParams, EOS_MODULES, makeEos, physicalEosParams, type EosKind, type EosParams } from '../../src/eos/eos';
import { assemble } from '../../src/gpu/shaders';
import { compSlot, compSrcSlot, hydrogenParticles, mixMoments, momentCoefficients, problemCoefficients, speciesNuclides } from '../../src/species/mixture';
import { bubbleDensity } from '../../src/problems/shock_bubble';
import { COMPOSITIONS, composition, meanMolecularWeight, nuclearAverages } from '../../src/units/composition';
import { flowUnits } from '../../src/units/units';

const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

describe('species that set the equation of state', () => {
  it('only nuclides (and air) do', () => {
    expect(speciesNuclides(['h1', 'he4'])?.map((s) => s.A)).toEqual([1, 4]);
    expect(speciesNuclides(['air', 'he4'])?.map((s) => s.A)).toEqual([28.96, 4]);
    expect(speciesNuclides(['a', 'b'])).toBeNull();
    expect(speciesNuclides(['h1', 'tracer'])).toBeNull();
    expect(speciesNuclides([])).toBeNull();
    expect(problemCoefficients({ species: ['a'] }, 1.4)).toBeNull();
    expect(problemCoefficients({}, 1.4)).toBeNull();
  });

  it('give the reference composition the uniform constants', () => {
    for (const ref of [COMPOSITIONS.solar, COMPOSITIONS.solarNeutral, COMPOSITIONS.carbonOxygen, COMPOSITIONS.hydrogen]) {
      const k = mixMoments(momentCoefficients(ref.species, ref, 1.4), ref.X);
      const ion = physicalEosParams(undefined, ref);
      expect(k[0]).toBeCloseTo(1, 12);
      expect(k[1]).toBeCloseTo(1, 12);
      expect(k[2]).toBeCloseTo(ion.ionH!, 12);
      expect(k[0] - hydrogenParticles(ref) * k[2]).toBeCloseTo(ion.ionOther!, 12);
      // The heat capacity: gamma = 1 + x / w is the run's.
      expect(1 + k[0] / k[3]).toBeCloseTo(1.4, 12);
    }
  });

  it('mix as mass-weighted sums, giving mu and Ye of any mixture', () => {
    const ref = COMPOSITIONS.solar;
    const mu = meanMolecularWeight(ref);
    const ye = nuclearAverages(ref).ye;
    const c = momentCoefficients(speciesNuclides(['h1', 'he4', 'c12'])!, ref);
    const X = [0.2, 0.3, 0.5];
    const k = mixMoments(c, X);
    const mix = composition('mix', { h1: 0.2, he4: 0.3, c12: 0.5 });
    expect(rel(k[0], mu / meanMolecularWeight(mix))).toBeLessThan(1e-12);
    expect(rel(k[1], nuclearAverages(mix).ye / ye)).toBeLessThan(1e-12);
    // Unnormalized and missing fractions are treated as packSpecies stores them.
    expect(mixMoments(c, [0.4, 0.6, 1])).toEqual(k);
    expect(mixMoments(c, undefined)).toEqual(c[0]);
  });

  it('make the reference composition the uniform equation of state, and others what they are', () => {
    const units = flowUnits('cgs', 1e-2, 1e3, 1e-2);
    const ref = COMPOSITIONS.carbonOxygen;
    const cases: [EosKind, Partial<EosParams>][] = [
      ['ideal', { gamma: 5 / 3 }],
      ['gas_radiation', { gamma: 5 / 3, R: 1, arad: 1 }],
      ['degenerate', { gamma: 5 / 3, R: 1, arad: 1e-2, degRho: 1, degP: 0.2 }],
      ['ionizing', { gamma: 5 / 3 }],
    ];
    for (const [kind, over] of cases) {
      const params = { ...defaultEosParams(), ...physicalEosParams(undefined, ref), ...over } as EosParams;
      const k = mixMoments(momentCoefficients(ref.species, ref, params.gamma), ref.X);
      const a = makeEos(kind, params);
      const b = makeEos(kind, params, k, hydrogenParticles(ref));
      for (const [rho, eint] of [[1, 2], [0.3, 0.7], [3, 10]]) {
        expect(rel(b.pressure(rho, eint), a.pressure(rho, eint))).toBeLessThan(1e-9);
        expect(rel(b.temperature(rho, eint), a.temperature(rho, eint))).toBeLessThan(1e-9);
      }
    }
    // Helium against hydrogen, same density and pressure: an ideal gas's temperature goes as mu.
    const c = momentCoefficients(speciesNuclides(['h1', 'he4'])!, COMPOSITIONS.solar);
    const p = { ...defaultEosParams(), gamma: 5 / 3 };
    const h = makeEos('ideal', p, mixMoments(c, [1, 0]));
    const he = makeEos('ideal', p, mixMoments(c, [0, 1]));
    expect(rel(he.temperature(1, 1.5) / h.temperature(1, 1.5), (4 / 3) / 0.5)).toBeLessThan(1e-12);
    // Degenerate electrons: pure helium (Ye = 1/2) with a C/O reference has the reference's
    // degenerate pressure, and hydrogen (Ye = 1) that of twice the density.
    const cgs = { ...defaultEosParams(), ...physicalEosParams(units, ref), arad: 0 } as EosParams;
    const cd = momentCoefficients(speciesNuclides(['h1', 'he4'])!, ref);
    const cold = makeEos('degenerate', cgs);
    expect(rel(makeEos('degenerate', cgs, mixMoments(cd, [0, 1])).pressure(1e6, 0), cold.pressure(1e6, 0))).toBeLessThan(1e-12);
    expect(rel(makeEos('degenerate', cgs, mixMoments(cd, [1, 0])).pressure(1e6, 0), cold.pressure(2e6, 0))).toBeLessThan(1e-12);
  });

  it('give each species its own gamma, mixed as gases at one temperature', () => {
    const air = composition('air', { air: 1 }, 'neutral');
    const c = momentCoefficients(speciesNuclides(['he4', 'air'])!, air, 1.4, { he4: 5 / 3 });
    // Quirk & Karni's bubble: helium with 28% air by mass has density 0.182 and gamma 1.648
    // beside air at its pressure and temperature.
    const k = mixMoments(c, [0.72, 0.28]);
    expect(bubbleDensity(0.72)).toBeCloseTo(0.182, 3);
    expect(1 + k[0] / k[3]).toBeCloseTo(1.648, 2);
    expect(1 + mixMoments(c, [0, 1])[0] / mixMoments(c, [0, 1])[3]).toBeCloseTo(1.4, 12);
    expect(1 + mixMoments(c, [1, 0])[0] / mixMoments(c, [1, 0])[3]).toBeCloseTo(5 / 3, 12);
    // A cell holding the two by volume, at one pressure and temperature, has their pressure.
    const params = { ...defaultEosParams(), gamma: 1.4 };
    for (const f of [0.1, 0.5, 0.9]) {
      const rhoHe = bubbleDensity(1);
      const rho = f * rhoHe + (1 - f);
      const eint = f * (1 / (5 / 3 - 1)) + (1 - f) * (1 / 0.4); // p = 1 in both
      const eos = makeEos('ideal', params, mixMoments(c, [(f * rhoHe) / rho, (1 - f) / rho]));
      expect(eos.pressure(rho, eint)).toBeCloseTo(1, 12);
    }
  });

  it('assemble into every pass that reads or writes them', () => {
    const c = momentCoefficients(speciesNuclides(['h1', 'he4', 'c12', 'o16', 'ne20'])!);
    const comp = compSlot(c, 2);
    expect(comp).toContain('const COMP = true;');
    expect(comp).toContain('const COMP_NG: u32 = 2u;');
    expect(compSlot(null, 0)).toBe('eos/comp_off.wgsl');
    expect(compSlot(null, 3)).toContain('const COMP_NG: u32 = 3u;');
    const slots = {
      eos: EOS_MODULES.degenerate, riemann: 'hydro/riemann/hllc.wgsl', limiter: 'hydro/limiters/koren.wgsl', cooling: 'cooling/off.wgsl',
      bins: 'amr/bins_off.wgsl', mid: 'amr/mid_off.wgsl', diffusion: 'const VISCOSITY = true;\nconst CONDUCTION = true;\n',
      damping: 'const DAMPING = false;\nfn damping_target(x: vec3<f32>) -> DampTarget { return DampTarget(0.0, 1.0, vec3<f32>(0.0), 1.0); }\n',
      species: 'hydro/species_on.wgsl', comp, comp_src: compSrcSlot(c, 40),
      species_n: 'const NG: u32 = 2u;\nconst SP_DEPTH: i32 = 40;\n',
    };
    const entries = [
      'hydro/rates.wgsl', 'hydro/predict.wgsl', 'hydro/update.wgsl', 'gravity/correct.wgsl', 'edit/brush.wgsl', 'hydro/diagnostics.wgsl', 'render/field.wgsl',
      'amr/assemble.wgsl', 'amr/update.wgsl', 'amr/correct.wgsl', 'amr/brush.wgsl', 'amr/diagnostics.wgsl', 'amr/field.wgsl',
      'hydro/species.wgsl', 'amr/species.wgsl', 'amr/species_regrid.wgsl',
    ];
    for (const entry of entries) {
      const src = assemble(entry, slots);
      expect(src).not.toMatch(/^\s*#include/m);
      expect(src.match(/const COMP = true;/g)?.length).toBe(1);
    }
    expect(assemble('amr/assemble.wgsl', slots)).toContain('fn comp_assembled');
  });
});

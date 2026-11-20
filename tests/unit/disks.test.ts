// Formulas in a setup, disks with profiles of one's own, disks about stars, and point masses
// held on their orbits (src/setup/expr.ts, src/setup/parts/disk.ts).
import { describe, expect, it } from 'vitest';
import { defaultEosParams, makeEos } from '../../src/eos/eos';
import { preset } from '../../src/presets/index';
import { readSetup } from '../../src/setup/codec';
import { compileSetup } from '../../src/setup/compile';
import { parseFormula } from '../../src/setup/expr';
import { emptySetup } from '../../src/setup/spec';
import { completeMass, configFor, MASS_HEADER, MASS_STRIDE, packMasses, unpackMasses } from '../../src/sim';

describe('formulas in a setup', () => {
  const f = (text: string, vars: Record<string, number> = {}) => parseFormula(text, Object.keys(vars)).at(vars);

  it('does arithmetic with the usual precedence, powers and functions', () => {
    expect(f('1 + 2 * 3')).toBe(7);
    expect(f('(1 + 2) * 3')).toBe(9);
    expect(f('-2^2')).toBe(-4);
    expect(f('2^-1')).toBe(0.5);
    expect(f('2 ** 3 ** 2')).toBe(512);
    expect(f('8 / 4 / 2')).toBe(1);
    expect(f('1e-3 * 2.5E2')).toBeCloseTo(0.25, 15);
    expect(f('sqrt(16) + abs(-1) + max(2, 3) + min(2, 3) + pow(2, 3) + clamp(5, 0, 1)')).toBe(19);
    expect(f('exp(log(3)) + log10(100) + cos(pi) + step(-1) + step(2)')).toBeCloseTo(5, 12);
    expect(f('sigma0 * (r / rIn)^-1.5 * exp(-(r / rOut)^2)', { sigma0: 2, r: 4, rIn: 1, rOut: 4 })).toBeCloseTo(2 * 4 ** -1.5 * Math.exp(-1), 14);
  });

  it('refuses anything that is not a formula, and says where', () => {
    const bad: [string, RegExp][] = [
      ['', /ends too soon/], ['1 +', /ends too soon/], ['(1 + 2', /"\)" is missing/], ['1 2', /not expected/], ['x + 1', /"x" is not a name/],
      ['alert(1)', /"alert" is not a function/], ['constructor(1)', /not a function/], ['__proto__', /not a name/], ['r.constructor', /not expected/],
      ['pow(1)', /takes 2 values/], ['1; 2', /not expected/], ['"a"', /not expected/], ['r => r', /not expected/], ['x'.repeat(600), /longer than/],
      [`${'('.repeat(80)}1${')'.repeat(80)}`, /nested too deep/],
    ];
    for (const [text, why] of bad) expect(() => parseFormula(text, ['r']), text.slice(0, 20)).toThrow(why);
  });

  it('writes WGSL made only of numbers, operators, its own functions and the variables it was given', () => {
    const w = parseFormula('sigma0 * (r / rIn)^-1.5 * exp(-(r / 3)^2) + step(r - 1)', ['r', 'rIn', 'sigma0']).wgsl({ r: 'r', rIn: '0.5', sigma0: '0.001' });
    expect(w).toBe('((((0.001) * pow(((r) / (0.5)), (-1.5))) * exp((-pow(((r) / 3.0), 2.0)))) + select(0.0, 1.0, ((r) - 1.0) > 0.0))');
    expect(w).toMatch(/^[\d\s.()+\-*/,<>a-z]+$/);
  });
});

describe('disks of one\'s own', () => {
  const eos = makeEos('barotropic', { ...defaultEosParams(), gamma: 1, K: 0.01 });
  const base = { domain: { box: 8 }, physics: { eos: 'barotropic', eosParams: { gamma: 1, K: 0.01 }, G: 1 } };
  const compile = (parts: object[], more: object = {}) => {
    const { spec, report } = readSetup({ ...emptySetup(), ...base, ...more, parts });
    expect(report.error).toBeUndefined();
    return compileSetup(spec!).problem;
  };
  const mass = { kind: 'mass', id: 'm', gm: 1, soft: 0.05 };
  const disk = (more: object) => ({ kind: 'disk', id: 'd', about: ['m'], profile: 'balanced', rIn: 0.5, rOut: 3, sigma0: 1e-3, cs: 0.1, floor: 0, ...more });

  it('takes its midplane density from a formula of the radius, and its rotation from that density', () => {
    const at = compile([mass, disk({ radialFn: 'sigma0 * (r / rIn)^-2 * step(r - rIn) * step(rOut - r)' })]).init(64, eos, 3, {});
    expect(at(1, 0, 0).rho).toBeCloseTo(1e-3 * 2 ** -2, 12);
    expect(at(0, 2, 0).rho).toBeCloseTo(1e-3 * 4 ** -2, 12);
    expect(at(0.3, 0, 0).rho).toBe(0);
    expect(at(3.5, 0, 0).rho).toBe(0);
    // v^2 = G M R^2 / (R^2 + eps^2)^(3/2) + cs^2 dln(rho)/dln(R), the slope here being -2.
    const w = at(2, 0, 0);
    const eps = Math.max(0.05, (2 * 8) / 64);
    expect(w.vy ** 2).toBeCloseTo(4 / (4 + eps * eps) ** 1.5 - 2 * 0.01, 6);
  });

  it('takes its vertical profile from a formula of the height, in scale heights or in lengths', () => {
    const inH = compile([mass, disk({ verticalFn: '1 / cosh(z)^2' })]).init(64, eos, 3, {});
    const H = 0.1 * 2 ** 1.5;
    expect(inH(2, 0, H).rho / inH(2, 0, 0).rho).toBeCloseTo(1 / Math.cosh(1) ** 2, 10);
    expect(inH(2, 0, -H).rho).toBe(inH(2, 0, H).rho);
    const inL = compile([mass, disk({ verticalFn: 'max(1 - z / 0.5, 0)', verticalIn: 'length' })]).init(64, eos, 3, {});
    expect(inL(2, 0, 0.25).rho / inL(2, 0, 0).rho).toBeCloseTo(0.5, 12);
    expect(inL(1, 0, 0.6).rho).toBe(0);
  });

  it('keeps a thin disk\'s surface density whatever the vertical formula', () => {
    const at = compile([mass, disk({ profile: 'thin', floor: 0, radialFn: '2e-3 * exp(-(r - 2)^2)', verticalFn: 'exp(-z^2 / 2) * (1 + z^2)' })]).init(64, eos, 3, {});
    let sigma = 0;
    const dz = 1e-3;
    for (let z = -3; z < 3; z += dz) sigma += at(2, 0, z + dz / 2).rho * dz;
    expect(sigma).toBeCloseTo(2e-3, 6);
  });

  it('reads the same profiles from tables of points', () => {
    const at = compile([mass, disk({ radial: [[1, 1e-3], [2, 2.5e-4], [3, 0]], vertical: [[0, 1], [2, 0]] })]).init(64, eos, 3, {});
    expect(at(1.5, 0, 0).rho).toBeCloseTo(1e-3 * 1.5 ** -2, 12);
    expect(at(2.5, 0, 0).rho).toBeCloseTo(1.25e-4, 12);
    expect(at(0.9, 0, 0).rho).toBe(0);
    const H = 0.1 * 1.5 ** 1.5;
    expect(at(1.5, 0, H).rho / at(1.5, 0, 0).rho).toBeCloseTo(0.5, 10);
  });

  it('gives the damping zone the same formula, as generated WGSL', () => {
    const p = compile([mass, disk({ radialFn: 'sigma0 * (r / rIn)^-1', verticalFn: 'exp(-z^2 / 2)', damp: { from: 3, rate: 2 } })]);
    const w = p.damping!.wgsl;
    expect(w).toContain('fn disk_radial_value(r: f32) -> f32');
    expect(w).toContain('((0.001) * pow(((r) / (0.5)), (-1.0)))');
    expect(w).toContain('fn disk_vertical(h: f32) -> f32');
    expect(w).toContain('let disk = mid * disk_vertical(x.z / (0.1 * pow(R, 1.5)));');
    // Text of a formula never reaches the shader: only what the parser made of it.
    expect(() => compile([mass, disk({ radialFn: '1; } fn evil() {' })])).toThrow(/d: .*not expected/);
  });

  it('says what is wrong with a formula before anything runs', () => {
    const { spec } = readSetup({ ...emptySetup(), ...base, parts: [mass, disk({ radialFn: 'sigma0 * q' })] });
    expect(() => compileSetup(spec!)).toThrow(/d: "q" is not a name this formula knows \(r, rIn, rOut, sigma0, cs, H, pi, e\)/);
  });

  it('orbits a star: centered on it, moving with it, and left alone by its settling', () => {
    const star = { kind: 'star', id: 's', type: 'polytrope', n: 1.5, mass: 1, radius: 0.4, pos: [1, 0.5, 0], vel: [0, 0.2, 0] };
    const ideal = makeEos('ideal', { ...defaultEosParams(), gamma: 5 / 3 });
    const p = compile([star, { ...disk({ rIn: 0.8, rOut: 2 }), about: ['s'] }], { physics: { gravity: 'isolated', G: 1 }, background: { rho: 1e-8, cut: 1e-4, relativeTo: 's' } });
    const at = p.init(64, ideal, 3, {});
    // The star at its place, the disk around it, turning about the star and drifting with it.
    expect(at(1, 0.5, 0).rho).toBeGreaterThan(1);
    const [right, left] = [at(1 + 1.2, 0.5, 0), at(1 - 1.2, 0.5, 0)];
    expect(right.rho).toBeCloseTo(left.rho, 12);
    expect(right.rho).toBeGreaterThan(1e-4);
    expect(right.vy - 0.2).toBeCloseTo(-(left.vy - 0.2), 12);
    expect(right.vy - 0.2).toBeGreaterThan(0.5);
    const set = p.settle!({}, ideal);
    expect(set.keep).toBe(true);
    expect(set.stars[0].reach).toBeCloseTo(1.3 * 0.4, 12);
    // A star alone is settled as before: everything outside it is ambient gas.
    expect(preset('tidal-disruption').problem.settle!({}, ideal).keep).toBeUndefined();
  });
});

describe('a point mass held on its orbit', () => {
  it('is a flag of the part, of the solver\'s mass, and of the binary preset', () => {
    const { spec } = readSetup({ ...emptySetup(), parts: [{ kind: 'mass', id: 'm', gm: 1, onOrbit: true }, { kind: 'mass', id: 'n', gm: 1, pos: [1, 0, 0] }] });
    const p = compileSetup(spec!).problem;
    expect(p.masses!({}, 1, 0.01, 3).map((m) => !!m.onOrbit)).toEqual([true, false]);
    const packed = packMasses(p.masses!({}, 1, 0.01, 3).map((m) => completeMass(m, 0.01)));
    expect([packed[MASS_HEADER + 15], packed[MASS_HEADER + MASS_STRIDE + 15]]).toEqual([4, 0]);
    expect(unpackMasses(packed, 2).map((m) => m.onOrbit)).toEqual([true, false]);
    const binary = preset('binary').problem;
    expect(binary.masses!({ ...configFor(binary, 64).params, hold: 1 }, 1, 0.01, 2).every((m) => m.onOrbit)).toBe(true);
    expect(binary.params!.find((q) => q.id === 'hold')).toMatchObject({ kind: 'checkbox', live: true, value: 0 });
  });
});

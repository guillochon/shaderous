// Binaries with stars in them: the Roche geometry (src/problems/roche.ts), how a binary orbit
// places a star (setup/parts/bodies.ts) and what the compiled setup hands the run.
import { describe, expect, it } from 'vitest';
import { defaultEosParams, makeEos } from '../../src/eos/eos';
import { binaryOmega, eggleton, fillSeparation, l1Distance, rochePotential } from '../../src/problems/roche';
import { binaryPhi, holdPull } from '../../src/problems/settle';
import { compileSetup } from '../../src/setup/compile';
import { buildBodies } from '../../src/setup/parts/bodies';
import { checkSetup } from '../../src/setup/check';
import { completeMass, configFor, followsBinary, frameTrack, MASS_HEADER, packMasses, unpackMasses } from '../../src/sim';
import { emptySetup, type OrbitPart, type Part, type SetupSpec } from '../../src/setup/spec';

const eos = makeEos('ideal', defaultEosParams());

describe('the Roche geometry', () => {
  it('has Eggleton’s lobe radii', () => {
    // q = 1: 0.49 / (0.6 + ln 2).
    expect(eggleton(1)).toBeCloseTo(0.49 / (0.6 + Math.log(2)), 12);
    // Small q: it tends to 0.49 q^(1/3).
    expect(eggleton(1e-9) / (0.49 * Math.cbrt(1e-9))).toBeCloseTo(1, 3);
    // Eggleton's own table: 0.2068 at q = 0.1.
    expect(eggleton(0.1)).toBeCloseTo(0.2068, 4);
    // The heavier body's lobe is the larger.
    expect(eggleton(2)).toBeGreaterThan(eggleton(0.5));
    expect(fillSeparation(0.5, 1, 1, 1)).toBeCloseTo(0.5 / eggleton(1), 12);
  });

  it('finds L1 where the pull along the line of centers vanishes', () => {
    expect(l1Distance(1, 1)).toBeCloseTo(0.5, 12);
    for (const [m1, m2] of [[1, 0.5], [0.3, 1], [1, 1e-3]]) {
      const d = l1Distance(m1, m2);
      const M = m1 + m2;
      // In the frame of the Roche potential (G = a = 1): body 1 at -m2 / M, L1 d from it.
      const x = -m2 / M + d;
      const h = 1e-6;
      const slope = (rochePotential(m1, m2, 1, x + h, 0, 0) - rochePotential(m1, m2, 1, x - h, 0, 0)) / (2 * h);
      expect(Math.abs(slope)).toBeLessThan(1e-6);
      // A saddle: a maximum along x.
      expect(rochePotential(m1, m2, 1, x, 0, 0)).toBeGreaterThan(rochePotential(m1, m2, 1, x + 0.01, 0, 0));
    }
    // Hill's radius about a light body: (m2 / 3 m1)^(1/3) from it.
    expect(1 - l1Distance(1, 1e-6)).toBeCloseTo(Math.cbrt(1e-6 / 3), 4);
  });

  it('turns a softened binary as binaryMasses moves it', () => {
    expect(binaryOmega(1, 1)).toBe(1);
    expect(binaryOmega(8, 2, 0)).toBeCloseTo(1, 12);
    expect(binaryOmega(1, 1, 0.1)).toBeLessThan(1);
  });
});

const star: Part = { kind: 'star', id: 'donor', type: 'polytrope', n: 1.5, mass: 1, radius: 0.5 };
const companion: Part = { kind: 'mass', id: 'acc', gm: 0.5, soft: 0.05, sink: 0.05 };
const binary = (orbit: Partial<OrbitPart> = {}, parts: Part[] = [star, companion]): SetupSpec => ({
  ...emptySetup(),
  domain: { ndim: 3, n: 400, amr: 0, box: 4 },
  physics: { gravity: 'isolated', G: 1 },
  parts: [...parts, { kind: 'orbit', of: ['donor', 'acc'], type: 'binary', fill: 1, ...orbit }],
});

describe('a binary with a star', () => {
  it('puts the star against its Roche lobe, the center of mass at the box center', () => {
    const b = buildBodies(binary(), { G: 1, dx: 0.01, ndim: 3, box: 4 });
    const a = 0.5 / eggleton(2);
    expect(b.binary!.separation).toBeCloseTo(a, 12);
    const st = b.stars[0];
    const m = b.masses[0];
    expect(st.center[0]).toBeCloseTo((-0.5 / 1.5) * a, 12);
    expect(m.pos[0]).toBeCloseTo((1 / 1.5) * a, 12);
    expect(st.M * st.center[0] + m.gm * m.pos[0]).toBeCloseTo(0, 12);
    // Tidally locked: the star's center moves as the orbit turns it, and its gas turns about it.
    const w = binaryOmega(1.5, a, 0.05);
    expect(b.binary!.omega).toBeCloseTo(w, 12);
    expect(st.vel[1]).toBeCloseTo(w * st.center[0], 12);
    expect(m.vel[1]).toBeCloseTo(w * m.pos[0], 12);
    expect(st.spin).toBeCloseTo(w, 12);
    expect(b.binary!.l1.get('donor')).toBeCloseTo(l1Distance(1, 0.5) * a, 12);
  });

  it('starts every body at rest in the corotating frame, which the run turns with', () => {
    const spec = binary({ corotating: true });
    const b = buildBodies(spec, { G: 1, dx: 0.01, ndim: 3, box: 4 });
    expect(b.stars[0].vel).toEqual([0, 0, 0]);
    expect(b.stars[0].spin).toBeUndefined();
    expect(b.masses[0].vel).toEqual([0, 0, 0]);
    const { problem } = compileSetup(spec);
    const r = problem.rotation!({}, 3);
    expect(r.omega[0]).toBe(0);
    expect(r.omega[2]).toBeCloseTo(b.binary!.omega, 12);
    expect(compileSetup(binary()).problem.rotation).toBeUndefined();
    // A parameter that turns the frame off leaves the run inertial.
    const off = { ...spec, featured: [{ path: 'parts.2.corotating', id: 'corot', label: 'Corotating', tip: '', min: 0, max: 1, control: 'checkbox' as const }] };
    expect(compileSetup(off).problem.rotation!({ corot: 0 }, 3).omega).toEqual([0, 0, 0]);
  });

  it('holds a star where the orbit put it while it settles', () => {
    const set = compileSetup(binary()).problem.settle!({}, eos);
    const st = set.stars[0];
    const c = [...st.center];
    // Where the orbit put it, nothing pulls it: the companion and the centrifugal term balance.
    const still = holdPull(set, 0, [c], [0, 0, 0]);
    for (const v of still) expect(Math.abs(v)).toBeLessThan(2e-3 * st.stiff!);
    // Strayed toward L1, it is pulled back: the tide that slid it is cancelled, and more.
    const sum = [0, 0, 0];
    const back = holdPull(set, 0, [[c[0] + 0.1, c[1], c[2]]], sum);
    const h = 1e-5;
    const tide = (binaryPhi(set.binary!, c[0] + 0.1 + h, 0, 0) - binaryPhi(set.binary!, c[0] + 0.1 - h, 0, 0)) / (2 * h);
    expect(back[0]).toBeCloseTo(-tide + (0.3 + 0.05) * st.stiff! * 0.1, 6);
    expect(sum[0]).toBeCloseTo(0.05 * st.stiff! * 0.1, 12);
  });

  it('settles the star in the Roche potential, out to its L1 point', () => {
    const spec = binary();
    const set = compileSetup(spec).problem.settle!({}, eos);
    const b = buildBodies(spec, { G: 1, dx: 0, ndim: 3, box: 4 });
    expect(set.binary!.omega).toBeCloseTo(b.binary!.omega, 12);
    expect(set.binary!.masses).toHaveLength(1);
    expect(set.stars[0].reach).toBeCloseTo(b.binary!.l1.get('donor')!, 12);
    expect(set.stars[0].spin).toBeCloseTo(b.binary!.omega, 12);
    expect(set.stars[0].stiff).toBeCloseTo(1 / 0.5 ** 3, 12);
    // The grid holds the star's own pull: the rest is the companion's and the centrifugal term.
    const a = b.binary!.separation;
    const x1 = b.stars[0].center[0];
    for (const [x, y, z] of [[0.1, 0.2, 0], [x1 + 0.3, -0.1, 0.2], [-1, 0.5, 0.4]]) {
      const own = -1 / Math.hypot(x - x1, y, z);
      expect(binaryPhi(set.binary!, x, y, z) + own).toBeCloseTo(rochePotential(1, 0.5, a, x, y, z, 0.05), 10);
    }
  });

  it('takes a point-mass companion’s mass from q against the star’s', () => {
    const b = buildBodies(binary({ q: 0.25 }), { G: 2, dx: 0.01, ndim: 3, box: 4 });
    expect(b.masses[0].gm).toBeCloseTo(2 * 0.25, 12);
    expect(b.binary!.separation).toBeCloseTo(0.5 / eggleton(4), 12);
  });

  it('places two stars, each reaching to L1, and an eccentric binary as point masses are placed', () => {
    const second: Part = { kind: 'star', id: 'acc', type: 'polytrope', n: 1.5, mass: 0.5, radius: 0.3 };
    const two = buildBodies(binary({ fill: undefined, separation: 1.5 }, [star, second]), { G: 1, dx: 0.01, ndim: 3, box: 4 });
    const d1 = two.binary!.l1.get('donor')!;
    expect(d1 + two.binary!.l1.get('acc')!).toBeCloseTo(1.5, 12);
    expect(two.stars[1].center[0] - two.stars[0].center[0]).toBeCloseTo(1.5, 12);
    const ecc = buildBodies(binary({ fill: undefined, separation: 1.5, ecc: 0.3 }), { G: 1, dx: 0.01, ndim: 3, box: 4 });
    expect(ecc.binary).toBeUndefined();
    // At pericenter, a (1 - e) apart.
    expect(ecc.masses[0].pos[0] - ecc.stars[0].center[0]).toBeCloseTo(1.5 * 0.7, 12);
    expect(ecc.stars[0].spin).toBeUndefined();
  });

  it('leaves a binary of two point masses as it was, unless it turns the frame', () => {
    const masses: Part[] = [{ kind: 'mass', id: 'donor', gm: 0.5, soft: 0.06 }, { kind: 'mass', id: 'acc', gm: 0.5, soft: 0.06 }];
    const spec = binary({ fill: undefined, separation: 1, q: 1 }, masses);
    const b = buildBodies(spec, { G: 1, dx: 0.01, ndim: 3, box: 4 });
    expect(b.binary).toBeUndefined();
    expect(b.masses[1].vel[1]).toBeGreaterThan(0);
    const turning = buildBodies(binary({ fill: undefined, separation: 1, q: 1, corotating: true }, masses), { G: 1, dx: 0.01, ndim: 3, box: 4 });
    expect(turning.masses[1].vel).toEqual([0, 0, 0]);
    expect(turning.binary!.omega).toBeCloseTo(binaryOmega(1, 1, 0.06), 12);
  });
});

describe('a frame that follows a binary', () => {
  const spec = binary({ corotating: true, track: true });

  it('follows the axis of the stars’ quadrupole, counting gas denser than where the stars end', () => {
    const b = buildBodies(spec, { G: 1, dx: 0.01, ndim: 3, box: 4 });
    expect(b.binary!.track).toBe(true);
    // No tracers, no extra masses: the gas itself says where the stars are.
    expect(b.massIds).toEqual(['acc']);
    const r = compileSetup(spec).problem.rotation!({}, 3);
    const cut = compileSetup(spec).problem.settle!({}, eos).rhoCut;
    expect(r.track!.rhoMin).toBeCloseTo(cut, 12);
    // Two point masses: their quadrupole alone.
    const masses: Part[] = [{ kind: 'mass', id: 'donor', gm: 0.5, soft: 0.06 }, { kind: 'mass', id: 'acc', gm: 0.5, soft: 0.06 }];
    const pair = compileSetup(binary({ fill: undefined, separation: 1, q: 1, corotating: true, track: true }, masses)).problem.rotation!({}, 3);
    expect(pair.track!.rhoMin).toBeGreaterThan(1e37);
  });

  it('starts at the orbit’s angular velocity, along the line as it lies, and is carried in the masses’ block', () => {
    const problem = compileSetup(spec).problem;
    expect(problem.rotation!({}, 3).track!.rhoMin).toBeGreaterThan(0);
    const c = configFor(problem, 400);
    expect(followsBinary(c)).toBe(true);
    const masses = problem.masses!(c.params, c.G, 0.01, 3).map((m) => completeMass(m, 0.01));
    const t = frameTrack(c, masses);
    const w = buildBodies(spec, { G: 1, dx: 0.01, ndim: 3, box: 4 }).binary!.omega;
    expect(t[0]).toBeCloseTo(w, 6);
    expect(t.slice(1, 4)).toEqual([0, t[2], 1]);
    expect(t[2]).toBeCloseTo(w, 6);
    // The axis is taken on the GPU before the first step; the threshold rides along.
    expect(t.slice(4)).toEqual([0, 0, problem.rotation!({}, 3).track!.rhoMin, 0]);
    const packed = packMasses(masses, t);
    expect(Array.from(packed.subarray(4, MASS_HEADER))).toEqual(Array.from(new Float32Array(t)));
    expect(unpackMasses(packed, 1)[0].gm).toBeCloseTo(masses[0].gm, 6);
    // A snapshot's frame comes back as it was.
    const saved = [0.4, 1e-4, w, 1, 0.6, 0.8, -1, 0];
    expect(frameTrack({ ...c, restart: { time: 1, steps: 10, frame: saved } }, masses)).toEqual(saved);
    // Without tracking, nothing.
    const plain = configFor(compileSetup(binary({ corotating: true })).problem, 400);
    expect(followsBinary(plain)).toBe(false);
    expect(frameTrack(plain, masses)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('is checked: corotating only; two stars without cores are fine', () => {
    const texts = (s: SetupSpec) => checkSetup(s).map((i) => `${i.level}: ${i.text}`);
    expect(texts(spec).filter((t) => /follow|inside/.test(t))).toEqual([]);
    expect(texts(binary({ track: true })).some((t) => /^warning: .*only in the corotating frame/.test(t))).toBe(true);
    const second: Part = { kind: 'star', id: 'acc', type: 'polytrope', n: 1.5, mass: 0.5, radius: 0.3 };
    const stars = binary({ fill: undefined, separation: 1.5, corotating: true, track: true }, [star, second]);
    expect(texts(stars).filter((t) => /^error/.test(t))).toEqual([]);
  });
});

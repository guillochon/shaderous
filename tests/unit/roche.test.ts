// Binaries with stars in them: the Roche geometry (src/problems/roche.ts), how a binary orbit
// places a star (setup/parts/bodies.ts) and what the compiled setup hands the run.
import { describe, expect, it } from 'vitest';
import { defaultEosParams, makeEos } from '../../src/eos/eos';
import { binaryOmega, eggleton, fillSeparation, innerLagrange, l1Distance, lobePotential, rochePotential } from '../../src/problems/roche';
import { binaryPhi, holdPull, orbitOmega, Settler, turnBinary, type SettleCells } from '../../src/problems/settle';
import { compileSetup } from '../../src/setup/compile';
import { buildBodies } from '../../src/setup/parts/bodies';
import { checkSetup } from '../../src/setup/check';
import { completeMass, configFor, followsBinary, frameTrack, MASS_HEADER, orbitConfig, packMasses, rotationOf, settledOrbit, unpackMasses } from '../../src/sim';
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

  it('finds L1 as the top of the pass between two bodies, and bounds each lobe by its level', () => {
    for (const [m1, m2] of [[1, 0.6], [0.3, 1], [1, 1e-2]]) {
      const M = m1 + m2;
      const bodies = [{ gm: m1, at: [-m2 / M, 0, 0], soft: 0 }, { gm: m2, at: [m1 / M, 0, 0], soft: 0 }];
      const L1 = innerLagrange(bodies, Math.sqrt(M));
      expect(L1.at[0]).toBeCloseTo(-m2 / M + l1Distance(m1, m2), 6);
      expect(L1.phi).toBeCloseTo(rochePotential(m1, m2, 1, L1.at[0], 0, 0), 12);
      // Each lobe lies within the sphere of its body's distance to L1, and is the whole of what is
      // under L1's level in that sphere: along any ray from the body the potential crosses it once.
      bodies.forEach((b, k) => {
        const r1 = Math.abs(L1.at[0] - b.at[0]);
        for (let t = 0; t < 40; t++) {
          const th = Math.acos(1 - (2 * (t + 0.5)) / 40);
          const ph = 2.4 * t;
          const dir = [Math.cos(th), Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph)];
          let out = false;
          for (let s = 1; s <= 200; s++) {
            const r = (s / 200) * r1;
            const x = b.at.map((c, a) => c + r * dir[a]);
            const inside = lobePotential(bodies, Math.sqrt(M), x[0], x[1], x[2]) <= L1.phi;
            if (!inside) out = true;
            else expect(out, `body ${k}: back inside after leaving`).toBe(false);
          }
        }
      });
    }
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
    // L1 of the softened companion's pull: near the point masses' own.
    expect(b.binary!.l1.get('donor')).toBeCloseTo(l1Distance(1, 0.5) * a, 2);
    const L1 = innerLagrange(b.binary!.lobe!.bodies, w);
    expect(b.binary!.l1.get('donor')).toBeCloseTo(L1.at[0] - st.center[0], 12);
    expect(b.binary!.lobe!.phi).toBeCloseTo(L1.phi, 12);
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
    expect(set.stars[0].lobe).toBe(true);
    expect(set.binary!.lobe).toEqual(b.binary!.lobe);
    // The grid holds the star's own pull: the rest is the companion's and the centrifugal term.
    const a = b.binary!.separation;
    const x1 = b.stars[0].center[0];
    for (const [x, y, z] of [[0.1, 0.2, 0], [x1 + 0.3, -0.1, 0.2], [-1, 0.5, 0.4]]) {
      const own = -1 / Math.hypot(x - x1, y, z);
      expect(binaryPhi(set.binary!, x, y, z) + own).toBeCloseTo(rochePotential(1, 0.5, a, x, y, z, 0.05), 10);
    }
  });

  it('takes no cell outside the Roche lobe, however much the star overfills it', () => {
    const set = compileSetup(binary({ fill: 1.15 })).problem.settle!({}, eos);
    const st = set.stars[0];
    const lobe = set.binary!.lobe!;
    const R = st.reach!;
    const n = 48;
    const h = (2.2 * R) / n;
    const pos: [number, number, number][] = [];
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) pos.push([st.center[0] + (i + 0.5) * h - 1.1 * R, (j + 0.5) * h - 1.1 * R, (k + 0.5) * h - 1.1 * R]);
    // The grid's potential: the star's own, as a point.
    const phi = new Float32Array(pos.map(([x, y, z]) => -1 / Math.max(Math.hypot(x - st.center[0], y, z), h)));
    const settle = (inLobe: boolean) => {
      const u1 = new Float32Array(pos.length * 4);
      const u2 = new Float32Array(pos.length * 4);
      // The star as painted, a sphere heavy enough to overfill its lobe.
      pos.forEach(([x, y, z], q) => (u1[4 * q] = Math.hypot(x - st.center[0], y, z) < st.R ? 4 : set.rhoAmb));
      const cells: SettleCells = { count: pos.length, pos: (q) => pos[q], vol: () => h ** 3, u1, u2 };
      const s = new Settler({ ...set, stars: [{ ...st, lobe: inLobe }] }, cells);
      for (let r = 0; r < 4; r++) s.round(phi);
      let outside = 0;
      let filled = 0;
      pos.forEach(([x, y, z], q) => {
        if (!(u1[4 * q] > set.rhoCut)) return;
        filled++;
        if (lobePotential(lobe.bodies, set.binary!.omega, x, y, z) > lobe.phi) outside++;
      });
      return { outside, filled };
    };
    // Bounded by its reach alone, it spills past the lobe around L1; bounded by the lobe, not a cell.
    expect(settle(false).outside).toBeGreaterThan(100);
    const inLobe = settle(true);
    expect(inLobe.filled).toBeGreaterThan(1000);
    expect(inLobe.outside).toBe(0);
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

  it('released after a merger: phase 3 from the start, and a snapshot saved following the axis is released from there', () => {
    const released = binary({ corotating: true, track: true, release: true });
    const problem = compileSetup(released).problem;
    expect(problem.rotation!({}, 3).track!.release).toBe(true);
    expect(compileSetup(spec).problem.rotation!({}, 3).track!.release).toBeUndefined();
    const c = configFor(problem, 400);
    const masses = problem.masses!(c.params, c.G, 0.01, 3).map((m) => completeMass(m, 0.01));
    const t = frameTrack(c, masses);
    expect(t[3]).toBe(3);
    expect(t[7]).toBe(0);
    // Saved following the axis: released from the next step, with e taken afresh.
    const w = t[2];
    expect(frameTrack({ ...c, restart: { time: 1, steps: 10, frame: [0.4, 1e-4, w, 2, 0.6, 0.8, 0.01, 0] } }, masses)).toEqual([0.4, 1e-4, w, 4, 0.6, 0.8, 0.01, -1]);
    // Saved released, part way or at rest: as it was.
    for (const saved of [[0.2, -1e-3, w, 4, 0.6, 0.8, 0.01, 0.35], [0, 0, w, 5, 0.6, 0.8, 0.01, 0.1]]) {
      expect(frameTrack({ ...c, restart: { time: 1, steps: 10, frame: saved } }, masses)).toEqual(saved);
    }
  });

  it('is checked: corotating only; two stars without cores are fine', () => {
    const texts = (s: SetupSpec) => checkSetup(s).map((i) => `${i.level}: ${i.text}`);
    expect(texts(spec).filter((t) => /follow|inside/.test(t))).toEqual([]);
    expect(texts(binary({ track: true })).some((t) => /^warning: .*only in the corotating frame/.test(t))).toBe(true);
    expect(texts(binary({ corotating: true, release: true })).some((t) => /^warning: .*only a frame that follows the orbit is released/.test(t))).toBe(true);
    expect(texts(binary({ corotating: true, track: true, release: true })).filter((t) => /released/.test(t))).toEqual([]);
    const second: Part = { kind: 'star', id: 'acc', type: 'polytrope', n: 1.5, mass: 0.5, radius: 0.3 };
    const stars = binary({ fill: undefined, separation: 1.5, corotating: true, track: true }, [star, second]);
    expect(texts(stars).filter((t) => /^error/.test(t))).toEqual([]);
  });
});

describe('a binary’s orbit, found as its stars settle', () => {
  const second: Part = { kind: 'star', id: 'acc', type: 'polytrope', n: 1.5, mass: 0.5, radius: 0.35 };
  const pair = (orbit: Partial<OrbitPart> = {}) => binary({ fill: 0.9, corotating: true, ...orbit }, [star, second]);

  it('asks for it in the frame turning with the binary, and in an inertial one when only stars orbit', () => {
    expect(compileSetup(pair()).problem.settle!({}, eos).binary!.orbit).toBe(true);
    expect(compileSetup(pair({ corotating: false })).problem.settle!({}, eos).binary!.orbit).toBe(true);
    expect(compileSetup(binary({ corotating: true })).problem.settle!({}, eos).binary!.orbit).toBe(true);
    // A point mass's velocity is not the settling's to change.
    expect(compileSetup(binary()).problem.settle!({}, eos).binary!.orbit).toBeUndefined();
  });

  it('finds the Omega that balances gravity, whatever Omega the pulls were measured at', () => {
    const set = compileSetup(pair()).problem.settle!({}, eos);
    const [a, b] = set.stars;
    const coms = [a.center, b.center];
    const sep = b.center[0] - a.center[0];
    // Gravity 4% stronger than the points': g_i = -(1.04) G M_j / sep^2 toward the other.
    const g = [(1.04 * b.gm!) / sep ** 2, (-1.04 * a.gm!) / sep ** 2];
    for (const w of [set.binary!.omega, 0.5 * set.binary!.omega]) {
      const pulls = coms.map((c, i) => [g[i] + w * w * c[0], w * w * c[1], 0]);
      expect(orbitOmega(set, coms, pulls, w)).toBeCloseTo(Math.sqrt(1.04) * binaryOmega(1.5, sep), 10);
    }
    // No inward pull: no orbit to speak of.
    expect(orbitOmega(set, coms, coms.map(() => [0, 0, 0]), 0)).toBeNull();
  });

  it('turns the binary: L1 found again, and stars that move in an inertial frame scaled to it', () => {
    const set = compileSetup(pair({ corotating: false })).problem.settle!({}, eos);
    const w0 = set.binary!.omega;
    const vel = [...set.stars[1].vel];
    const lobeVel = set.stars[1].spin!;
    turnBinary(set, 1.02 * w0);
    expect(set.binary!.omega).toBeCloseTo(1.02 * w0, 12);
    expect(set.binary!.lobe!.phi).toBeCloseTo(innerLagrange(set.binary!.lobe!.bodies, 1.02 * w0).phi, 12);
    set.stars[1].vel.forEach((v, k) => expect(v).toBeCloseTo(1.02 * vel[k], 12));
    expect(set.stars[1].spin).toBeCloseTo(1.02 * lobeVel, 12);
    // At rest in the frame turning with the binary: still at rest.
    const co = compileSetup(pair()).problem.settle!({}, eos);
    turnBinary(co, 1.02 * co.binary!.omega);
    expect(co.stars.map((s) => s.vel)).toEqual([[0, 0, 0], [0, 0, 0]]);
  });

  it('settles two stars at the Omega their own pull asks for, holding each where the orbit put it', () => {
    const settle = (orbit: boolean) => {
      const set = compileSetup(pair()).problem.settle!({}, eos);
      if (!orbit) delete set.binary!.orbit;
      const [a, b] = set.stars;
      const n = 64;
      const h = 4 / n;
      const pos: [number, number, number][] = [];
      for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) pos.push([(i + 0.5) * h - 2, (j + 0.5) * h - 2, (k + 0.5) * h - 2]);
      const own = pos.map(([x, y, z]) => (Math.hypot(x - a.center[0], y, z) / a.R < Math.hypot(x - b.center[0], y, z) / b.R ? 0 : 1));
      const u1 = new Float32Array(pos.length * 4);
      const u2 = new Float32Array(pos.length * 4);
      pos.forEach(([x, y, z], q) => (u1[4 * q] = Math.hypot(x - a.center[0], y, z) < a.R ? 3 : Math.hypot(x - b.center[0], y, z) < b.R ? 3 : set.rhoAmb));
      const cells: SettleCells = { count: pos.length, pos: (q) => pos[q], vol: () => h ** 3, u1, u2 };
      // The grid's potential, after each round: each star's own as a point at its center of mass
      // (self-gravity moves with the star, and does not pull it), and its companion's 5% stronger
      // than a point's.
      const gm = [a.gm!, b.gm!];
      const potential = () => {
        const c = [[0, 0, 0, 0], [0, 0, 0, 0]];
        pos.forEach((x, q) => {
          const r = u1[4 * q];
          if (!(r > set.rhoCut)) return;
          const s4 = c[own[q]];
          s4[0] += r;
          for (let k = 0; k < 3; k++) s4[k + 1] += r * x[k];
        });
        const at = c.map((s4) => [s4[1] / s4[0], s4[2] / s4[0], s4[3] / s4[0]]);
        // (Its own as a uniform sphere of its radius, which a uniform pull moves as it moves a star.)
        const R = [a.R, b.R];
        const pt = (i: number, [x, y, z]: number[]) => {
          const r = Math.hypot(x - at[i][0], y - at[i][1], z - at[i][2]);
          return r < R[i] ? (-gm[i] * (3 * R[i] ** 2 - r * r)) / (2 * R[i] ** 3) : -gm[i] / r;
        };
        return { at, phi: new Float32Array(pos.map((x, q) => pt(own[q], x) + 1.05 * pt(1 - own[q], x))) };
      };
      const w0 = set.binary!.omega;
      const s = new Settler(set, cells);
      for (let r = 0; r < 40; r++) s.round(potential().phi);
      // The Omega that balances the gravity on the settled stars, summed cell by cell: each
      // star's mean pull from its companion (its own pulls it nowhere) against Omega^2 r.
      const { at } = potential();
      const f = [[0, 0, 0], [0, 0, 0]];
      pos.forEach(([x, y], q) => {
        const r = u1[4 * q];
        if (!(r > set.rhoCut)) return;
        const i = own[q];
        const j = 1 - i;
        const dx = x - at[j][0];
        const dy = y - at[j][1];
        const d = Math.hypot(dx, dy, pos[q][2] - at[j][2]);
        f[i][0] += r;
        f[i][1] -= (r * 1.05 * gm[j] * dx) / d ** 3;
        f[i][2] -= (r * 1.05 * gm[j] * dy) / d ** 3;
      });
      const g = f.map((v) => [v[1] / v[0], v[2] / v[0]]);
      const num = -gm.reduce((acc, m, i) => acc + m * (g[i][0] * at[i][0] + g[i][1] * at[i][1]), 0);
      const den = gm.reduce((acc, m, i) => acc + m * (at[i][0] ** 2 + at[i][1] ** 2), 0);
      return { w0, w: set.binary!.omega, balance: Math.sqrt(num / den), at, centers: [a.center, b.center] };
    };
    const held = settle(false);
    expect(held.w).toBe(held.w0);
    // The companions' pull 5% stronger than the points': Omega about sqrt(1.05) times theirs, more
    // for the tides (a star stretched toward its companion is pulled harder than its center), and
    // the Omega that balances the pull on the settled stars.
    const found = settle(true);
    expect(found.w / found.w0).toBeGreaterThan(Math.sqrt(1.05));
    expect(found.w / found.w0).toBeLessThan(1.05);
    // (To 0.2%: the stand-in for self-gravity, a sphere, pulls on the gas a stretched star puts
    // outside it, which a star's own pull does not.)
    expect(Math.abs(found.w / found.balance - 1)).toBeLessThan(2e-3);
    found.at.forEach((c, i) => expect(Math.abs(c[0] - found.centers[i][0])).toBeLessThan(0.01));
  });

  it('is the frame’s Omega in the run, and a snapshot’s', () => {
    const problem = compileSetup(pair({ track: true })).problem;
    const c = configFor(problem, 400);
    const w0 = rotationOf(c)[2];
    expect(rotationOf({ ...c, orbitOmega: 1.01 * w0 })[2]).toBeCloseTo(1.01 * w0, 12);
    expect(rotationOf({ ...c, restart: { time: 1, steps: 1, orbitOmega: 1.01 * w0 } })[2]).toBeCloseTo(1.01 * w0, 12);
    expect(frameTrack(orbitConfig(c, 1.01 * w0))[0]).toBeCloseTo(1.01 * w0, 6);
    expect(orbitConfig(c, undefined)).toBe(c);
    // After a settling: taken when the frame is the binary's, and the settling turned it.
    const set = problem.settle!({}, eos);
    expect(settledOrbit(c, set, set.binary!.omega)).toBeUndefined();
    const before = set.binary!.omega;
    turnBinary(set, 1.01 * before);
    expect(settledOrbit(c, set, before)).toBeCloseTo(1.01 * before, 12);
    // An inertial run's frame does not turn.
    const still = configFor(compileSetup(pair({ corotating: false })).problem, 400);
    expect(settledOrbit(still, set, before)).toBeUndefined();
  });
});

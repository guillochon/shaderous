// Random setups for Shaderous, run in the page: injected by tools/random-test/run.mjs, or pasted
// into a browser console on the simulator. Defines window.randomTest:
//
//   randomTest.presetSetup(seed)    a preset with random grid and solver options (link keys)
//   randomTest.composedSetup(seed)  a setup document assembled from parts: geometry, gas
//                                   regions, obstacles, stars, point masses and disks, fields,
//                                   frames, cosmology, shear, mirrors, per-block timesteps,
//                                   radiation transport, implicit pressure
//   randomTest.runOne(seed, opts)   load one (alternating presets and documents by seed parity
//                                   unless opts.kind says), run it, return a row with flags
//   randomTest.runBatch(seeds, opts) runOne over a list; the rows also land in randomTest.rows
//
// A row is flagged when its values go non-finite, more than 1% of cells hit the floors, the
// Mach number passes 100, div B passes 1e-4, the page reports an error, or (for a document whose
// faces are all walls or periodic, with no sink) the mass changes by more than 1e-4. Floors and
// high Mach numbers alone are common in setups with vacuum or cold gas: read those rows before
// calling them bugs. Documents that the page's own check refuses are skipped and listed.
//
// Every choice is a deterministic function of the seed, so a row's seed reproduces its setup:
// randomTest.composedSetup(seed).doc is the document, presetSetup(seed) the link keys.
(() => {
  const rng = (seed) => {
    // Mixed first, so that neighboring seeds give unrelated setups.
    let s = Math.imul((seed >>> 0) ^ 0x9e3779b9, 2654435761) >>> 0;
    s = Math.imul(s ^ (s >>> 15), 2246822519) >>> 0;
    const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
    return { rnd, pick: (a) => a[Math.floor(rnd() * a.length)], U: (a, b) => a + (b - a) * rnd() };
  };

  /** A preset with random dimensions, grid, refinement, solver and mirrors. */
  async function presetSetup(seed) {
    const { rnd, pick } = rng(seed);
    const d = await window.shaderous.describe();
    const p = pick(d.problems);
    const dim = p.supports2d ? (rnd() < 0.5 ? 2 : 3) : 3;
    const setup = { problem: p.id, dim };
    if (dim === 2) {
      setup.n = pick([128, 256, 512]);
      setup.amr = pick([0, 0, 1, 2]);
    } else {
      const amr = pick([0, 0, 1, 2]);
      setup.amr = amr;
      setup.n = amr ? pick([128, 256]) : pick([64, 64, 128]);
      // A run that cannot refine falls back to the uniform grid: keep it at 128^3 or less.
      if (/shear|mri-box|spherical/.test(p.id)) setup.n = Math.min(setup.n, 128);
    }
    if (rnd() < 0.3) setup.riemann = pick(d.riemann);
    if (rnd() < 0.3) setup.limiter = pick(d.limiter);
    if (rnd() < 0.25) setup.sym = pick(['z', 'xy', 'xyz', 'x']);
    return setup;
  }

  /** A setup document from parts. Returns { doc, kinds, geometry, ndim, scenario }. */
  function composedSetup(seed) {
    const { rnd, pick, U } = rng(seed);
    const geometry = pick(['cartesian', 'cartesian', 'cartesian', 'cylindrical', 'spherical']);
    const curved = geometry !== 'cartesian';
    const ndim = curved ? 2 : pick([2, 2, 3]);
    const amr = pick([0, 0, 1, 2]);
    const n = ndim === 2 ? pick([128, 256, 512]) : pick([64, 128]);
    const doc = {
      format: 'shaderous-setup', version: 1, name: `random ${seed}`,
      domain: { ndim, n, amr, box: 1, geometry, supports2d: true },
      physics: { eos: 'ideal', eosParams: { gamma: pick([5 / 3, 1.4, 1.1]) }, gravity: 'off', G: 0 },
      background: { rho: 1, p: pick([1e-4, 0.01, 1]) },
      parts: [],
      run: { tEnd: 1 },
    };
    if (geometry === 'spherical') {
      doc.domain.rMin = pick([0.005, 0.02, 0.05]);
      doc.domain.box = 0.5;
      doc.domain.bc = { lo: [2, 2, 0], hi: [1, 2, 0] };
      if (rnd() < 0.4) doc.domain.logR = true;
    } else if (geometry === 'cylindrical') {
      doc.domain.bc = { lo: [2, pick([1, 2]), 0], hi: [1, pick([1, 2]), 0] };
    } else {
      const b = pick([0, 1, 2]);
      doc.domain.bc = { lo: [b, b, b], hi: [b, b, b] };
    }
    const kinds = [];
    const nParts = 1 + Math.floor(rnd() * 3);
    // In (R, z) and (r, theta) runs a body sits on the axis (y), and v_phi is the third component.
    const centre = () => (curved ? [0, geometry === 'spherical' ? 0 : U(-0.2, 0.2), 0] : [U(-0.25, 0.25), U(-0.25, 0.25), ndim === 3 ? U(-0.25, 0.25) : 0]);
    const scenario = pick(['gas', 'gas', 'gas', 'stars', 'disk']);
    if (scenario === 'stars' && (curved || ndim === 3)) {
      doc.physics.gravity = 'isolated';
      doc.physics.G = 1;
      doc.background = { rho: 1e-6, p: 1e-8 };
      if (geometry === 'cylindrical') doc.domain.bc = { lo: [2, 1, 0], hi: [1, 1, 0] };
      else if (geometry === 'cartesian') doc.domain.bc = { lo: [1, 1, 1], hi: [1, 1, 1] };
      const m = U(0.05, 0.2);
      const r = U(0.06, 0.12);
      if (geometry === 'spherical') {
        doc.parts.push({ kind: 'mass', id: 'c', gm: m, pos: [0, 0, 0] }, { kind: 'region', shape: { type: 'sphere', center: [0, 0, 0], radius: r }, rho: U(1, 10) });
        kinds.push('mass+gas');
      } else if (curved || rnd() < 0.4) {
        doc.parts.push({ kind: 'star', id: 'a', type: 'polytrope', n: pick([1, 1.5, 3]), mass: m, radius: r, pos: [0, 0, 0] });
        kinds.push('star');
      } else {
        doc.parts.push(
          { kind: 'star', id: 'a', type: 'polytrope', n: 1.5, mass: m, radius: r },
          { kind: 'star', id: 'b', type: 'polytrope', n: 1.5, mass: m * U(0.3, 1), radius: r * U(0.6, 1) },
          { kind: 'orbit', of: ['a', 'b'], type: 'collision', separation: U(0.25, 0.4), impact: U(0, 1) },
        );
        kinds.push('star-pair');
      }
      if (rnd() < 0.3 && !curved) {
        doc.physics.rotation = { omega: [0, 0, U(0.1, 1)], centrifugal: true };
        kinds.push('rot');
      }
    } else if (scenario === 'disk' && geometry !== 'spherical') {
      doc.background = { rho: 1e-5, p: 1e-8 };
      doc.domain.bc = curved ? { lo: [2, 1, 0], hi: [1, 1, 0] } : { lo: [1, 1, 1], hi: [1, 1, 1] };
      const profile = ndim === 3 ? pick(['thin', 'balanced']) : 'thin';
      doc.parts.push({ kind: 'mass', id: 'bh', gm: 1, pos: [0, 0, 0], soft: 0.02, sink: 0.02 }, { kind: 'disk', about: ['bh'], profile, rIn: 0.08, rOut: 0.4, sigma0: 1, slope: -1, cs: pick([0.05, 0.1]), floor: 1e-4 });
      if (rnd() < 0.4 && profile === 'balanced') {
        doc.parts[1].loops = { beta: 100, span: 2, count: 2 };
        doc.physics.fluff = { rho: 0, u: 0, v: 0, va: 3 };
        kinds.push('loops');
      }
      kinds.push('disk');
    } else {
      for (let i = 0; i < nParts; i++) {
        const what = pick(['blast', 'blob', 'obstacle', 'shell', 'jet']);
        if (what === 'blast') doc.parts.push({ kind: 'region', shape: { type: 'sphere', center: centre(), radius: U(0.02, 0.06) }, energy: U(0.1, 2), s: 1 });
        else if (what === 'blob') doc.parts.push({ kind: 'region', shape: { type: 'sphere', center: centre(), radius: U(0.05, 0.15) }, rho: U(3, 30), v: curved ? (rnd() < 0.3 ? [0, 0, U(-1, 1)] : [0, U(-1, 1), 0]) : [U(-1, 1), U(-1, 1), 0], s: 1 });
        else if (what === 'obstacle' && geometry === 'cartesian') {
          const c = centre();
          const h = U(0.03, 0.08);
          doc.parts.push({ kind: 'obstacle', shape: rnd() < 0.5 ? { type: 'sphere', center: c, radius: h } : { type: 'box', min: c.map((x) => x - h), max: c.map((x) => x + h) } });
        } else if (what === 'shell') doc.parts.push({ kind: 'region', shape: { type: 'box', min: [-0.5, -0.5, -0.5], max: [0.5, U(-0.3, 0), 0.5] }, rho: U(0.1, 5), p: doc.background.p });
        else doc.parts.push({ kind: 'region', shape: { type: 'sphere', center: centre(), radius: 0.04 }, rho: 1, v: curved ? [0, 2, 0] : [2, 0, 0], p: doc.background.p * 10, s: 1 });
        kinds.push(what);
      }
      if (rnd() < 0.25 && geometry !== 'spherical') {
        doc.physics.gravity = 'constant';
        doc.physics.gext = [0, -0.5, 0];
        kinds.push('gext');
      }
      if (rnd() < 0.2 && geometry === 'cartesian') {
        doc.parts.push({ kind: 'field', b: [U(0, 0.3), U(0, 0.3), U(0, 0.3)] });
        kinds.push('field');
      }
      if (rnd() < 0.15 && geometry === 'cylindrical') {
        doc.parts.push({ kind: 'field', b: [0, U(0.05, 0.3), 0] });
        kinds.push('field-axis');
      }
      if (rnd() < 0.15 && geometry === 'cartesian') {
        // (No centrifugal force in a periodic box: it is not periodic.)
        doc.physics.rotation = { omega: [0, 0, U(0.5, 2)], centrifugal: rnd() < 0.5 && doc.domain.bc.lo[0] !== 0 };
        kinds.push('rot');
      }
    }
    if (rnd() < 0.2) {
      doc.physics.eos = 'gas_radiation';
      doc.physics.eosParams = { gamma: 5 / 3, arad: U(0.01, 1), R: 1 };
      kinds.push('gasrad');
    }
    if (rnd() < 0.15) {
      doc.physics.cooling = pick(['power', 'ism']);
      kinds.push('cool');
    }
    if (rnd() < 0.15 && !curved) {
      doc.physics.viscosity = U(1e-4, 1e-3);
      kinds.push('visc');
    }
    // The coordinate features: mirrors, comoving coordinates, a shearing box, per-block timesteps.
    const extra = rnd();
    if (geometry === 'cartesian' && extra < 0.12) {
      doc.domain.mirror = pick(['x', 'y', 'xy', 'xyz']);
      doc.domain.symmetric = doc.domain.mirror;
      kinds.push('mirror');
    } else if (geometry === 'cartesian' && extra < 0.2 && scenario === 'gas') {
      doc.domain.bc = { lo: [0, 0, 0], hi: [0, 0, 0] };
      doc.physics.cosmology = { omegaM: 1, omegaL: 0, zStart: pick([10, 50]), zEnd: 0 };
      doc.physics.gravity = ndim === 3 ? pick(['off', 'periodic']) : 'off';
      doc.physics.G = 1;
      delete doc.physics.gext;
      delete doc.physics.rotation;
      kinds.push('cosmo');
    } else if (geometry === 'cartesian' && extra < 0.28 && scenario === 'gas') {
      doc.domain.bc = { lo: [0, 0, 0], hi: [0, 0, 0] };
      doc.domain.amr = 0;
      doc.physics.shear = { omega: 1, q: 1.5 };
      doc.physics.gravity = 'off';
      delete doc.physics.gext;
      delete doc.physics.rotation;
      // A shearing box starts from its background flow.
      doc.parts.unshift({ kind: 'flow', name: 'shearing-noise', noise: 0.01, pressure: 0.6 });
      delete doc.background;
      kinds.push('shear');
    }
    if (doc.physics.shear || doc.physics.cosmology) {
      doc.parts = doc.parts.filter((p) => p.kind !== 'obstacle' && p.kind !== 'field');
      delete doc.physics.viscosity;
    }
    if (doc.domain.amr > 0 && !doc.physics.shear && !doc.physics.cosmology && rnd() < 0.4) {
      doc.mesh = { timeBins: pick([1, 2, 3]) };
      kinds.push('bins');
    }
    if (rnd() < 0.3) doc.physics.riemann = pick(['hllc', 'hllclm', 'hll']);
    if (rnd() < 0.3) doc.physics.limiter = pick(['koren', 'mc', 'vanleer', 'minmod']);
    // Radiation transport and implicit pressure, drawn last so that a seed's setup is otherwise
    // what it was before they were added. Each only where it can run at all; the page's check
    // turns away the rest.
    const newer = rnd();
    const magnetic = doc.parts.some((p) => p.kind === 'field' || p.loops);
    const frame = doc.physics.shear || doc.physics.cosmology;
    if (newer < 0.18 && !curved && !magnetic && !frame) {
      doc.physics.radiation = 'm1';
      doc.physics.opacity = 'constant';
      doc.physics.opacityParams = { absorption: pick([0.1, 10, 1000]), scattering: pick([0, 0, 1]) };
      doc.physics.radC = pick([100, 1000]);
      doc.physics.radChat = pick([20, 50]);
      if (doc.physics.eos === 'ideal') doc.physics.eosParams.arad = U(1e-3, 0.1);
      kinds.push('rad');
    } else if (newer < 0.32 && !curved && !frame && doc.physics.eos === 'ideal' && scenario !== 'disk' && !doc.physics.rotation && !doc.domain.mirror && !doc.physics.cooling) {
      doc.physics.pressure = 'implicit';
      delete doc.mesh;
      const at = kinds.indexOf('bins');
      if (at >= 0) kinds.splice(at, 1);
      kinds.push('implicit');
    }
    return { doc, kinds, geometry, ndim, scenario };
  }

  /** Load and run one setup; a row with what was run and its flags. */
  async function runOne(seed, opts = {}) {
    const s = window.shaderous;
    const kind = opts.kind ?? (seed % 2 ? 'document' : 'preset');
    const steps = opts.steps ?? 400;
    const row = { seed, kind };
    try {
      const e0 = s.errors().length;
      let closed = false;
      if (kind === 'preset') {
        row.setup = await presetSetup(seed);
        row.query = (await s.load(row.setup)).query;
      } else {
        const g = composedSetup(seed);
        row.kinds = g.kinds.join(',');
        row.geometry = g.geometry;
        const c = s.check(g.doc);
        if (!c.ok) {
          row.skipped = (c.issues ?? []).filter((i) => i.level === 'error').map((i) => i.text).join(' | ').slice(0, 300) || c.error;
          return row;
        }
        const k = g.ndim === 2 ? 2 : 3;
        const faces = [...(g.doc.domain.bc?.lo ?? []).slice(0, k), ...(g.doc.domain.bc?.hi ?? []).slice(0, k)];
        closed = faces.every((b) => b !== 1) && !g.doc.parts.some((p) => p.kind === 'mass' && p.sink);
        await s.load({}, g.doc);
        await window.shaderousApp?.sim?.settled;
      }
      const d0 = await s.diagnostics();
      const r = await s.run({ steps, timeoutMs: opts.timeoutMs ?? 25000 });
      const d = await s.diagnostics();
      const nums = [d.mass, d.eTot, d.maxMach, ...Object.values(d.ranges).flat()];
      const cfg = window.shaderousApp?.cfg;
      const cells = cfg ? cfg.n ** cfg.ndim : Infinity;
      row.t = +r.t.toPrecision(4);
      row.steps = r.steps;
      row.massChange = d0.mass > 0 ? +(d.mass / d0.mass - 1).toPrecision(3) : null;
      row.flags = [];
      if (!nums.every(Number.isFinite)) row.flags.push('non-finite');
      if (d.floored > 0.01 * cells) row.flags.push(`floored ${d.floored}`);
      if (d.maxMach > 100) row.flags.push(`mach ${d.maxMach.toPrecision(3)}`);
      if (d.magnetic && d.magnetic.divB > 1e-4) row.flags.push(`divB ${d.magnetic.divB}`);
      if (d.radiation && !Object.values(d.radiation).every(Number.isFinite)) row.flags.push('non-finite radiation');
      if (closed && Math.abs(row.massChange) > 1e-4) row.flags.push(`mass ${row.massChange} in a closed box`);
      const errs = s.errors().slice(e0);
      if (errs.length) row.flags.push(`errors: ${errs.join(' | ').slice(0, 300)}`);
    } catch (e) {
      row.error = String(e).slice(0, 300);
    }
    return row;
  }

  async function runBatch(seeds, opts = {}) {
    window.randomTest.rows = [];
    for (const seed of seeds) {
      const row = await runOne(seed, opts);
      window.randomTest.rows.push(row);
      opts.onRow?.(row);
    }
    return window.randomTest.rows;
  }

  /** The rows worth reading: errors, and flags other than floors or high Mach numbers alone. */
  const serious = (rows) => rows.filter((r) => r.error || (r.flags ?? []).some((f) => !/^(floored|mach)/.test(f)));

  window.randomTest = { presetSetup, composedSetup, runOne, runBatch, serious, rows: [] };
})();

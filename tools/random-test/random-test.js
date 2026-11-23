// Random setups for Shaderous, run in the page: injected by tools/random-test/run.mjs, or pasted
// into a browser console on the simulator. Defines window.randomTest:
//
//   randomTest.runDoc(gen, opts)    load one generated setup ({ seed, doc, closed, ... } from
//                                   tools/random-test/generate.ts), run it, return a row with flags
//   randomTest.runOne(seed, opts)   generate and run one, in the page (a dev server only: it
//                                   imports generate.ts from the source)
//   randomTest.runBatch(seeds, opts) runOne over a list; the rows also land in randomTest.rows
//
// A row is flagged when its values go non-finite, more than 1% of cells hit the floors, the
// Mach number passes 100, div B passes 1e-4, the page reports an error, or (for a document
// whose faces are all walls or periodic, with nothing that adds or takes gas) the mass changes by
// more than 1e-4. Floors and high Mach numbers alone are common in setups with vacuum or cold
// gas: read those rows before calling them bugs. A document the page's own check refuses is
// skipped and listed: the generator only makes documents that pass that check in node, so a
// refusal here is the page disagreeing with it (or a grid too big for the GPU memory).
(() => {
  /** Load and run one generated setup; a row with what was run and its flags. */
  async function runDoc(gen, opts = {}) {
    const s = window.shaderous;
    const steps = opts.steps ?? 400;
    const row = { seed: gen.seed, mode: gen.mode, preset: gen.preset, features: gen.features.join(',') };
    try {
      const e0 = s.errors().length;
      // The page's error panel stays up once an error comes, and covered every screenshot after
      // the first run that had one (src/ui/errors.ts): hidden for each run, it shows again only for
      // an error of this run's own.
      const panel = document.getElementById('error');
      if (panel) panel.style.display = 'none';
      const c = s.check(gen.doc);
      if (!c.ok) {
        row.skipped = (c.issues ?? []).filter((i) => i.level === 'error').map((i) => i.text).join(' | ').slice(0, 300) || c.error;
        return row;
      }
      await s.load({}, gen.doc);
      await window.shaderousApp?.sim?.settled;
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
      // div B is relative to the largest |B| now: where the field has left the box (a loop carried
      // out through an outflow face) or decayed, its round-off over a vanishing largest field is no
      // divergence (seed 20160: 0.58 with 1.5e-10 left of 1e-3; seed 20497: 1.3e-4 with 4e-6 left
      // of 1e-2). Judged against the field the run started with. (And NaN compares false.)
      if (d.magnetic) {
        const b0 = d0.magnetic?.bMax ?? 0;
        const scaled = d.magnetic.divB * (b0 > 0 ? Math.min(1, d.magnetic.bMax / b0) : 1);
        if (!Number.isFinite(d.magnetic.divB) || !Number.isFinite(d.magnetic.bMax)) row.flags.push('non-finite field');
        else if (scaled > 1e-4) row.flags.push(`divB ${d.magnetic.divB}`);
      }
      if (d.radiation && !Object.values(d.radiation).every(Number.isFinite)) row.flags.push('non-finite radiation');
      if (gen.closed && Math.abs(row.massChange) > 1e-4) row.flags.push(`mass ${row.massChange} in a closed box`);
      const errs = s.errors().slice(e0);
      if (errs.length) row.flags.push(`errors: ${errs.join(' | ').slice(0, 300)}`);
    } catch (e) {
      row.error = String(e).slice(0, 300);
    }
    return row;
  }

  let generator;
  /** Generate in the page: the generator is TypeScript, which only a dev server serves. */
  async function runOne(seed, opts = {}) {
    generator ??= await import('/tools/random-test/generate.ts');
    return runDoc(generator.generate(seed, opts), opts);
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

  /** The rows worth reading: errors, refusals, and flags other than floors or high Mach numbers alone. */
  const serious = (rows) => rows.filter((r) => r.error || r.skipped || (r.flags ?? []).some((f) => !/^(floored|mach)/.test(f)));

  window.randomTest = { runDoc, runOne, runBatch, serious, rows: [] };
})();

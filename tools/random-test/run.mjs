#!/usr/bin/env node
// Random-setup testing: generate setup documents at random from the option table
// (tools/random-test/generate.ts), load each into the simulator, run it, and report the ones
// whose values go bad (tools/random-test/random-test.js says what is flagged).
//
//   npm run random-test -- --runs 96 --seed 1 --steps 400 --out random-test-out
//   npm run random-test -- --coverage 3000        (no browser: which features the seeds reach)
//
// Options:
//   --runs N           how many setups (default 48); seeds are seed, seed + 1, ...: odd seeds are
//                      composed from a box of gas, even ones start from a preset
//   --seed S           first seed (default: from the clock); a row's seed reproduces its setup
//   --seeds A,B,C      these seeds instead
//   --mode M           compose or preset, for every seed
//   --dim D            every setup in D dimensions (1, 2 or 3); presets that do not take it are
//                      not drawn
//   --docs A.json,...  these setup documents instead of generated ones (a row's `doc`, edited:
//                      for bisecting a failure)
//   --steps N          steps per run (default 400)
//   --out DIR          where results.json and the screenshots go (default random-test-out)
//   --shot-all         a screenshot of every run's end state (run-<seed>.png)
//   --shot-every K     a screenshot every K runs (default 8; 0 for none); --shot-all is K = 1
//   --gpu-memory GB    the page's GPU memory budget (default 4: keeps runs under about 4 GB)
//   --headed           show the browser
//   --url URL          an already running page (a frozen `vite preview`, say) instead of a dev
//                      server started for the run
//   --coverage N       generate N documents from the seed and report the features of the option
//                      table they reach (coverage.json in --out), without running any
//
// Exit status 1 when any run errored, was refused by the page, or was flagged with more than
// floors or a high Mach number.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openSession, lowerPriority, REPO } from '../headless/session.mjs';

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i < 0 ? dflt : args[i + 1];
};
const runs = Number(opt('runs', 48));
const seed0 = Number(opt('seed', Date.now() % 1_000_000_000));
const steps = Number(opt('steps', 400));
const mode = opt('mode', undefined);
const dim = opt('dim', undefined) ? Number(opt('dim', undefined)) : undefined;
const out = path.resolve(opt('out', 'random-test-out'));
const shotEvery = args.includes('--shot-all') ? 1 : Number(opt('shot-every', 8));
const gpuMemory = opt('gpu-memory', '4');
const coverageOnly = opt('coverage', undefined);
const seeds = opt('docs', undefined)?.split(',').map((_, i) => i) ?? opt('seeds', undefined)?.split(',').map(Number) ?? Array.from({ length: coverageOnly ? Number(coverageOnly) : runs }, (_, i) => seed0 + i);

lowerPriority();
fs.mkdirSync(out, { recursive: true });

// The generator is TypeScript over the app's own modules: Vite loads it in node.
const { createServer } = await import('vite');
const vite = await createServer({ configFile: path.join(REPO, 'vite.config.ts'), server: { middlewareMode: true, hmr: false, watch: null }, appType: 'custom', logLevel: 'error' });
const gen = await vite.ssrLoadModule('/tools/random-test/generate.ts');
const docFiles = opt('docs', undefined)?.split(',');
const make = (seed, i) => {
  if (!docFiles) return gen.generate(seed, { ...(mode ? { mode } : {}), ...(dim ? { ndim: dim } : {}) });
  const doc = JSON.parse(fs.readFileSync(docFiles[i], 'utf8'));
  return { seed: path.basename(docFiles[i]), mode: 'document', doc, tried: [], features: gen.featuresOf(doc), closed: gen.closedBox(doc) };
};

if (coverageOnly) {
  const t0 = Date.now();
  const docs = seeds.map(make);
  const c = gen.coverage(docs);
  await vite.close();
  const total = gen.features().length;
  const top = Object.entries(c.refused).sort((a, b) => b[1] - a[1]).slice(0, 15);
  console.log(`${docs.length} documents in ${((Date.now() - t0) / 1000).toFixed(1)} s: ${total - c.missing.length} of ${total} features reached; ${c.pairs.seen} of ${c.pairs.of} pairs of main features seen together`);
  if (c.missing.length) console.log(`never reached:\n  ${c.missing.join('\n  ')}`);
  console.log(`most often turned away:\n  ${top.map(([id, n]) => `${id}: ${n}`).join('\n  ')}`);
  console.log(`not aimed at:\n  ${Object.entries(gen.EXCLUDED).map(([k, v]) => `${k}: ${v}`).join('\n  ')}`);
  fs.writeFileSync(path.join(out, 'coverage.json'), JSON.stringify(c, null, 1));
  console.log(`coverage: ${path.join(out, 'coverage.json')}`);
  process.exit(0);
}

const script = path.join(path.dirname(fileURLToPath(import.meta.url)), 'random-test.js');
const session = await openSession({ setup: 'problem=sod-x&dim=2&n=128', gpuMemory, headed: args.includes('--headed'), interactive: true, url: opt('url', undefined) });
const { page } = session;
await page.addScriptTag({ path: script });

const rows = [];
try {
  for (let i = 0; i < seeds.length; i++) {
    const seed = seeds[i];
    const g = make(seed, i);
    const row = await page.evaluate(([gg, n]) => window.randomTest.runDoc(gg, { steps: n }), [g, steps]);
    row.doc = g.doc;
    rows.push(row);
    const what = `${g.preset ? `${g.preset} + ` : ''}${g.tried.filter((t) => t.ok).map((t) => t.id).join(', ') || '(as is)'}`;
    const note = row.error ? `ERROR ${row.error}` : row.skipped ? `skipped (${row.skipped})` : (row.flags ?? []).join(', ') || 'ok';
    console.log(`${String(i + 1).padStart(4)}/${seeds.length} seed ${seed} ${what}: ${note}`);
    if (shotEvery > 0 && (i + 1) % shotEvery === 0 && !row.skipped && !row.error) await page.screenshot({ path: path.join(out, `run-${seed}.png`) });
    fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(rows, null, 1));
  }
} finally {
  await session.close();
  await vite.close();
}

const ran = rows.filter((r) => !r.skipped && !r.error);
const serious = rows.filter((r) => r.error || r.skipped || (r.flags ?? []).some((f) => !/^(floored|mach)/.test(f)));
console.log(`\n${rows.length} setups: ${ran.length} ran, ${rows.filter((r) => r.skipped).length} refused by the page, ${rows.filter((r) => r.error).length} errored, ${serious.length} to look at.`);
for (const r of serious) console.log(`  seed ${r.seed}: ${r.preset ?? r.mode} ${r.error ?? r.skipped ?? r.flags.join(', ')}`);
console.log(`results: ${path.join(out, 'results.json')} (each row holds its document)`);
process.exit(serious.length ? 1 : 0);

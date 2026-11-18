#!/usr/bin/env node
// Random-setup testing: load random presets and setup documents into the simulator, run each,
// and report the ones whose values go bad (tools/random-test/random-test.js says what is flagged).
//
//   npm run random-test -- --runs 96 --seed 1 --steps 400 --out random-test-out
//
// Options:
//   --runs N           how many setups (default 48); seeds are seed, seed + 1, ...: odd seeds are
//                      setup documents assembled from parts, even ones presets
//   --seed S           first seed (default: from the clock); a row's seed reproduces its setup
//   --steps N          steps per run (default 400)
//   --out DIR          where results.json and the screenshots go (default random-test-out)
//   --shot-every K     a screenshot every K runs (default 8; 0 for none)
//   --gpu-memory GB    the page's GPU memory budget (default 4: keeps runs under about 4 GB)
//   --headed           show the browser
//   --url URL          an already running page (a frozen `vite preview`, say) instead of a dev
//                      server started for the run
//
// Exit status 1 when any run errored or was flagged with more than floors or a high Mach number.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openSession, lowerPriority } from '../headless/session.mjs';

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i < 0 ? dflt : args[i + 1];
};
const runs = Number(opt('runs', 48));
const seed0 = Number(opt('seed', Date.now() % 1_000_000_000));
const steps = Number(opt('steps', 400));
const out = path.resolve(opt('out', 'random-test-out'));
const shotEvery = Number(opt('shot-every', 8));
const gpuMemory = opt('gpu-memory', '4');

lowerPriority();
fs.mkdirSync(out, { recursive: true });
const script = path.join(path.dirname(fileURLToPath(import.meta.url)), 'random-test.js');
const session = await openSession({ setup: 'problem=sod-x&dim=2&n=128', gpuMemory, headed: args.includes('--headed'), interactive: true, url: opt('url', undefined) });
const { page } = session;
await page.addScriptTag({ path: script });

const rows = [];
try {
  for (let i = 0; i < runs; i++) {
    const seed = seed0 + i;
    const row = await page.evaluate(([s, n]) => window.randomTest.runOne(s, { steps: n }), [seed, steps]);
    rows.push(row);
    const what = row.query ?? `${row.geometry} ${row.kinds}`;
    const note = row.error ? `ERROR ${row.error}` : row.skipped ? `skipped (${row.skipped})` : (row.flags ?? []).join(', ') || 'ok';
    console.log(`${String(i + 1).padStart(4)}/${runs} seed ${seed} ${what}: ${note}`);
    if (shotEvery > 0 && (i + 1) % shotEvery === 0) await page.screenshot({ path: path.join(out, `run-${seed}.png`) });
    fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(rows, null, 1));
  }
} finally {
  await session.close();
}

const ran = rows.filter((r) => !r.skipped && !r.error);
const serious = rows.filter((r) => r.error || (r.flags ?? []).some((f) => !/^(floored|mach)/.test(f)));
console.log(`\n${rows.length} setups: ${ran.length} ran, ${rows.filter((r) => r.skipped).length} refused by the page's check, ${rows.filter((r) => r.error).length} errored, ${serious.length} to look at.`);
for (const r of serious) console.log(`  seed ${r.seed}: ${r.query ?? r.kinds} ${r.error ?? r.flags.join(', ')}`);
console.log(`results: ${path.join(out, 'results.json')}`);
process.exit(serious.length ? 1 : 0);

#!/usr/bin/env node
// Data for the paper's figures that the scripting API cannot reach (the validation suite's own
// setups, and readbacks of the simulator's internals): runs src/validation/figure_data.ts inside
// the headless page that tools/headless/session.mjs opens, and writes what it returns as JSON.
//
//   node paper/figures/collect.mjs NAME OUT.json [ARGS_JSON] [--normal]
//
// NAME is a function exported by src/validation/figure_data.ts; ARGS_JSON a JSON list of its
// arguments. Run from the code's repository, which Vite serves to the page. The run is at
// below-normal CPU priority, except with --normal (for benchmarks).
import { writeFileSync } from 'node:fs';
import { lowerPriority, openSession } from '../../../tools/headless/session.mjs';

const argv = process.argv.slice(2);
// --normal: stay at normal priority, for benchmarks; everything else runs below normal.
const normal = argv.includes('--normal');
const [name, out, args] = argv.filter((a) => a !== '--normal');
if (!name || !out) {
  process.stderr.write('usage: node paper/figures/collect.mjs NAME OUT.json [ARGS_JSON] [--normal]\n');
  process.exit(1);
}
if (!normal) lowerPriority();
// A small, idle run: the page's own simulator stays out of the way of the one the data needs.
const s = await openSession({ setup: 'problem=sedov&dim=2&n=128', log: () => {} });
try {
  const t0 = Date.now();
  const data = await s.page.evaluate(async ([n, a]) => {
    const m = await import('/src/validation/figure_data.ts');
    return m[n](...a);
  }, [name, JSON.parse(args ?? '[]')]);
  writeFileSync(out, JSON.stringify(data) + '\n');
  process.stderr.write(`${name}: ${((Date.now() - t0) / 1000).toFixed(0)} s -> ${out}\n`);
} finally {
  await s.close();
}

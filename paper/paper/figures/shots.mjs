#!/usr/bin/env node
// Images for the paper's figures, drawn by the code's own renderer: each shot runs a setup to a
// time in headless Chrome (tools/headless/session.mjs, docs/headless.md) and saves the view as a
// PNG, or a screenshot of the whole page as a person sees it.
//
//   node paper/figures/shots.mjs SPEC.json OUTDIR [NAME...]
//
// SPEC.json is a list of shots: { name, setup, until, steps?, view?, width?, height?, page?, images? }
// (with NAMEs after OUTDIR, only those).
// `setup` is a link's query string; `view` is passed to shaderous.view() (docs/api.md). `images`,
// a list of { name, view? }, takes several images of the one run, each with its own view on top
// of the shot's (the shot's own name is then only for the log). With `page`, the page
// opens as a person sees it, at width x height, and the whole window is captured once the run
// has reached `until` and drawn a few frames. Run from the code's repository, at low priority.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { lowerPriority, openSession, pngBytes } from '../../../tools/headless/session.mjs';

const [specFile, outDir, ...only] = process.argv.slice(2);
if (!specFile || !outDir) {
  process.stderr.write('usage: node paper/figures/shots.mjs SPEC.json OUTDIR [NAME...]\n');
  process.exit(1);
}
lowerPriority();
mkdirSync(outDir, { recursive: true });
// NAMEs: take only those shots again.
const shots = JSON.parse(readFileSync(specFile, 'utf8')).filter((s) => !only.length || only.includes(s.name));
const log = (m) => process.stderr.write(m + '\n');

for (const shot of shots) {
  const t0 = Date.now();
  const width = shot.width ?? 1000;
  const height = shot.height ?? width;
  const s = await openSession({ setup: shot.setup, interactive: !!shot.page, width, height, log: () => {} });
  try {
    if (shot.page) await s.call('pause');
    const st = await s.call('state');
    let r = await s.call('run', { until: shot.until });
    // `steps`: then that many ordinary steps. A run that lands on a time shortens its last step,
    // and with per-block timesteps that last cycle's bins are not the run's usual ones.
    if (shot.steps) r = await s.call('run', { steps: shot.steps });
    // The whole view in force (a preset sets its own camera angles), for the figure's axes triad.
    let full = await s.call('view', shot.view ?? {});
    // One image per entry of `images`, each with its own view on top of the shot's.
    let errors = [];
    for (const img of shot.images ?? [{ name: shot.name }]) {
      if (img.view) full = await s.call('view', img.view);
      const file = path.join(outDir, `${img.name}.png`);
      if (shot.page) {
        // Let the page's own loop draw the paused run and refresh its readouts.
        await s.page.waitForTimeout(1500);
        await s.page.screenshot({ path: file });
      } else {
        writeFileSync(file, pngBytes(await s.call('screenshot', { width, height })));
      }
      errors = await s.call('errors');
      const meta = { name: img.name, setup: st.query, t: r.t, steps: r.steps, view: { ...shot.view, ...img.view }, camera: { yaw: full.yaw, pitch: full.pitch, dist: full.dist }, adapter: st.adapter, errors };
      writeFileSync(path.join(outDir, `${img.name}.json`), JSON.stringify(meta, null, 1) + '\n');
    }
    log(`${shot.name}: ${st.query} to t = ${r.t} (${r.steps} steps, ${((Date.now() - t0) / 1000).toFixed(0)} s)${errors.length ? `, ${errors.length} page error(s)` : ''}`);
    if (errors.length) process.exitCode = 2;
  } finally {
    await s.close();
  }
}

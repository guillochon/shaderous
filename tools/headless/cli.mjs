#!/usr/bin/env node
// The simulator from the command line: runs in headless Chrome, writing diagnostics, images,
// slices and .athdf files to a folder. See docs/headless.md.
//
//   node tools/headless/cli.mjs run "problem=sedov&dim=2&n=512" --every 0.01 --png --out runs/sedov
//   node tools/headless/cli.mjs sweep --problem sedov --dim 2 --vary gamma=1.1,1.4,1.667 --until 0.05
//   node tools/headless/cli.mjs describe [problem]
import { mkdirSync, writeFileSync, appendFileSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { debrisSummary, lowerPriority, openSession, pngBytes, setupQuery } from './session.mjs';

const HELP = `Usage: node tools/headless/cli.mjs <command> [setup] [options]

Commands
  run       Run one setup, writing outputs to a folder
  sweep     Run a setup once per combination of --vary values
  describe  Print the problems, fields and choices as JSON (or one problem's details)

Setup (the same keys as a shared link; see docs/api.md)
  [setup]              A query string or link: "problem=sedov&dim=2&n=512"
  --problem ID         --dim 2|3     --n CELLS     --amr LEVELS
  --set KEY=VALUE      Any other setup key, repeatable: --set gamma=1.4 --set p.mach=3

Stopping (default: the problem's end time)
  --until T            Run to this time (landed on exactly)
  --steps N            Run at most this many steps
  --every DT           Write outputs every DT of simulated time (and at the start)
  --timeout SECONDS    Give up on a run after this much wall time

Outputs (diag.jsonl and manifest.json are always written)
  --out DIR            Folder (default: runs/<problem>-<date>)
  --png                A screenshot at each output
  --width PX --height PX   Screenshot size (default 1024 square)
  --view KEY=VALUE     View setting for screenshots, repeatable: --view field=3 --view colormap=viridis
  --slice VAR          A slice of this variable at each output, as JSON (repeatable)
  --slice-axis A --slice-pos P --slice-res R
  --athdf              An .athdf file at each output
  --debris             Debris energies and dM/dt at each output, as JSON (problems that track
                       them, such as tidal-disruption); plot with tools/headless/plot_dmdt.py

Sweeps
  --vary KEY=V1,V2,..  Values of a setup key, repeatable; every combination is run

Other
  --url URL            Use the page served here (default: serve this checkout with Vite)
  --headed             Show the browser (if headless Chrome finds no GPU adapter)
  --low-priority       Run at below-normal CPU priority (the browser inherits it on Windows)
  --quiet              No progress messages
`;

const { values: o, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    problem: { type: 'string' }, dim: { type: 'string' }, n: { type: 'string' }, amr: { type: 'string' },
    set: { type: 'string', multiple: true },
    until: { type: 'string' }, steps: { type: 'string' }, every: { type: 'string' }, timeout: { type: 'string' },
    out: { type: 'string' }, png: { type: 'boolean' }, width: { type: 'string' }, height: { type: 'string' },
    view: { type: 'string', multiple: true },
    slice: { type: 'string', multiple: true }, 'slice-axis': { type: 'string' }, 'slice-pos': { type: 'string' }, 'slice-res': { type: 'string' },
    athdf: { type: 'boolean' }, debris: { type: 'boolean' },
    vary: { type: 'string', multiple: true },
    url: { type: 'string' }, headed: { type: 'boolean' }, 'low-priority': { type: 'boolean' }, quiet: { type: 'boolean' },
    help: { type: 'boolean', short: 'h' },
  },
});

const [command, setupArg] = positionals;
const log = (m) => o.quiet || process.stderr.write(m + '\n');
/** A mistake in the command line, before anything has started. */
const usage = (m) => {
  process.stderr.write(`error: ${m}\n`);
  process.exit(1);
};
/** A mistake found once the browser is up: thrown, so the browser is closed on the way out. */
const fail = (m) => {
  throw new Error(m);
};
const num = (v, name) => {
  if (v === undefined) return undefined;
  const x = Number(v);
  if (!Number.isFinite(x)) fail(`--${name} needs a number, not "${v}"`);
  return x;
};
const pair = (s, flag) => {
  const i = s.indexOf('=');
  if (i < 1) fail(`--${flag} takes KEY=VALUE, not "${s}"`);
  return [s.slice(0, i), s.slice(i + 1)];
};

if (o.help || !command) {
  process.stdout.write(HELP);
  process.exit(command || o.help ? 0 : 1);
}
if (!['run', 'sweep', 'describe'].includes(command)) usage(`unknown command "${command}"\n\n${HELP}`);
if (o['low-priority']) lowerPriority();

/** The setup from the positional argument and the flags. */
function baseSetup() {
  const q = new URLSearchParams(setupQuery(command === 'describe' ? '' : setupArg));
  for (const k of ['problem', 'dim', 'n', 'amr']) if (o[k] !== undefined) q.set(k, o[k]);
  for (const s of o.set ?? []) q.set(...pair(s, 'set'));
  return q;
}

/** View settings: numbers and booleans parsed, a field given by name or by index. */
function viewSettings(fields) {
  const v = {};
  for (const s of o.view ?? []) {
    const [k, raw] = pair(s, 'view');
    if (k === 'field' && !/^\d+$/.test(raw)) {
      const f = fields.find((x) => x.name.toLowerCase() === raw.toLowerCase());
      if (!f) fail(`no field "${raw}": use one of ${fields.map((x) => x.name).join(', ')}`);
      v.field = f.index;
    } else v[k] = raw === 'true' ? true : raw === 'false' ? false : Number.isFinite(Number(raw)) && raw !== '' ? Number(raw) : raw;
  }
  return v;
}

/** Every combination of the --vary values, as lists of [key, value]. */
function combinations() {
  let out = [[]];
  for (const s of o.vary ?? []) {
    const [k, list] = pair(s, 'vary');
    const vals = list.split(',').filter(Boolean);
    if (!vals.length) fail(`--vary ${k} has no values`);
    out = out.flatMap((c) => vals.map((v) => [...c, [k, v]]));
  }
  return out;
}

const stamp = () => new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
const tag = (t) => String(+t.toPrecision(6));

/** One run of a setup, outputs into `dir`. Returns its summary. */
async function runOne(session, q, dir) {
  const { call } = session;
  const st = await call('load', q.toString());
  mkdirSync(dir, { recursive: true });
  const until = num(o.until, 'until') ?? (o.steps === undefined ? st.tEnd : undefined);
  const steps = num(o.steps, 'steps');
  const every = num(o.every, 'every');
  if (until == null && steps === undefined) fail(`${st.problem} has no end time: give --until or --steps`);
  if (every !== undefined && !(every > 0)) fail('--every must be positive');
  if (every !== undefined && until == null) fail('--every needs --until (or a problem with an end time)');
  const timeoutMs = o.timeout === undefined ? undefined : num(o.timeout, 'timeout') * 1000;

  const d0 = await call('describe');
  const view = viewSettings(d0.fields);
  if (Object.keys(view).length) await call('view', view);
  const manifest = {
    program: 'shaderous',
    build: await session.page.evaluate(() => window.shaderous.build),
    adapter: st.adapter,
    setup: st.query,
    view,
    problem: st.problem,
    started: new Date().toISOString(),
    command: process.argv.slice(2),
    outputs: [],
  };
  const writeManifest = () => writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  writeManifest();

  const t0 = Date.now();
  let k = 0;
  /** Diagnostics and the requested files, at the run's current time. */
  const output = async () => {
    const d = await call('diagnostics');
    const name = `${String(k).padStart(4, '0')}-t${tag(d.t)}`;
    const files = [];
    appendFileSync(path.join(dir, 'diag.jsonl'), JSON.stringify({ output: k, wall: (Date.now() - t0) / 1000, ...d }) + '\n');
    if (o.png) {
      const png = await call('screenshot', { width: num(o.width, 'width') ?? 1024, height: num(o.height, 'height') ?? num(o.width, 'width') ?? 1024 });
      writeFileSync(path.join(dir, `${name}.png`), pngBytes(png));
      files.push(`${name}.png`);
    }
    for (const variable of o.slice ?? []) {
      const sl = await call('slice', { variable, axis: num(o['slice-axis'], 'slice-axis'), pos: num(o['slice-pos'], 'slice-pos'), res: num(o['slice-res'], 'slice-res') });
      writeFileSync(path.join(dir, `${name}-${variable}.json`), JSON.stringify(sl) + '\n');
      files.push(`${name}-${variable}.json`);
    }
    if (o.debris) {
      const r = await call('debris');
      writeFileSync(path.join(dir, `${name}-debris.json`), JSON.stringify(r) + '\n');
      files.push(`${name}-debris.json`);
      const s = debrisSummary(r);
      const p = s.physical;
      log(`    debris: escaped ${s.totals.escaped.toPrecision(4)}  in box ${s.totals.inBox.toPrecision(4)}  core ${s.totals.core.toPrecision(5)}  bound ${s.totals.bound.toPrecision(4)}` +
        (p ? `  dM/dt peak ${p.peakMsunPerYear.toPrecision(3)} Msun/yr at ${p.peakDaysAfterPericenter.toPrecision(3)} d after pericenter` : ''));
    }
    if (o.athdf) {
      const f = await call('athdf');
      writeFileSync(path.join(dir, `${name}.athdf`), Buffer.from(f.base64, 'base64'));
      files.push(`${name}.athdf`);
    }
    manifest.outputs.push({ output: k, t: d.t, steps: d.steps, files });
    writeManifest();
    log(`  t = ${tag(d.t)}  steps ${d.steps}  mass ${d.mass.toPrecision(7)}  eTot ${d.eTot.toPrecision(7)}  (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
    k++;
  };

  let stopped = 'steps';
  let r = { t: 0, steps: 0 };
  await output();
  if (every !== undefined) {
    // Output times are multiples of `every`, so runs line up whatever their timesteps.
    for (let j = 1; ; j++) {
      const target = Math.min(j * every, until);
      r = await call('run', { until: target, ...(steps !== undefined ? { steps: steps - r.steps } : {}), timeoutMs });
      stopped = r.stopped;
      await output();
      if (r.stopped !== 'time' || target >= until) break;
    }
  } else {
    r = await call('run', { ...(until != null ? { until } : {}), ...(steps !== undefined ? { steps } : {}), timeoutMs });
    stopped = r.stopped;
    await output();
  }
  manifest.finished = new Date().toISOString();
  manifest.stopped = stopped;
  manifest.seconds = (Date.now() - t0) / 1000;
  const errors = await call('errors');
  if (errors.length) manifest.errors = errors;
  writeManifest();
  if (errors.length) log(`  the page reported ${errors.length} error(s): see manifest.json`);
  return { dir, setup: st.query, t: r.t, steps: r.steps, stopped, seconds: manifest.seconds, errors: errors.length };
}

const session = await openSession({ url: o.url, headed: o.headed, log });
let code = 0;
try {
  if (command === 'describe') {
    const d = await session.call('describe');
    const id = setupArg;
    if (id) {
      const p = d.problems.find((x) => x.id === id);
      if (!p) fail(`no problem "${id}"`);
      process.stdout.write(JSON.stringify(p, null, 2) + '\n');
    } else process.stdout.write(JSON.stringify(d, null, 2) + '\n');
  } else {
    const base = baseSetup();
    const problem = base.get('problem') ?? (await session.call('state')).problem;
    const out = o.out ?? path.join('runs', `${problem}-${stamp()}`);
    const combos = command === 'sweep' ? combinations() : [[]];
    if (command === 'sweep' && !o.vary?.length) fail('sweep needs at least one --vary KEY=V1,V2');
    for (const combo of combos) {
      const q = new URLSearchParams(base);
      for (const [key, v] of combo) q.set(key, v);
      const dir = combo.length ? path.join(out, combo.map(([key, v]) => `${key}=${v}`).join(',')) : out;
      log(`${q.toString()}  ->  ${dir}`);
      const s = await runOne(session, q, dir);
      if (s.errors) code = 2;
      if (command === 'sweep') appendFileSync(path.join(out, 'sweep.jsonl'), JSON.stringify({ ...Object.fromEntries(combo), ...s }) + '\n');
    }
    log(`done: ${out}`);
  }
} catch (e) {
  process.stderr.write(`error: ${e.message}\n`);
  code = 1;
} finally {
  await session.close();
}
process.exit(code);

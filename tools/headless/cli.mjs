#!/usr/bin/env node
// The simulator from the command line: runs in headless Chrome, writing diagnostics, images,
// slices and .athdf files to a folder. See docs/headless.md.
//
//   node tools/headless/cli.mjs run "problem=sedov&dim=2&n=512" --every 0.01 --png --out runs/sedov
//   node tools/headless/cli.mjs sweep --problem sedov --dim 2 --vary gamma=1.1,1.4,1.667 --until 0.05
//   node tools/headless/cli.mjs run "problem=kelvin-helmholtz&dim=2&n=256" --until 2 --mp4
//   node tools/headless/cli.mjs describe [problem]
import { mkdirSync, readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { helpText, OPTIONS } from './flags.mjs';
import { debrisSummary, fetchSlice, lowerPriority, openSession, pngBytes, setupQuery } from './session.mjs';
import { npyBytes } from './npy.mjs';

const HELP = `Usage: shaderous <command> [setup] [options]   (in a checkout: npm run headless -- <command> ...)

Commands
  run       Run one setup, writing outputs to a folder
  sweep     Run a setup once per combination of --vary values
  describe  Print the problems, fields and choices as JSON (or one problem's details)
${helpText()}
`;

const { values: o, positionals } = parseArgs({ allowPositionals: true, options: OPTIONS });

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
const sliceFormat = o['slice-format'] ?? 'json';
if (!['json', 'npy'].includes(sliceFormat)) usage(`--slice-format is json or npy, not "${sliceFormat}"`);

/** The setup document of --setup, read once: the page checks it. */
let setupDocument;
if (o.setup) {
  try {
    setupDocument = JSON.parse(readFileSync(o.setup, 'utf8'));
  } catch (e) {
    usage(`--setup ${o.setup}: ${e.message}`);
  }
}

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
  const st = await call('load', q.toString(), ...(setupDocument ? [setupDocument] : []));
  mkdirSync(dir, { recursive: true });
  const until = num(o.until, 'until') ?? (o.steps === undefined ? st.tEnd : undefined);
  const steps = num(o.steps, 'steps');
  const every = num(o.every, 'every');
  if (until == null && steps === undefined) fail(`${st.problem} has no end time: give --until or --steps`);
  if (every !== undefined && !(every > 0)) fail('--every must be positive');
  if (every !== undefined && until == null) fail('--every needs --until (or a problem with an end time)');
  const timeoutMs = o.timeout === undefined ? undefined : num(o.timeout, 'timeout') * 1000;
  if (o.mp4 && (every !== undefined || steps !== undefined)) fail('--mp4 records the whole run: give --until, not --every or --steps');
  if (!o.mp4 && (o['frame-every'] || o['frame-dt'] || o.fps || o.bitrate || o['no-overlays'])) fail('--frame-every, --frame-dt, --fps, --bitrate and --no-overlays go with --mp4');
  if (o['frame-every'] !== undefined && o['frame-dt'] !== undefined) fail('give --frame-every or --frame-dt, not both');
  if (o.native && !o.png) fail('--native goes with --png');
  if (o.native && (o.width !== undefined || o.height !== undefined) && !o.mp4) fail('--native sets the screenshot size: give it without --width or --height');

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
  // A run's diagnostics start empty, as its manifest does: a second run into the same folder
  // used to append to the first one's, and a plot of the file drew both.
  writeFileSync(path.join(dir, 'diag.jsonl'), '');

  const t0 = Date.now();
  let k = 0;
  /** Diagnostics and the requested files, at the run's current time. */
  const output = async () => {
    const d = await call('diagnostics');
    const name = `${String(k).padStart(4, '0')}-t${tag(d.t)}`;
    const files = [];
    appendFileSync(path.join(dir, 'diag.jsonl'), JSON.stringify({ output: k, wall: (Date.now() - t0) / 1000, ...d }) + '\n');
    if (o.png) {
      const png = await call(
        'screenshot',
        o.native
          ? { native: true, overlays: { colorbar: true, triad: true, status: true } }
          : { width: num(o.width, 'width') ?? 1024, height: num(o.height, 'height') ?? num(o.width, 'width') ?? 1024 },
      );
      writeFileSync(path.join(dir, `${name}.png`), pngBytes(png));
      files.push(`${name}.png`);
    }
    // Fetched as float32 bytes either way (the values are those of the JSON result, bit for
    // bit), which is many times faster than a list of numbers through page.evaluate.
    const slices = [];
    for (const variable of o.slice ?? []) {
      const { head, values } = await fetchSlice(call, { variable, axis: num(o['slice-axis'], 'slice-axis'), pos: num(o['slice-pos'], 'slice-pos'), res: num(o['slice-res'], 'slice-res') });
      const file = `${name}-${variable}.${sliceFormat}`;
      if (sliceFormat === 'npy') {
        writeFileSync(path.join(dir, file), npyBytes(values, [head.res, head.res]));
        slices.push({ file, ...head });
      } else writeFileSync(path.join(dir, file), JSON.stringify({ ...head, values: Array.from(values) }) + '\n');
      files.push(file);
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
    manifest.outputs.push({ output: k, t: d.t, steps: d.steps, files, ...(slices.length ? { slices } : {}) });
    writeManifest();
    log(`  t = ${tag(d.t)}  steps ${d.steps}  mass ${d.mass.toPrecision(7)}  eTot ${d.eTot.toPrecision(7)}  (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
    k++;
  };

  let stopped = 'steps';
  let r = { t: 0, steps: 0 };
  await output();
  if (o.mp4) {
    const frameDt = num(o['frame-dt'], 'frame-dt');
    const noOverlays = !!o['no-overlays'];
    const m = await call('record', {
      ...(o['frame-every'] !== undefined ? { every: num(o['frame-every'], 'frame-every') } : frameDt !== undefined ? { dt: frameDt } : {}),
      until,
      timeoutMs,
      fps: num(o.fps, 'fps'),
      bitrate: o.bitrate === undefined ? undefined : num(o.bitrate, 'bitrate') * 1e6,
      width: num(o.width, 'width') ?? 1920,
      height: num(o.height, 'height') ?? (o.width !== undefined ? undefined : 1080),
      overlays: { colorbar: !noOverlays, triad: !noOverlays, status: !noOverlays },
      download: false,
      base64: true,
    });
    writeFileSync(path.join(dir, 'movie.mp4'), Buffer.from(m.base64, 'base64'));
    const { base64, ...movie } = m;
    manifest.movie = { file: 'movie.mp4', ...movie };
    log(`  movie.mp4: ${m.frames} frames, ${m.seconds.toFixed(1)} s at ${m.width} x ${m.height}, ${(m.bytes / 1e6).toFixed(1)} MB`);
    const st1 = await call('state');
    r = { t: st1.t, steps: st1.steps };
    stopped = m.stopped;
    await output();
  } else if (every !== undefined) {
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

const session = await openSession({ url: o.url, headed: o.headed, gpuMemory: o['gpu-memory'], log });
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

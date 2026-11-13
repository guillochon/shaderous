#!/usr/bin/env node
// The simulator as an MCP server (Model Context Protocol, over stdio), so an AI agent can run
// and look at simulations as tools: load a setup, step it, read the gas back, see screenshots.
// One simulator page, opened in headless Chrome on the first tool call and kept for the session.
// See docs/headless.md.
//
//   node tools/headless/mcp.mjs [--url URL] [--headed] [--low-priority]
//
// Messages are JSON-RPC 2.0, one per line on stdin and stdout; logging goes to stderr.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { debrisSummary, lowerPriority, openSession } from './session.mjs';

const { values: o } = parseArgs({
  options: { url: { type: 'string' }, headed: { type: 'boolean' }, 'low-priority': { type: 'boolean' } },
});
if (o['low-priority']) lowerPriority();
const log = (m) => process.stderr.write(`[shaderous-mcp] ${m}\n`);

const INSTRUCTIONS = `Shaderous is a GPU gas dynamics code (compressible Euler equations with gravity, cooling, adaptive mesh refinement). These tools drive one simulation in a headless browser.
Typical use: describe (pick a problem) -> load (a setup: problem, dim, n, amr and other keys) -> run (to a time, landed on exactly, or a number of steps) -> diagnostics / stats / slice / probe for numbers, screenshot to look at it -> save_athdf to keep the state.
All values are in code units; positions are fractions of the box (0 to 1). A run stays paused between calls. load restarts from t = 0; control changes a setting mid-run as a person would.`;

// --- the simulator page, opened on first use -----------------------------------------------
let session = null;
let opening = null;
async function sim() {
  if (session) return session;
  opening ??= openSession({ url: o.url, headed: o.headed, log }).then((s) => (session = s)).finally(() => (opening = null));
  return opening;
}
const call = async (name, ...args) => (await sim()).call(name, ...args);

// --- tools ---------------------------------------------------------------------------------
const obj = (properties, required = []) => ({ type: 'object', properties, required, additionalProperties: false });
const text = (v) => ({ content: [{ type: 'text', text: typeof v === 'string' ? v : JSON.stringify(v, null, 1) }] });
const round = (v) => (typeof v === 'number' && Number.isFinite(v) && !Number.isInteger(v) ? +v.toPrecision(7) : v);
const resolvePath = (p) => path.resolve(process.cwd(), p);

const TOOLS = [
  {
    name: 'describe',
    description: 'List the problems (presets) with their ids, dimensions, end times and parameters, and the fields, variables and choices a setup can make. Give `problem` for one problem in full, with its description.',
    inputSchema: obj({ problem: { type: 'string', description: 'A problem id for its full entry' } }),
    async run({ problem }) {
      const d = await call('describe');
      if (problem) {
        const p = d.problems.find((x) => x.id === problem);
        if (!p) throw new Error(`no problem "${problem}": call describe without arguments for the list`);
        return text(p);
      }
      return text({
        problems: d.problems.map((p) => ({ id: p.id, label: p.label, variant: p.variant, supports2d: p.supports2d, tEnd: p.tEnd, params: p.params.map((q) => `${q.id}=${q.value} [${q.min}, ${q.max}]`) })),
        fields: d.fields.map((f) => `${f.index}: ${f.name}`),
        variables: d.variables,
        resolutions: d.resolutions,
        eos: d.eos, cooling: d.cooling, riemann: d.riemann, limiter: d.limiter, gravity: d.gravity, colormaps: d.colormaps,
        setupKeys: 'problem, dim (2|3), n, amr (0-5), tend, eos, gamma, K, riemann, limiter, cfl, gravity, G, cool, nu, kappa, bc (6 digits), p.<param id>, and view keys mode, field, axis, slice, log, cmap. Unknown keys are ignored, values out of range clamped.',
      });
    },
  },
  {
    name: 'load',
    description: 'Build a new run from a setup and leave it paused at t = 0. The setup uses the keys of a shareable link, as a query string ("problem=sedov&dim=2&n=512&gamma=1.4") or an object ({"problem": "sedov", "dim": 2, "n": 512}). Returns the run\'s state, including `query`, the setup in force.',
    inputSchema: obj({ setup: { type: ['string', 'object'], description: 'Query string, link, or object of setup keys' } }, ['setup']),
    run: async ({ setup }) => text(await call('load', setup)),
  },
  {
    name: 'run',
    description: 'Advance the run: to time `until` (the last step is shortened to land on it exactly), by `steps` steps, or whichever comes first. Stops at the problem\'s end time (stopped: "end"); to go further, load with a later `tend`. Returns t, dt, steps, steps taken, wall seconds and why it stopped.',
    inputSchema: obj({
      until: { type: 'number', description: 'Run until the clock reads this time' },
      steps: { type: 'integer', minimum: 0, description: 'Take at most this many steps' },
      timeout_seconds: { type: 'number', description: 'Give up after this much wall time (default 600)' },
    }),
    run: async ({ until, steps, timeout_seconds }) => text(await call('run', { until, steps, timeoutMs: 1000 * (timeout_seconds ?? 600) })),
  },
  {
    name: 'state',
    description: 'The run: problem, dimensions, grid, amr levels, time, timestep, steps, end time, whether it has finished, the GPU adapter, and `query`, the setup that reproduces it.',
    inputSchema: obj({}),
    run: async () => text(await call('state')),
  },
  {
    name: 'diagnostics',
    description: 'Totals and extremes read back now: mass, momentum, kinetic/internal/gravitational/total energy, max Mach number, floored cells, Jeans-unresolved cells, center of mass, point masses, and the [min, max] of every field.',
    inputSchema: obj({}),
    async run() {
      const d = await call('diagnostics');
      const r = (v) => (Array.isArray(v) ? v.map(r) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, r(x)])) : round(v));
      return text(r(d));
    },
  },
  {
    name: 'stats',
    description: 'Least, greatest and volume-weighted mean of each variable over the box (rho, press, vel1-3, r0 and species mass fractions).',
    inputSchema: obj({}),
    run: async () => text(await call('stats')),
  },
  {
    name: 'slice',
    description: 'A variable on a res x res grid on a plane through the box (perpendicular to `axis` at fraction `pos`). Returns min, max and mean, and the values (row j, column i: values[j][i]) when res is at most 64 or include_values is set; with save_path the full slice is written there as JSON instead. Variables: rho, press, vel1, vel2, vel3, r0, level, or a species name.',
    inputSchema: obj({
      variable: { type: 'string', description: 'Default rho' },
      axis: { type: 'integer', minimum: 0, maximum: 2, description: 'Default 2 (z); always 2 in 2D' },
      pos: { type: 'number', minimum: 0, maximum: 1, description: 'Default 0.5' },
      res: { type: 'integer', minimum: 2, maximum: 2048, description: 'Default 32' },
      include_values: { type: 'boolean' },
      save_path: { type: 'string', description: 'Write the slice as JSON to this file' },
    }),
    async run({ variable, axis, pos, res = 32, include_values, save_path }) {
      const s = await call('slice', { variable, axis, pos, res });
      const mean = s.values.reduce((a, b) => a + b, 0) / s.values.length;
      const out = { t: s.t, variable: s.variable, axis: s.axis, pos: s.pos, res: s.res, axes: s.axes, min: s.min, max: s.max, mean };
      if (save_path) {
        const file = resolvePath(save_path);
        mkdirSync(path.dirname(file), { recursive: true });
        writeFileSync(file, JSON.stringify(s) + '\n');
        return text({ ...out, saved: file });
      }
      if (include_values ?? res <= 64) {
        out.values = Array.from({ length: s.res }, (_, j) => s.values.slice(j * s.res, (j + 1) * s.res).map((v) => +v.toPrecision(5)));
      }
      return text(out);
    },
  },
  {
    name: 'probe',
    description: 'The gas at points given as fractions of the box ([x, y, z], z ignored in 2D): density, pressure, velocity, passive scalar, refinement level and species mass fractions.',
    inputSchema: obj({ points: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 }, minItems: 1 } }, ['points']),
    run: async ({ points }) => text(await call('probe', points)),
  },
  {
    name: 'screenshot',
    description: 'A picture of the run as a PNG image. `view` changes what is drawn first: mode (volume|slice|projection), field (index, see describe), log, lo, hi, sliceAxis, slicePos, yaw, pitch, dist, zoom, cx, cy, showBlocks, contours, nContours, fog, surfaceOpacity, colormap, autoRange. Settings persist for later screenshots.',
    inputSchema: obj({
      width: { type: 'integer', minimum: 16, maximum: 4096, description: 'Default 768' },
      height: { type: 'integer', minimum: 16, maximum: 4096, description: 'Default: width' },
      view: { type: 'object', description: 'View settings to apply first' },
      save_path: { type: 'string', description: 'Also write the PNG to this file' },
    }),
    async run({ width = 768, height, view, save_path }) {
      if (view && Object.keys(view).length) await call('view', view);
      const png = await call('screenshot', { width, height: height ?? width });
      const data = png.slice(png.indexOf(',') + 1);
      const content = [{ type: 'image', data, mimeType: 'image/png' }];
      if (save_path) {
        const file = resolvePath(save_path);
        mkdirSync(path.dirname(file), { recursive: true });
        writeFileSync(file, Buffer.from(data, 'base64'));
        content.push({ type: 'text', text: `saved ${file}` });
      }
      return { content };
    },
  },
  {
    name: 'view',
    description: 'Change what screenshots show without touching the run (same keys as screenshot\'s `view`). Returns the view settings.',
    inputSchema: obj({ changes: { type: 'object' } }, ['changes']),
    run: async ({ changes }) => text(await call('view', changes)),
  },
  {
    name: 'controls',
    description: 'The control panel as it stands: every control\'s id, kind, label, tooltip, value and options. Use with `control` to change settings mid-run.',
    inputSchema: obj({}),
    run: async () => text(await call('controls')),
  },
  {
    name: 'control',
    description: 'Set a panel control (or press a button, with no value) exactly as a person would, mid-run, and wait for what it starts (a shader rebuild, a restart). Examples: control("riemann", "hll"), control("eos", "gas_radiation"), control("reset"). Number fields take values in the units the panel shows; set control("units", "code") first for code units. The run stays paused.',
    inputSchema: obj({ id: { type: 'string' }, value: { type: ['string', 'number', 'boolean'] } }, ['id']),
    async run({ id, value }) {
      await call('control', id, value);
      return text(await call('state'));
    },
  },
  {
    name: 'restart',
    description: 'Start the run again from its initial conditions, paused.',
    inputSchema: obj({}),
    run: async () => text(await call('restart')),
  },
  {
    name: 'save_athdf',
    description: 'Save the state of the gas to an .athdf file (HDF5 in Athena++\'s layout; yt reads it). The file carries the setup, so load_athdf continues the run exactly.',
    inputSchema: obj({ path: { type: 'string', description: 'File or folder to write to (default: the current folder, under the file\'s own name)' } }),
    async run({ path: p }) {
      const f = await call('athdf');
      let file = resolvePath(p ?? f.name);
      if (!p || p.endsWith('/') || p.endsWith('\\')) file = path.join(resolvePath(p ?? '.'), f.name);
      mkdirSync(path.dirname(file), { recursive: true });
      writeFileSync(file, Buffer.from(f.base64, 'base64'));
      return text({ saved: file, time: f.time, bytes: f.bytes });
    },
  },
  {
    name: 'load_athdf',
    description: 'Start a run from an .athdf file: one Shaderous wrote continues exactly; another code\'s is loaded into the current problem. Paused at the file\'s time.',
    inputSchema: obj({ path: { type: 'string' } }, ['path']),
    run: async ({ path: p }) => text(await call('loadAthdf', readFileSync(resolvePath(p)).toString('base64'))),
  },
  {
    name: 'debris',
    description: 'Debris energies, for problems that track them (describe: problems[].debris, e.g. tidal-disruption): the gas binned by its specific orbital energy about the point mass, both what has left the box (tallied as it left) and what is in it, a surviving self-bound core set apart, and the return rate of the bound debris to pericenter, dM/dt (Kepler). Returns totals, the core, the first return and the peak of dM/dt (in days and solar masses per year where the problem has units). save_path writes the full report (bins, dmdt.t and dmdt.rate) as JSON.',
    inputSchema: obj({ save_path: { type: 'string', description: 'Write the full report as JSON to this file' } }),
    async run({ save_path }) {
      const r = await call('debris');
      const out = debrisSummary(r);
      if (save_path) {
        const file = resolvePath(save_path);
        mkdirSync(path.dirname(file), { recursive: true });
        writeFileSync(file, JSON.stringify(r) + '\n');
        out.saved = file;
      }
      return text(out);
    },
  },
  {
    name: 'errors',
    description: 'Everything the simulator has reported going wrong (exceptions, WebGPU and shader errors), oldest first.',
    inputSchema: obj({}),
    run: async () => text(await call('errors')),
  },
];
const byName = new Map(TOOLS.map((t) => [t.name, t]));

// --- JSON-RPC over stdio ---------------------------------------------------------------------
const send = (msg) => process.stdout.write(JSON.stringify({ jsonrpc: '2.0', ...msg }) + '\n');

async function handle(msg) {
  const { id, method, params } = msg;
  const reply = (result) => id !== undefined && send({ id, result });
  const error = (code, message) => id !== undefined && send({ id, error: { code, message } });
  switch (method) {
    case 'initialize':
      return reply({
        protocolVersion: params?.protocolVersion ?? '2025-06-18',
        capabilities: { tools: {} },
        serverInfo: { name: 'shaderous', version: '0.1.0' },
        instructions: INSTRUCTIONS,
      });
    case 'ping':
      return reply({});
    case 'tools/list':
      return reply({ tools: TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })) });
    case 'tools/call': {
      const tool = byName.get(params?.name);
      if (!tool) return error(-32602, `unknown tool "${params?.name}"`);
      try {
        return reply(await tool.run(params.arguments ?? {}));
      } catch (e) {
        // A tool that fails is reported to the model, which can correct its call.
        return reply({ content: [{ type: 'text', text: `Error: ${e?.message ?? e}` }], isError: true });
      }
    }
    default:
      if (method?.startsWith('notifications/')) return; // nothing to answer
      return error(-32601, `method not found: ${method}`);
  }
}

const rl = createInterface({ input: process.stdin });
rl.on('line', (line) => {
  if (!line.trim()) return;
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return send({ id: null, error: { code: -32700, message: 'parse error' } });
  }
  handle(msg).catch((e) => log(`unhandled: ${e?.stack ?? e}`));
});
const shutdown = async () => {
  await (session ?? (await opening?.catch(() => null)))?.close();
  process.exit(0);
};
rl.on('close', shutdown);
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
log('ready (the simulator starts on the first tool call)');

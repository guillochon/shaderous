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
  options: { url: { type: 'string' }, headed: { type: 'boolean' }, 'low-priority': { type: 'boolean' }, 'gpu-memory': { type: 'string' } },
});
if (o['low-priority']) lowerPriority();
const log = (m) => process.stderr.write(`[shaderous-mcp] ${m}\n`);

// The keys of a setup, the view's settings and the variables, as the page's own tables give
// them (schema.json, written by `npm run gen`): nothing here restates them.
const SCHEMA = JSON.parse(readFileSync(new URL('./schema.json', import.meta.url), 'utf8'));
const linkKeysOf = (group) => SCHEMA.link.filter((k) => k.group === group).map((k) => k.key).join(', ');
const SETUP_KEYS = `${linkKeysOf('run')}, ${linkKeysOf('settings')}, and view keys ${linkKeysOf('view')}. dim is 2 or 3, amr 0 to 5, bc six digits. Unknown keys are ignored, values out of range clamped.`;
// The page's API by method (src/ui/api_schema.ts): what each does, and its arguments as JSON
// Schema. A tool that passes an argument on takes its schema from there with arg(); what a tool
// adds of its own (a file to save to, seconds instead of milliseconds) is written here.
const API = new Map(SCHEMA.api.map((m) => [m.name, m]));
const about = (method) => API.get(method).summary;
function arg(method, name) {
  const m = API.get(method);
  const found = m.args.find((a) => a.name === name) ?? m.args.find((a) => a.name === 'options')?.properties?.[name];
  if (!found) throw new Error(`the API's ${method} has no argument "${name}"`);
  const { name: _name, optional: _optional, ...schema } = found;
  return schema;
}
const VIEW_KEYS = SCHEMA.view.map((v) => (v.key === 'mode' ? 'mode (volume|slice|projection)' : v.key === 'field' ? 'field (index, see describe)' : v.key)).join(', ');

const INSTRUCTIONS = `Shaderous is a GPU gas dynamics code (compressible Euler equations with gravity, cooling, adaptive mesh refinement). These tools drive one simulation in a headless browser.
Typical use: describe (pick a problem) -> load (a setup: problem, dim, n, amr and other keys) -> run (to a time, landed on exactly, or a number of steps) -> diagnostics / stats / slice / probe for numbers, screenshot to look at it -> save_athdf to keep the state, record for a movie.
All values are in code units; positions are fractions of the box (0 to 1). A run stays paused between calls. load restarts from t = 0; control changes a setting mid-run as a person would.
GPU memory (how many blocks the adaptive mesh can hold) is the control "gpumem", in GB: control {id: "gpumem", value: 4} sets it and starts the run again; value "" returns to the guessed default (2.5 GB on a desktop). The server's --gpu-memory option sets it before the first run.`;

// --- the simulator page, opened on first use -----------------------------------------------
let session = null;
let opening = null;
async function sim() {
  if (session) return session;
  opening ??= openSession({ url: o.url, headed: o.headed, gpuMemory: o['gpu-memory'], log }).then((s) => (session = s)).finally(() => (opening = null));
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
    description: `${about('describe')} Give \`problem\` for one problem in full, with its description.`,
    inputSchema: obj({ problem: { type: 'string', description: 'A problem id for its full entry' } }),
    async run({ problem }) {
      const d = await call('describe');
      if (problem) {
        const p = d.problems.find((x) => x.id === problem);
        if (!p) throw new Error(`no problem "${problem}": call describe without arguments for the list`);
        // With its setup document: what `load` takes as `document`, to change and run.
        return text({ ...p, document: await call('preset', problem) });
      }
      return text({
        problems: d.problems.map((p) => ({ id: p.id, label: p.label, variant: p.variant, supports2d: p.supports2d, tEnd: p.tEnd, params: p.params.map((q) => `${q.id}=${q.value} [${q.min}, ${q.max}]`) })),
        fields: d.fields.map((f) => `${f.index}: ${f.name}`),
        variables: d.variables,
        resolutions: d.resolutions,
        eos: d.eos, cooling: d.cooling, riemann: d.riemann, limiter: d.limiter, gravity: d.gravity, colormaps: d.colormaps,
        setupDocument: {
          about: 'A run can also be assembled from parts as a setup document (load: document). describe with a problem returns the document of that preset to start from.',
          partKinds: d.setup.parts.map((k) => `${k.kind}: ${k.title}`),
          flows: d.setup.flows.map((f) => `${f.name}: ${f.label}`),
        },
        setupKeys: SETUP_KEYS,
      });
    },
  },
  {
    name: 'check_setup',
    description: about('check'),
    inputSchema: obj({ document: arg('check', 'document') }, ['document']),
    run: async ({ document }) => text(await call('check', document)),
  },
  {
    name: 'load',
    description: about('load'),
    inputSchema: obj({ setup: arg('load', 'setup'), document: arg('load', 'document') }),
    run: async ({ setup, document }) => text(await call('load', setup ?? '', ...(document ? [document] : []))),
  },
  {
    name: 'run',
    description: about('run'),
    inputSchema: obj({
      until: arg('run', 'until'),
      steps: arg('run', 'steps'),
      timeout_seconds: { type: 'number', description: 'Give up after this much wall time (default 600)' },
    }),
    run: async ({ until, steps, timeout_seconds }) => text(await call('run', { until, steps, timeoutMs: 1000 * (timeout_seconds ?? 600) })),
  },
  {
    name: 'state',
    description: about('state'),
    inputSchema: obj({}),
    run: async () => text(await call('state')),
  },
  {
    name: 'diagnostics',
    description: about('diagnostics'),
    inputSchema: obj({}),
    async run() {
      const d = await call('diagnostics');
      const r = (v) => (Array.isArray(v) ? v.map(r) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, r(x)])) : round(v));
      return text(r(d));
    },
  },
  {
    name: 'stats',
    description: about('stats'),
    inputSchema: obj({}),
    run: async () => text(await call('stats')),
  },
  {
    name: 'slice',
    description: 'A variable on a res x res grid on a plane through the box (perpendicular to `axis` at fraction `pos`). Returns min, max and mean, and the values (row j, column i: values[j][i]) when res is at most 64 or include_values is set; with save_path the full slice is written there as JSON instead. Variables: rho, press, vel1, vel2, vel3, r0, level, or a species name.',
    inputSchema: obj({
      variable: arg('slice', 'variable'),
      axis: arg('slice', 'axis'),
      pos: arg('slice', 'pos'),
      res: arg('slice', 'res'),
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
    description: about('probe'),
    inputSchema: obj({ points: arg('probe', 'points') }, ['points']),
    run: async ({ points }) => text(await call('probe', points)),
  },
  {
    name: 'screenshot',
    description: `${about('screenshot')} \`view\` changes what is drawn first: ${VIEW_KEYS}. Settings persist for later screenshots.`,
    inputSchema: obj({
      width: arg('screenshot', 'width'),
      height: arg('screenshot', 'height'),
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
    description: about('view'),
    inputSchema: obj({ changes: arg('view', 'changes') }, ['changes']),
    run: async ({ changes }) => text(await call('view', changes)),
  },
  {
    name: 'controls',
    description: about('controls'),
    inputSchema: obj({}),
    run: async () => text(await call('controls')),
  },
  {
    name: 'control',
    description: about('control'),
    inputSchema: obj({ id: arg('control', 'id'), value: arg('control', 'value') }, ['id']),
    async run({ id, value }) {
      await call('control', id, value);
      return text(await call('state'));
    },
  },
  {
    name: 'restart',
    description: about('restart'),
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
    name: 'record',
    description: `${about('record')} Returns the frame count, movie length and where the file went.`,
    inputSchema: obj({
      path: { type: 'string', description: 'File or folder to write to (default: the current folder, under the name of the movie)' },
      dt: arg('record', 'dt'),
      every: arg('record', 'every'),
      until: arg('record', 'until'),
      frames: arg('record', 'frames'),
      fps: arg('record', 'fps'),
      width: arg('record', 'width'),
      height: arg('record', 'height'),
      bitrate_mbps: { type: 'number', exclusiveMinimum: 0, description: 'Megabits per second of movie (default 0.2 bits per pixel per frame: 25 at 1080p and 60 fps)' },
      colorbar: { type: 'boolean', description: 'Default true' },
      axes: { type: 'boolean', description: 'Default true' },
      status: { type: 'boolean', description: 'The status line (grid, simulation time, point masses, mesh). Default true' },
      timeout_s: { type: 'number', description: 'Give up after this much wall time, keeping what was recorded' },
    }),
    async run({ path: p, every, dt, until, frames, fps, width, height, bitrate_mbps, colorbar = true, axes = true, status = true, timeout_s }) {
      if (until === undefined && frames === undefined) throw new Error('give `until` or `frames`, so that the movie ends');
      const m = await call('record', {
        ...(every !== undefined ? { every } : dt !== undefined ? { dt } : {}),
        until, frames, fps, width: width ?? 1920, height: height ?? (width === undefined ? 1080 : undefined),
        bitrate: bitrate_mbps === undefined ? undefined : bitrate_mbps * 1e6,
        overlays: { colorbar, triad: axes, status },
        timeoutMs: timeout_s === undefined ? undefined : timeout_s * 1000,
        download: false,
        base64: true,
      });
      let file = resolvePath(p ?? m.name);
      if (!p || p.endsWith('/') || p.endsWith('\\')) file = path.join(resolvePath(p ?? '.'), m.name);
      mkdirSync(path.dirname(file), { recursive: true });
      writeFileSync(file, Buffer.from(m.base64, 'base64'));
      const { base64, ...rest } = m;
      return text({ saved: file, ...rest });
    },
  },
  {
    name: 'load_athdf',
    description: about('loadAthdf'),
    inputSchema: obj({ path: { type: 'string' } }, ['path']),
    run: async ({ path: p }) => text(await call('loadAthdf', readFileSync(resolvePath(p)).toString('base64'))),
  },
  {
    name: 'debris',
    description: `${about('debris')} Returns totals, the core, the first return and the peak of dM/dt (in days and solar masses per year where the problem has units). save_path writes the full report (bins, dmdt.t and dmdt.rate) as JSON.`,
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
    description: about('errors'),
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

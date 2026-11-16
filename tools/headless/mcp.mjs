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
import { debrisSummary, fetchSlice, lowerPriority, openSession } from './session.mjs';
import { npyBytes } from './npy.mjs';

const { values: o } = parseArgs({
  options: { url: { type: 'string' }, headed: { type: 'boolean' }, 'low-priority': { type: 'boolean' }, 'gpu-memory': { type: 'string' } },
});
if (o['low-priority']) lowerPriority();
const log = (m) => process.stderr.write(`[shaderous-mcp] ${m}\n`);

// The keys of a setup, the view's settings and the variables, as the page's own tables give
// them (schema.json, written by `npm run gen`): nothing here restates them.
const SCHEMA = JSON.parse(readFileSync(new URL('./schema.json', import.meta.url), 'utf8'));
const PKG = JSON.parse(readFileSync(new URL('../../package.json', import.meta.url), 'utf8'));
const linkKeysOf = (group) => SCHEMA.link.filter((k) => k.group === group).map((k) => k.key).join(', ');
const SETUP_KEYS = `${linkKeysOf('run')}, ${linkKeysOf('settings')}, and view keys ${linkKeysOf('view')}. dim is 2 or 3, amr 0 to 5, bc six digits. Unknown keys are ignored, values out of range clamped.`;

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
const text = (v) => ({ content: [{ type: 'text', text: typeof v === 'string' ? v : JSON.stringify(v, null, 1) }] });
const round = (v) => (typeof v === 'number' && Number.isFinite(v) && !Number.isInteger(v) ? +v.toPrecision(7) : v);
const resolvePath = (p) => path.resolve(process.cwd(), p);

// What each tool does when called. Its name, description and input schema are in schema.json
// (written by `npm run gen` from src/ui/mcp_schema.ts and the API's own table), so nothing
// about what a tool accepts is written here.
const TOOLS = [
  {
    name: 'describe',
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
    run: async ({ document }) => text(await call('check', document)),
  },
  {
    name: 'load',
    run: async ({ setup, document }) => text(await call('load', setup ?? '', ...(document ? [document] : []))),
  },
  {
    name: 'run',
    run: async ({ until, steps, timeout_seconds }) => text(await call('run', { until, steps, timeoutMs: 1000 * (timeout_seconds ?? 600) })),
  },
  {
    name: 'state',
    run: async () => text(await call('state')),
  },
  {
    name: 'diagnostics',
    async run() {
      const d = await call('diagnostics');
      const r = (v) => (Array.isArray(v) ? v.map(r) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, r(x)])) : round(v));
      return text(r(d));
    },
  },
  {
    name: 'stats',
    run: async () => text(await call('stats')),
  },
  {
    name: 'slice',
    async run({ variable, axis, pos, res = 32, include_values, save_path }) {
      // Out of the page as float32 bytes: the same values as a JSON list, many times faster.
      const { head: s, values } = await fetchSlice(call, { variable, axis, pos, res });
      let sum = 0;
      for (const v of values) sum += v;
      const out = { t: s.t, variable: s.variable, axis: s.axis, pos: s.pos, res: s.res, axes: s.axes, min: s.min, max: s.max, mean: sum / values.length };
      if (save_path) {
        const file = resolvePath(save_path);
        mkdirSync(path.dirname(file), { recursive: true });
        writeFileSync(file, /\.npy$/i.test(file) ? npyBytes(values, [s.res, s.res]) : JSON.stringify({ ...s, values: Array.from(values) }) + '\n');
        return text({ ...out, saved: file });
      }
      if (include_values ?? res <= 64) {
        out.values = Array.from({ length: s.res }, (_, j) => Array.from(values.subarray(j * s.res, (j + 1) * s.res), (v) => +v.toPrecision(5)));
      }
      return text(out);
    },
  },
  {
    name: 'probe',
    run: async ({ points }) => text(await call('probe', points)),
  },
  {
    name: 'screenshot',
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
    run: async ({ changes }) => text(await call('view', changes)),
  },
  {
    name: 'controls',
    run: async () => text(await call('controls')),
  },
  {
    name: 'control',
    async run({ id, value }) {
      await call('control', id, value);
      return text(await call('state'));
    },
  },
  {
    name: 'restart',
    run: async () => text(await call('restart')),
  },
  {
    name: 'save_athdf',
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
    run: async ({ path: p }) => text(await call('loadAthdf', readFileSync(resolvePath(p)).toString('base64'))),
  },
  {
    name: 'debris',
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
    run: async () => text(await call('errors')),
  },
];
const byName = new Map(TOOLS.map((t) => [t.name, t]));
// Every tool the schema describes is one this file runs, and the other way round.
{
  const described = SCHEMA.tools.map((t) => t.name);
  const odd = [...described.filter((n) => !byName.has(n)), ...TOOLS.map((t) => t.name).filter((n) => !described.includes(n))];
  if (odd.length) throw new Error(`mcp.mjs and schema.json disagree about these tools: ${odd.join(', ')} (npm run gen?)`);
}

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
        serverInfo: { name: 'shaderous', version: PKG.version },
        instructions: INSTRUCTIONS,
      });
    case 'ping':
      return reply({});
    case 'tools/list':
      return reply({ tools: SCHEMA.tools.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })) });
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

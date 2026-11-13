#!/usr/bin/env node
// End-to-end check of the MCP server and the command-line runner, on this machine's GPU:
//   node tools/headless/selftest.mjs        (npm run test:headless)
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { REPO } from './session.mjs';

const tmp = mkdtempSync(path.join(os.tmpdir(), 'shaderous-selftest-'));
let failures = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
  if (!ok) failures++;
};

// --- MCP: a short conversation, as a client would have it --------------------------------
{
  const server = spawn(process.execPath, [path.join(REPO, 'tools/headless/mcp.mjs')], { stdio: ['pipe', 'pipe', 'inherit'] });
  const pending = new Map();
  let next = 1;
  createInterface({ input: server.stdout }).on('line', (line) => {
    const msg = JSON.parse(line); // anything on stdout that is not JSON-RPC fails here
    pending.get(msg.id)?.(msg);
  });
  const rpc = (method, params) =>
    new Promise((resolve) => {
      const id = next++;
      pending.set(id, resolve);
      server.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
    });
  const tool = async (name, args = {}) => (await rpc('tools/call', { name, arguments: args })).result;
  const json = (r) => JSON.parse(r.content[0].text);

  const init = await rpc('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'selftest', version: '0' } });
  check(init.result?.serverInfo?.name === 'shaderous', 'initialize');
  server.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n');
  const list = await rpc('tools/list', {});
  check(list.result.tools.some((t) => t.name === 'run') && list.result.tools.every((t) => t.inputSchema?.type === 'object'), `tools/list (${list.result.tools.length} tools)`);

  const st = json(await tool('load', { setup: 'problem=sedov&dim=2&n=256' }));
  check(st.problem === 'sedov' && st.t === 0, 'load');
  const r = json(await tool('run', { until: 0.03 }));
  check(r.stopped === 'time' && Math.abs(r.t - 0.03) < 1e-6, `run to t = 0.03 (${r.steps} steps)`);
  const d = json(await tool('diagnostics'));
  check(Math.abs(d.mass - 1) < 1e-5, 'diagnostics: mass conserved');
  const sl = json(await tool('slice', { variable: 'rho', res: 16 }));
  check(sl.values?.length === 16 && sl.values[0].length === 16 && sl.max > 1, 'slice');
  const shot = await tool('screenshot', { width: 256, view: { field: 3 } });
  check(shot.content[0].type === 'image' && shot.content[0].data.length > 5000, 'screenshot');
  const saved = json(await tool('save_athdf', { path: path.join(tmp, 'mcp/') }));
  check(existsSync(saved.saved), 'save_athdf');
  const back = json(await tool('load_athdf', { path: saved.saved }));
  check(Math.abs(back.t - r.t) < 1e-6, 'load_athdf continues at the same time');
  const ctl = json(await tool('control', { id: 'riemann', value: 'hll' }));
  check(ctl.query.includes('riemann=hll'), 'control');
  const bad = await tool('load', { setup: { problem: 'no-such-problem' } });
  check(bad.isError && /unknown problem/.test(bad.content[0].text), 'a bad call is reported as a tool error');
  const unknown = await rpc('tools/call', { name: 'nope', arguments: {} });
  check(unknown.error?.code === -32602, 'an unknown tool is a protocol error');
  check(json(await tool('errors')).length === 0, 'no page errors');
  server.stdin.end();
  await new Promise((r) => server.on('exit', r));
}

// --- the command-line runner --------------------------------------------------------------
{
  const out = path.join(tmp, 'cli');
  const res = spawnSync(process.execPath, [path.join(REPO, 'tools/headless/cli.mjs'), 'run', '--problem', 'sedov', '--dim', '2', '--n', '128', '--every', '0.02', '--until', '0.04', '--png', '--width', '128', '--athdf', '--slice', 'press', '--slice-res', '8', '--out', out, '--quiet'], { encoding: 'utf8' });
  check(res.status === 0, `cli run exits 0${res.status ? `: ${res.stderr}` : ''}`);
  const files = existsSync(out) ? readdirSync(out) : [];
  check(['0000-t0.png', '0002-t0.04.athdf', '0001-t0.02-press.json', 'diag.jsonl', 'manifest.json'].every((f) => files.includes(f)), 'cli outputs');
  const lines = existsSync(path.join(out, 'diag.jsonl')) ? readFileSync(path.join(out, 'diag.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l)) : [];
  check(lines.length === 3 && Math.abs(lines[2].t - 0.04) < 1e-6, 'cli outputs at t = 0, 0.02, 0.04');
  const bad = spawnSync(process.execPath, [path.join(REPO, 'tools/headless/cli.mjs'), 'run', '--problem', 'nope', '--out', path.join(tmp, 'bad'), '--quiet'], { encoding: 'utf8' });
  check(bad.status === 1 && /unknown problem/.test(bad.stderr) && !existsSync(path.join(tmp, 'bad')), 'cli reports a bad setup and writes nothing');
}

rmSync(tmp, { recursive: true, force: true });
console.log(failures ? `${failures} check(s) failed` : 'all checks passed');
process.exit(failures ? 1 : 0);

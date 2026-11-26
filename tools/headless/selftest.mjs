#!/usr/bin/env node
// End-to-end check of the MCP server and the command-line runner, on this machine's GPU:
//   node tools/headless/selftest.mjs        (npm run test:headless)
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { REPO } from './session.mjs';
import { readNpy } from './npy.mjs';

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

  // What a tool passes on to the page's API it describes as the API does (schema.json).
  {
    const schema = JSON.parse(readFileSync(path.join(REPO, 'tools/headless/schema.json'), 'utf8'));
    const api = new Map(schema.api.map((m) => [m.name, m]));
    const methodOf = Object.fromEntries(schema.tools.map((t) => [t.name, t.method]));
    const differing = [];
    for (const t of list.result.tools) {
      const m = api.get(methodOf[t.name]);
      if (!m) differing.push(`${t.name}: no such method`);
      for (const [key, schema] of Object.entries(t.inputSchema.properties ?? {})) {
        const found = m?.args.find((a) => a.name === key) ?? m?.args.find((a) => a.name === 'options')?.properties?.[key];
        if (!found) continue; // the tool's own (a file to save to)
        const { name: _n, optional: _o, ...want } = found;
        if (JSON.stringify(want) !== JSON.stringify(schema)) differing.push(`${t.name}.${key}`);
      }
    }
    check(differing.length === 0, `tools describe their arguments as the API does${differing.length ? `: ${differing.join(', ')}` : ''}`);
  }

  const st = json(await tool('load', { setup: 'problem=sedov&dim=2&n=256' }));
  check(st.problem === 'sedov' && st.t === 0, 'load');
  const r = json(await tool('run', { until: 0.03 }));
  check(r.stopped === 'time' && Math.abs(r.t - 0.03) < 1e-6, `run to t = 0.03 (${r.steps} steps)`);
  const d = json(await tool('diagnostics'));
  check(Math.abs(d.mass - 1) < 1e-5, 'diagnostics: mass conserved');
  const sl = json(await tool('slice', { variable: 'rho', res: 16 }));
  check(sl.values?.length === 16 && sl.values[0].length === 16 && sl.max > 1, 'slice');
  const big = json(await tool('slice', { variable: 'rho', res: 512, save_path: path.join(tmp, 'mcp/rho.npy') }));
  const npy = existsSync(big.saved) ? readNpy(readFileSync(big.saved)) : { shape: [], values: [] };
  const npyMean = npy.values.reduce((a, b) => a + b, 0) / npy.values.length;
  check(npy.shape.join() === '512,512' && Math.abs(npyMean - big.mean) < 1e-9 && npy.values.reduce((a, b) => Math.max(a, b), -Infinity) === big.max, 'slice: save_path .npy writes float32 [512, 512]');
  // A profile from the blast's center outward: two samples a cell (128 cells), inlined at 256.
  const pf = json(await tool('profile', { from: [0.5, 0.5], to: [1, 0.5], field: 'rho' }));
  const at = json(await tool('probe', { points: [[0.5 + 0.5 * (100 / 255), 0.5, 0]] }));
  check(pf.n === 256 && pf.values?.length === 256 && pf.max > 1 && pf.values[100] === +at.points[0].rho.toPrecision(5), 'profile: two samples a cell, as probe reads them');
  const pfs = json(await tool('profile', { from: [0, 0.5], to: [1, 0.5], field: 'Mach number', n: 1000, save_path: path.join(tmp, 'mcp/mach.csv') }));
  const csv = existsSync(pfs.saved ?? '') ? readFileSync(pfs.saved, 'utf8').trim().split('\n') : [];
  check(csv.length === 1001 && csv[0] === 's,value,level' && pfs.values === undefined && pfs.field === 'Mach number', 'profile: save_path .csv writes a row a sample');
  const shot = await tool('screenshot', { width: 256, view: { field: 3 } });
  check(shot.content[0].type === 'image' && shot.content[0].data.length > 5000, 'screenshot');
  // The PNG's size from its header: the grid's own (256 a side), and larger with overlays beside it.
  const pngSize = (r) => { const b = Buffer.from(r.content[0].data, 'base64'); return [b.readUInt32BE(16), b.readUInt32BE(20)]; };
  const native = pngSize(await tool('screenshot', { native: true }));
  const beside = pngSize(await tool('screenshot', { native: true, colorbar: true, axes: true, status: true }));
  check(native.join() === '256,256' && beside[0] > 256 && beside[1] > 256, `screenshot: native is one pixel per cell (${native.join(' x ')}; ${beside.join(' x ')} with overlays)`);
  const saved = json(await tool('save_athdf', { path: path.join(tmp, 'mcp/') }));
  check(existsSync(saved.saved), 'save_athdf');
  const back = json(await tool('load_athdf', { path: saved.saved }));
  check(Math.abs(back.t - r.t) < 1e-6, 'load_athdf continues at the same time');
  check(Array.isArray(back.warnings) && back.warnings.length === 0, 'load_athdf of a file from this build has no warnings');
  const mov = json(await tool('record', { path: path.join(tmp, 'mcp/'), every: 2, frames: 10, width: 320, height: 180 }));
  check(mov.frames === 10 && mov.width === 320 && existsSync(mov.saved) && readFileSync(mov.saved).subarray(4, 8).toString() === 'ftyp', 'record: an MP4 of 10 frames');
  const ctl = json(await tool('control', { id: 'riemann', value: 'hll' }));
  check(ctl.query.includes('riemann=hll'), 'control');
  const noDebris = await tool('debris');
  check(noDebris.isError && /does not track debris/.test(noDebris.content[0].text), 'debris on a problem without it is a tool error');
  const tde = json(await tool('load', { setup: 'problem=tidal-disruption&dim=3&n=32&p.q=1000000&p.beta=1' }));
  const tde0 = json(await tool('diagnostics'));
  await tool('run', { until: 1 });
  const tde1 = json(await tool('diagnostics'));
  const deb = json(await tool('debris', { save_path: path.join(tmp, 'debris.json') }));
  const budget = (tde1.mass + deb.totals.escaped - tde0.mass) / tde0.mass;
  check(tde.problem === 'tidal-disruption' && deb.totals.escaped > 0 && Math.abs(budget) < 1e-5 && existsSync(deb.saved), `debris: mass budget closes (${budget.toExponential(1)})`);
  // The editing tools, a session log saved to a file, and its replay.
  json(await tool('load', { setup: 'problem=mhd-rotor&dim=2&n=128' }));
  await tool('run', { steps: 10 });
  const pulled = json(await tool('edit', { name: 'pull', options: { from: [0.3, 0.5], to: [0.4, 0.55] } }));
  await tool('edit', { name: 'mass', options: { pos: [0.7, 0.3], amount: 0.3 } });
  await tool('run', { steps: 10 });
  const before = json(await tool('diagnostics'));
  const sess = json(await tool('session_log', { save_path: path.join(tmp, 'mcp/session.json') }));
  check(pulled.frames > 1 && sess.replayable && sess.kinds.brush === pulled.frames + 1 && existsSync(sess.saved), `edit and session_log (${sess.actions} actions)`);
  const rep = json(await tool('replay', { path: sess.saved, untilSteps: before.steps }));
  const after = json(await tool('diagnostics'));
  check(!rep.stopped && rep.applied === sess.actions && after.steps === before.steps && after.eTot === before.eTot && after.mass === before.mass, 'replay repeats the run to the bit');
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
  const nout = path.join(tmp, 'cli-npy');
  const nres = spawnSync(process.execPath, [path.join(REPO, 'tools/headless/cli.mjs'), 'run', '--problem', 'sedov', '--dim', '2', '--n', '64', '--until', '0.01', '--slice', 'rho', '--slice-res', '16', '--slice-format', 'npy', '--out', nout, '--quiet'], { encoding: 'utf8' });
  const nman = existsSync(path.join(nout, 'manifest.json')) ? JSON.parse(readFileSync(path.join(nout, 'manifest.json'), 'utf8')) : { outputs: [] };
  const last = nman.outputs.at(-1)?.slices?.[0];
  const nfile = last ? readFileSync(path.join(nout, last.file)) : Buffer.alloc(0);
  check(nres.status === 0 && last?.variable === 'rho' && last.res === 16 && nfile.length === 128 + 4 * 16 * 16 && readNpy(nfile).shape.join() === '16,16' && Math.max(...readNpy(nfile).values) === last.max,
    `cli --slice-format npy writes a .npy slice and its fields in the manifest${nres.status ? `: ${nres.stderr}` : ''}`);
  const mout = path.join(tmp, 'cli-mp4');
  const mres = spawnSync(process.execPath, [path.join(REPO, 'tools/headless/cli.mjs'), 'run', '--problem', 'sedov', '--dim', '2', '--n', '128', '--until', '0.01', '--mp4', '--frame-dt', '0.002', '--width', '320', '--out', mout, '--quiet'], { encoding: 'utf8' });
  const mman = existsSync(path.join(mout, 'manifest.json')) ? JSON.parse(readFileSync(path.join(mout, 'manifest.json'), 'utf8')) : {};
  check(mres.status === 0 && existsSync(path.join(mout, 'movie.mp4')) && mman.movie?.frames >= 5 && mman.movie?.stopped === 'time', `cli --mp4 writes movie.mp4${mres.status ? `: ${mres.stderr}` : ''}`);
  // Outputs by the wall clock, stopped by --wall, with two named views and a checkpoint only at the last.
  const wout = path.join(tmp, 'cli-wall');
  const wres = spawnSync(process.execPath, [path.join(REPO, 'tools/headless/cli.mjs'), 'run', '--problem', 'sedov', '--dim', '2', '--n', '256', '--set', 'tend=100', '--every-wall', '2', '--wall', '12', '--width', '64', '--shot', 'rho:field=0,log=true', '--shot', 'p:field=pressure,colormap=viridis', '--athdf-every', '0', '--overlays', '--out', wout, '--quiet'], { encoding: 'utf8' });
  const wman = existsSync(path.join(wout, 'manifest.json')) ? JSON.parse(readFileSync(path.join(wout, 'manifest.json'), 'utf8')) : { outputs: [] };
  const wfiles = wman.outputs.map((x) => x.files);
  check(wres.status === 0 && wman.stopped === 'wall' && wfiles.length >= 2 && wfiles.every((f, i) => f.some((n) => n.endsWith('-rho.png')) && f.some((n) => n.endsWith('-p.png')) && f.some((n) => n.endsWith('.athdf')) === (i === wfiles.length - 1)),
    `cli --every-wall, --wall, --shot and --athdf-every 0${wres.status ? `: ${wres.stderr}` : ''}`);
  // Carry on from the checkpoint the run above ended with.
  const wlast = wman.outputs.at(-1);
  const ckpt = wlast ? path.join(wout, wlast.files.find((f) => f.endsWith('.athdf')) ?? '') : '';
  const fout = path.join(tmp, 'cli-from');
  const fres = ckpt ? spawnSync(process.execPath, [path.join(REPO, 'tools/headless/cli.mjs'), 'run', '--from', ckpt, '--steps', '5', '--athdf-every', '1', '--out', fout, '--quiet'], { encoding: 'utf8' }) : { status: 1, stderr: 'no checkpoint' };
  const flines = existsSync(path.join(fout, 'diag.jsonl')) ? readFileSync(path.join(fout, 'diag.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l)) : [];
  check(fres.status === 0 && flines.length === 2 && Math.abs(flines[0].t - wlast.t) < 1e-6 * Math.max(1, wlast.t) && flines[1].steps === wlast.steps + 5, `cli --from carries on from a checkpoint${fres.status ? `: ${fres.stderr}` : ''}`);
  const fathdf = existsSync(fout) ? readdirSync(fout).filter((f) => f.endsWith('.athdf')) : [];
  check(fathdf.length === 1 && fathdf[0].startsWith('0001-'), `cli --from with --athdf-every writes no checkpoint of the file it started from (wrote ${fathdf.join(', ') || 'none'})`);
  // --every on a resumed run: outputs at the multiples of it after the start, none before.
  const ev = wlast ? wlast.t / 2.5 : 1;
  const eout = path.join(tmp, 'cli-from-every');
  const eres = ckpt ? spawnSync(process.execPath, [path.join(REPO, 'tools/headless/cli.mjs'), 'run', '--from', ckpt, '--every', String(ev), '--until', String(4 * ev), '--out', eout, '--quiet'], { encoding: 'utf8' }) : { status: 1, stderr: 'no checkpoint' };
  const elines = existsSync(path.join(eout, 'diag.jsonl')) ? readFileSync(path.join(eout, 'diag.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l)) : [];
  const near = (a, b) => Math.abs(a - b) < 1e-5 * b;
  check(eres.status === 0 && elines.length === 3 && near(elines[1].t, 3 * ev) && near(elines[2].t, 4 * ev),
    `cli --from with --every carries on at the next multiple (times ${elines.map((l) => l.t.toPrecision(4)).join(', ')})${eres.status ? `: ${eres.stderr}` : ''}`);
  const bad = spawnSync(process.execPath, [path.join(REPO, 'tools/headless/cli.mjs'), 'run', '--problem', 'nope', '--out', path.join(tmp, 'bad'), '--quiet'], { encoding: 'utf8' });
  check(bad.status === 1 && /unknown problem/.test(bad.stderr) && !existsSync(path.join(tmp, 'bad')), 'cli reports a bad setup and writes nothing');
}

rmSync(tmp, { recursive: true, force: true });
console.log(failures ? `${failures} check(s) failed` : 'all checks passed');
process.exit(failures ? 1 : 0);

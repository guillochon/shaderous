#!/usr/bin/env node
// The Moon-forming impact for make_impact.py: one run of the moon-forming-impact preset in
// headless Chrome (tools/headless/session.mjs, docs/headless.md), stopped at the figure's times.
// At each it saves slices through the orbital plane (z = 0.5 of the box) of the density, the
// passive scalar (1 in the proto-Earth, 0.5 in Theia) and the iron alloy's mass fraction as
// float32 .npy files, and the run's state, diagnostics and impact() report as JSON.
//
//   node paper/paper/figures/impact_run.mjs OUTDIR --setup QUERY --gpu-memory GB --times T1,T2,.. [--res R] [--every DT] [--vars V,..] [--shots FILE]
//
// --shots: a JSON file listing renderings to take at each of the times, { name, view, width? }
// (view as for shaderous.view(), docs/api.md), saved as tT-NAME.png with tT-NAME.json (the
// view in force, with the camera's angles).
// --doc PATH=VALUE (repeatable): a change to the preset's setup document, by its path (as in
// docs/setups.md: domain.box=10, mesh.refineRho=0.01); the value is read as JSON where it parses.
// --every: between the times, a line of the mesh's status and the totals every DT, to the log
// (diag.jsonl in OUTDIR). Run from the code's repository; it lowers its own priority.
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { fetchSlice, lowerPriority, openSession, pngBytes } from '../../../tools/headless/session.mjs';

const { values: o, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    setup: { type: 'string' },
    'gpu-memory': { type: 'string' },
    times: { type: 'string' },
    res: { type: 'string', default: '1024' },
    every: { type: 'string', default: '1' },
    vars: { type: 'string', default: 'rho,r0,fe85si15' },
    shots: { type: 'string' },
    doc: { type: 'string', multiple: true, default: [] },  },
});
const outDir = positionals[0];
if (!outDir || !o.setup || !o.times) {
  process.stderr.write('usage: node impact_run.mjs OUTDIR --setup QUERY --gpu-memory GB --times T1,T2,.. [--res R] [--every DT]\n');
  process.exit(1);
}
lowerPriority();
mkdirSync(outDir, { recursive: true });
const times = o.times.split(',').map(Number);
const res = Number(o.res);
const every = Number(o.every);
const shots = o.shots ? JSON.parse(readFileSync(o.shots, 'utf8')) : [];
const log = (m) => process.stderr.write(`[${new Date().toISOString().slice(11, 19)}] ${m}\n`);

/** A float32 [res, res] array as a .npy file (version 1.0, C order, little-endian). */
function npy(file, values, shape) {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}), }`;
  const pad = 64 - ((10 + header.length + 1) % 64);
  header += ' '.repeat(pad % 64) + '\n';
  const head = Buffer.alloc(10);
  head.write('\x93NUMPY', 0, 'latin1');
  head[6] = 1;
  head[7] = 0;
  head.writeUInt16LE(header.length, 8);
  writeFileSync(file, Buffer.concat([head, Buffer.from(header, 'latin1'), Buffer.from(values.buffer, values.byteOffset, values.byteLength)]));
}

const t0 = Date.now();
const s = await openSession({ setup: o.setup, gpuMemory: o['gpu-memory'], log });
try {
  if (o.doc.length) {
    const problem = new URLSearchParams(o.setup).get('problem');
    const doc = await s.call('preset', problem);
    for (const kv of o.doc) {
      const [key, raw] = [kv.slice(0, kv.indexOf('=')), kv.slice(kv.indexOf('=') + 1)];
      let value = raw;
      try {
        value = JSON.parse(raw);
      } catch {}
      const parts = key.split('.');
      let at = doc;
      for (const p of parts.slice(0, -1)) at = at[p] ??= {};
      at[parts.at(-1)] = value;
    }
    writeFileSync(path.join(outDir, 'document.json'), JSON.stringify(doc, null, 1) + '\n');
    await s.call('load', o.setup, doc);
  }
  const st = await s.call('state');
  log(`${st.query} on ${st.adapter}; mesh ${st.mesh}`);
  writeFileSync(path.join(outDir, 'state0.json'), JSON.stringify(st, null, 1) + '\n');
  const diagFile = path.join(outDir, 'diag.jsonl');
  const note = async (t) => {
    const r = await s.call('run', { until: t });
    const [stt, d, errors] = [await s.call('state'), await s.call('diagnostics'), await s.call('errors')];
    const line = { t: r.t, steps: r.steps, wall: (Date.now() - t0) / 1000, mesh: stt.mesh, mass: d.mass, eTot: d.eTot, floored: d.floored, errors: errors.length };
    appendFileSync(diagFile, JSON.stringify(line) + '\n');
    log(`t = ${r.t}, ${r.steps} steps, ${line.wall.toFixed(0)} s: ${stt.mesh}; mass ${d.mass}, eTot ${d.eTot}${errors.length ? `, ERRORS ${JSON.stringify(errors)}` : ''}`);
    return { r, stt, d, errors };
  };
  let t = (await s.call('state')).t;
  for (const T of times) {
    while (t + every < T - 1e-9) {
      t += every;
      await note(t);
    }
    const { r, stt, d, errors } = await note(T);
    t = T;
    const tag = `t${String(T).replace('.', 'p')}`;
    for (const v of o.vars.split(',')) {
      const { head, values } = await fetchSlice(s.call, { variable: v, axis: 2, pos: 0.5, res });
      npy(path.join(outDir, `${tag}-${v}.npy`), values, [res, res]);
      writeFileSync(path.join(outDir, `${tag}-${v}.json`), JSON.stringify(head, null, 1) + '\n');
    }
    for (const shot of shots) {
      const view = await s.call('view', shot.view ?? {});
      const width = shot.width ?? 1200;
      writeFileSync(path.join(outDir, `${tag}-${shot.name}.png`), pngBytes(await s.call('screenshot', { width, height: shot.height ?? width })));
      writeFileSync(path.join(outDir, `${tag}-${shot.name}.json`), JSON.stringify({ t: r.t, steps: r.steps, setup: stt.query, adapter: stt.adapter, view }, null, 1) + '\n');
    }
    let impact = null;
    try {
      impact = await s.call('impact');
    } catch (e) {
      impact = { error: String(e.message ?? e) };
    }
    const meta = { t: r.t, steps: r.steps, wall: (Date.now() - t0) / 1000, setup: stt.query, adapter: stt.adapter, mesh: stt.mesh, gpuMemory: o['gpu-memory'] ?? null, diagnostics: d, impact, errors };
    writeFileSync(path.join(outDir, `${tag}.json`), JSON.stringify(meta, null, 1) + '\n');
    if (impact && !impact.error) log(`  impact: planet ${impact.planet?.mass}, disk ${impact.disk?.mass}, escaping ${impact.escaping?.mass}, moon ${impact.moon}`);
  }
} finally {
  await s.close();
}
log(`done in ${((Date.now() - t0) / 1000).toFixed(0)} s`);

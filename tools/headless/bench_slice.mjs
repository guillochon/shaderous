#!/usr/bin/env node
// Times shaderous.slice() and splits the time between the page and the bridge to Node
// (plans/binary-slice-transport.md, steps 1 and 10). Runs at normal priority: it is a benchmark.
//   node tools/headless/bench_slice.mjs [--url http://127.0.0.1:5190/] [--reps 3]
// Against a static build (npx vite build --outDir X; npx vite preview --outDir X --port 5190),
// so that no edit to the checkout reloads the page mid-run.
import { openSession } from './session.mjs';

const arg = (k, d) => {
  const i = process.argv.indexOf(k);
  return i >= 0 ? process.argv[i + 1] : d;
};
const url = arg('--url');
const reps = Number(arg('--reps', 3));
const RES = [256, 1024, 2048];
const SETUPS = [
  { name: '2D uniform, n=1024', setup: 'problem=sedov&dim=2&n=1024', until: 0.02 },
  { name: '3D adaptive, n=64 amr=2', setup: 'problem=sedov&dim=3&n=64&amr=2', until: 0.02 },
];

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const ms = (x) => x.toFixed(0);
const mb = (x) => (x / 1e6).toFixed(1);

const s = await openSession({ url });
try {
  const hasEncoding = await s.page.evaluate(async () => 'array' in (await window.shaderous.slice({ res: 2, encoding: 'base64' }).catch(() => ({}))));
  console.log(`slice({ encoding: 'base64' }) ${hasEncoding ? 'is' : 'is not'} built in: ${hasEncoding ? 'timing it' : 'timing a prototype of it in the page'}`);
  for (const S of SETUPS) {
    await s.call('load', S.setup);
    await s.call('run', { until: S.until });
    const capture = [];
    for (let k = 0; k < reps; k++) capture.push(await s.page.evaluate(async () => { const t0 = performance.now(); await window.shaderous.probe([]); return performance.now() - t0; }));
    console.log(`\n${S.name}: snapshot capture ${ms(median(capture))} ms`);
    console.log('| res | page, json (ms) | bridge, json (ms) | json total (ms) | json bytes (MB) | page, base64 (ms) | bridge, base64 (ms) | base64 total (ms) | base64 bytes (MB) |');
    console.log('|---|---|---|---|---|---|---|---|---|');
    for (const res of RES) {
      const row = { pj: [], tj: [], pb: [], tb: [] };
      let jsonBytes = 0, b64Bytes = 0;
      for (let k = 0; k < reps; k++) {
        // JSON as today: the whole call, then the page's part of it alone.
        let t0 = performance.now();
        await s.call('slice', { res });
        row.tj.push(performance.now() - t0);
        const pj = await s.page.evaluate(async (res) => {
          const t0 = performance.now();
          const r = await window.shaderous.slice({ res });
          const dt = performance.now() - t0;
          return { dt, bytes: JSON.stringify(r).length };
        }, res);
        row.pj.push(pj.dt);
        jsonBytes = pj.bytes;
        // base64: built in if it is, else the prototype (a Float32Array, then a FileReader).
        const b64 = async (res, built, keep) => {
          const t0 = performance.now();
          let r;
          if (built) r = await window.shaderous.slice({ res, encoding: 'base64' });
          else {
            const { values, ...rest } = await window.shaderous.slice({ res });
            const f = Float32Array.from(values);
            const url = await new Promise((ok, no) => { const fr = new FileReader(); fr.onload = () => ok(fr.result); fr.onerror = () => no(fr.error); fr.readAsDataURL(new Blob([f])); });
            r = { ...rest, array: { dtype: 'float32', shape: [res, res], base64: url.slice(url.indexOf(',') + 1) } };
          }
          const dt = performance.now() - t0;
          return keep ? r : { dt, bytes: JSON.stringify(r).length };
        };
        t0 = performance.now();
        await s.page.evaluate(`(${b64})(${res}, ${hasEncoding}, true)`);
        row.tb.push(performance.now() - t0);
        const pb = await s.page.evaluate(`(${b64})(${res}, ${hasEncoding}, false)`);
        row.pb.push(pb.dt);
        b64Bytes = pb.bytes;
      }
      const [pj, tj, pb, tb] = [median(row.pj), median(row.tj), median(row.pb), median(row.tb)];
      console.log(`| ${res} | ${ms(pj)} | ${ms(tj - pj)} | ${ms(tj)} | ${mb(jsonBytes)} | ${ms(pb)} | ${ms(tb - pb)} | ${ms(tb)} | ${mb(b64Bytes)} |`);
    }
  }
} finally {
  await s.close();
}

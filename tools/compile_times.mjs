// How long each compute pipeline takes to compile when a setup loads, on a cold shader cache
// (a fresh browser profile), each compiled alone. Reports the slowest first. For finding passes whose shaders the
// browser takes long to compile.
//
//   node tools/compile_times.mjs [--url http://localhost:5183] [--setup "problem=orszag-tang"] [--top 15]
import { chromium } from '@playwright/test';
import { parseArgs } from 'node:util';

const { values: o } = parseArgs({
  options: { url: { type: 'string', default: 'http://localhost:5183' }, setup: { type: 'string', default: 'problem=orszag-tang' }, top: { type: 'string', default: '15' } },
});
const browser = await chromium.launch({
  channel: process.env.PW_CHANNEL ?? 'chrome',
  headless: !process.env.PW_HEADED,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=d3d11', '--disable-gpu-shader-disk-cache'],
});
const page = await browser.newPage();
await page.addInitScript(() => {
  const times = [];
  window.__compileTimes = times;
  for (const name of ['createComputePipelineAsync', 'createRenderPipelineAsync']) {
    const orig = GPUDevice.prototype[name];
    // One at a time, so each is timed alone (in parallel they share the compiler's threads).
    GPUDevice.prototype[name] = function (desc) {
      const run = () => {
        const t0 = performance.now();
        const p = orig.call(this, desc);
        return p.then((r) => (times.push({ label: desc.label ?? '?', ms: performance.now() - t0, kind: name.includes('Render') ? 'render' : 'compute' }), r));
      };
      const p = (window.__compileChain ?? Promise.resolve()).then(run, run);
      window.__compileChain = p.catch(() => {});
      return p;
    };
  }
});
// The setup in the page's own address, so it is the first and only thing loaded (as when a link
// is opened); headless=1 keeps it paused, with nothing drawn.
const t0 = Date.now();
await page.goto(`${o.url}/?headless=1&${o.setup}`);
await page.evaluate(() => window.shaderous.ready);
await page.evaluate(() => window.shaderous.run({ steps: 1 }));
const total = (Date.now() - t0) / 1000;
const times = await page.evaluate(() => window.__compileTimes);
times.sort((a, b) => b.ms - a.ms);
console.log(`${o.setup}: loaded and stepped in ${total.toFixed(1)} s; ${times.length} pipelines`);
for (const t of times.slice(0, +o.top)) console.log(`${(t.ms / 1000).toFixed(2).padStart(7)} s  ${t.kind.padEnd(7)} ${t.label}`);
await browser.close();

// Run the EOS benchmark (bench.js) in headless Chrome with WebGPU, as tools/headless does, and
// print its result as JSON. Assets first: tools/burn/.venv/Scripts/python tools/eos/bench/prep.py
//
//   node tools/eos/bench/run.mjs
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const { createServer } = await import('vite');
const server = await createServer({ root: HERE, configFile: false, server: { port: 0, host: '127.0.0.1' }, logLevel: 'error', clearScreen: false });
await server.listen();
const { chromium } = await import('@playwright/test');
const args = ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'];
if (process.platform === 'win32') args.push('--use-angle=d3d11');
const browser = await chromium.launch({ channel: process.env.PW_CHANNEL ?? 'chrome', headless: true, args });
try {
  const page = await browser.newPage();
  page.on('console', (m) => m.type() === 'error' && console.error(m.text()));
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(() => window.benchResult, undefined, { timeout: 600_000 });
  console.log(JSON.stringify(await page.evaluate(() => window.benchResult), null, 1));
} finally {
  await browser.close();
  await server.close();
}

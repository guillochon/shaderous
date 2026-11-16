// A headless simulator: the page served (by Vite from a checkout, from the built page in dist/
// when installed from npm, or from a URL given), opened in Chrome with WebGPU, and its scripting
// API (window.shaderous, docs/api.md) callable from Node. Shared by the command-line runner
// (cli.mjs) and the MCP server (mcp.mjs).
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createServer as createHttpServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const DIST = path.join(REPO, 'dist');

/**
 * What serves the page when no URL is given: 'vite' (the source, in a checkout) or 'dist' (the
 * built page; the npm package ships dist/ and no source). SHADEROUS_SERVE=dist or vite chooses.
 */
function serveMode() {
  const env = process.env.SHADEROUS_SERVE;
  if (env === 'dist' || env === 'vite') return env;
  return existsSync(path.join(REPO, 'src')) ? 'vite' : 'dist';
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.txt': 'text/plain',
};

/** The built page (dist/) on a free local port; resolves to { url, close }. */
async function serveDist() {
  if (!existsSync(path.join(DIST, 'index.html'))) throw new Error(`no built page in ${DIST}: run \`npm run build\`, or pass --url`);
  const server = createHttpServer(async (req, res) => {
    const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/^\/+/, '') || 'index.html';
    const file = path.resolve(DIST, rel);
    if (!file.startsWith(DIST + path.sep)) return res.writeHead(403).end();
    try {
      const body = await readFile(file);
      res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream' }).end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((ok, fail) => server.once('error', fail).listen(0, '127.0.0.1', ok));
  return { url: `http://127.0.0.1:${server.address().port}/`, close: () => new Promise((ok) => server.close(() => ok())) };
}

/** Run this process, and the browser it starts (which inherits it on Windows), at low priority. */
export function lowerPriority() {
  try {
    os.setPriority(os.constants.priority.PRIORITY_BELOW_NORMAL);
  } catch {
    // Not permitted here: carry on at normal priority.
  }
}

/** A setup as a query string: a link, a query string, or an object of query keys. */
export function setupQuery(setup) {
  if (!setup) return '';
  if (typeof setup === 'string') return setup.replace(/^[^?]*\?/, '').replace(/^\?/, '');
  return new URLSearchParams(Object.entries(setup).map(([k, v]) => [k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v)])).toString();
}

/** The message of an error thrown inside the page, without Playwright's wrapping. */
function pageMessage(e) {
  const m = String(e?.message ?? e);
  const inner = m.match(/page\.evaluate: (?:Error: )?([\s\S]*?)(?:\n\s+at |\n\nCall log|$)/);
  return (inner ? inner[1] : m).trim();
}

/**
 * Open a simulator page.
 * - url: where the page is served (default: a free local port, serving this checkout with Vite,
 *   or the built page in dist/ when there is no source, as in the npm package)
 * - setup: the run to open on (query string, link or object); default the page's own
 * - headed: show the browser window (when headless Chrome finds no GPU adapter)
 * - width, height: the window, which sets a screenshot's default size
 * - interactive: open the page as a person sees it (it draws and plays by itself) rather
 *   than in headless mode, for screenshots of the whole interface
 * - gpuMemory: the GPU memory the simulation may take, in GB (the panel's "GPU memory" setting):
 *   it sets how many blocks the adaptive mesh can hold. Set before the page loads, so its first
 *   run is built with it (each session starts from a fresh browser profile, with the guessed
 *   default, 2.5 GB on a desktop).
 * - scale: device pixels per CSS pixel (default 1): 2 for a sharper screenshot of the page
 * - log: where progress messages go (default stderr)
 */
export async function openSession(opts = {}) {
  const log = opts.log ?? ((m) => process.stderr.write(m + '\n'));
  let server = null;
  let base = opts.url;
  if (!base && serveMode() === 'dist') {
    server = await serveDist();
    base = server.url;
  } else if (!base) {
    const { createServer } = await import('vite');
    server = await createServer({
      root: REPO,
      configFile: path.join(REPO, 'vite.config.ts'),
      // No reload when a source file changes: a run keeps the code it started with.
      server: { port: 0, strictPort: false, host: '127.0.0.1', hmr: false, watch: null },
      logLevel: 'error',
      clearScreen: false,
    });
    await server.listen();
    base = server.resolvedUrls.local[0];
  }
  const { chromium } = await import('playwright-core');
  const args = ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist'];
  if (process.platform === 'win32') args.push('--use-angle=d3d11');
  // Linux (a GPU server or pod): WebGPU through Vulkan, with no window surface. Without
  // --disable-vulkan-surface the GPU process crashes headless, and Chrome falls back to
  // SwiftShader. The container needs the NVIDIA Vulkan driver (NVIDIA_DRIVER_CAPABILITIES=all).
  if (process.platform === 'linux') args.push('--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface');
  // SHADEROUS_NO_GPU_COMPOSITING=1: Chrome's own drawing off the GPU, for data-center GPUs with
  // little graphics hardware. On an H200 its compositor's first Vulkan submit lost the device,
  // which took WebGPU down with it. Runs, diagnostics and slices work so; a WebGPU canvas then
  // cannot be read, so screenshots and movie frames are drawn off it and read back from the GPU
  // (shaderousReadBack, src/ui/api.ts).
  if (process.env.SHADEROUS_NO_GPU_COMPOSITING) args.push('--disable-gpu-compositing', '--disable-gpu-rasterization', '--disable-accelerated-2d-canvas');
  // PW_CHANNEL=shell: Playwright's own Chromium headless shell (the Playwright Docker images
  // have it, not Chrome); on a Linux pod it is the one that finds the GPU.
  const channel = process.env.PW_CHANNEL ?? 'chrome';
  let browser;
  try {
    browser = await chromium.launch({ channel: channel === 'shell' ? undefined : channel, headless: !opts.headed, args });
  } catch (e) {
    await server?.close();
    throw new Error(`could not start Chrome (${pageMessage(e)}). Install Chrome, or set PW_CHANNEL=msedge for Edge.`);
  }
  const page = await browser.newPage({ viewport: { width: opts.width ?? 1280, height: opts.height ?? 800 }, deviceScaleFactor: opts.scale ?? 1 });
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  if (process.env.SHADEROUS_NO_GPU_COMPOSITING) await page.addInitScript(() => (globalThis.shaderousReadBack = true));
  if (opts.gpuMemory !== undefined) {
    const gb = Number(opts.gpuMemory);
    if (!(gb >= 0.25)) throw new Error(`GPU memory must be a number of GB, at least 0.25 (got "${opts.gpuMemory}")`);
    // The page's own budget key (src/gpu/memory.ts, USER_KEY), read when it starts.
    await page.addInitScript((bytes) => localStorage.setItem('shaderous.gpuBudget', String(bytes)), Math.round(gb * 1024 ** 3));
  }

  const u = new URL(base);
  const q = new URLSearchParams(setupQuery(opts.setup));
  if (!opts.interactive) q.set('headless', '1');
  u.search = q.toString();
  log(`shaderous: opening ${u.origin}${u.pathname}`);
  await page.goto(u.toString());
  const close = async () => {
    await browser.close().catch(() => {});
    await server?.close().catch(() => {});
  };
  try {
    await page.waitForFunction(() => window.shaderous, undefined, { timeout: 60_000 });
    await page.evaluate(() => window.shaderous.ready);
  } catch (e) {
    await close();
    throw new Error(`the simulator did not start: ${pageMessage(e)}${pageErrors.length ? `\n${pageErrors.join('\n')}` : ''}`);
  }

  /** Call window.shaderous[name](...args) in the page; errors come back as plain Errors. */
  const call = async (name, ...callArgs) => {
    try {
      return await page.evaluate(([n, a]) => window.shaderous[n](...a), [name, callArgs]);
    } catch (e) {
      throw new Error(pageMessage(e));
    }
  };
  return { page, call, close, url: u.toString() };
}

/**
 * The values of a slice fetched with encoding 'base64' (an EncodedArray, src/io/arrays.ts), as
 * a Float32Array of its own (copied out of the Buffer, so aligned).
 */
export function sliceArray(result) {
  const a = result.array;
  if (a?.dtype !== 'float32' || a.order !== 'C' || a.endian !== 'little') throw new Error('not a float32 array in C order, little-endian');
  const bytes = Buffer.from(a.base64, 'base64');
  const n = a.shape.reduce((x, y) => x * y, 1);
  if (bytes.length !== 4 * n) throw new Error(`array of shape [${a.shape}] came with ${bytes.length} bytes, not ${4 * n}`);
  const out = new Float32Array(n);
  new Uint8Array(out.buffer).set(bytes);
  return out;
}

/**
 * A slice brought out of the page as float32 bytes, many times faster than as a list of
 * numbers: its fields as slice() returns them but `values`, and the values as a Float32Array.
 * The values are those of slice()'s default JSON result, bit for bit.
 */
export async function fetchSlice(call, opts) {
  const r = await call('slice', { ...opts, encoding: 'base64' });
  const { array, ...head } = r;
  return { head, values: sliceArray(r) };
}

/** A PNG data URL as bytes. */
export const pngBytes = (dataUrl) => Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64');

const MSUN_KG = 1.98847e30;
const DAY_S = 86400;
const YEAR_S = 3.15576e7;
/**
 * The headline numbers of a debris() report: where the mass is, the surviving core, and the
 * peak of the return rate, in code units and, where the problem has units, in days and solar
 * masses per year.
 */
export function debrisSummary(r) {
  const k = r.dmdt.rate.reduce((b, x, i, a) => (x > a[b] ? i : b), 0);
  const peak = r.dmdt.rate.length ? { t: r.dmdt.t[k], rate: r.dmdt.rate[k] } : null;
  const first = r.dmdt.rate.findIndex((x) => x > 0);
  const out = {
    t: r.t, tPeri: r.tPeri, de: r.de, gm: r.gm, totals: r.totals, core: r.core,
    below: r.below, above: r.above,
    firstReturn: first >= 0 ? r.dmdt.t[first] : null,
    peak,
  };
  if (r.units && peak) {
    const days = (t) => ((t - r.tPeri) * r.units.time) / DAY_S;
    out.physical = {
      firstReturnDaysAfterPericenter: first >= 0 ? days(r.dmdt.t[first]) : null,
      peakDaysAfterPericenter: days(peak.t),
      peakMsunPerYear: (peak.rate * r.units.mass * YEAR_S) / (r.units.time * MSUN_KG),
    };
  }
  return out;
}

// A headless simulator: the page served (by Vite from a checkout, from the built page in dist/
// when installed from npm, or from a URL given), opened in Chrome with WebGPU, and its scripting
// API (window.shaderous, docs/api.md) callable from Node. Shared by the command-line runner
// (cli.mjs) and the MCP server (mcp.mjs).
import { closeSync, createReadStream, existsSync, openSync, readSync, statSync, writeSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createServer as createHttpServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { closeBrowser } from './close.mjs';

export const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
// SHADEROUS_DIST: another built page (a frozen build, kept while the source changes under a long batch).
const DIST = path.resolve(process.env.SHADEROUS_DIST ?? path.join(REPO, 'dist'));

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
/**
 * Files this process lets the page fetch (viewAthdf), by a token in their URL: served from where
 * they are, streamed, never read whole. Only files named here are served.
 */
const servedFiles = new Map();
const SERVED_PREFIX = '/__shaderous-file/';
function serveRegistered(req, res) {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (!p.startsWith(SERVED_PREFIX)) return false;
  const file = servedFiles.get(p.slice(SERVED_PREFIX.length).split('/')[0]);
  if (!file) return res.writeHead(404).end(), true;
  // Ranges: the page's reader asks for the parts of the file HDF5 needs (Emscripten's lazy file).
  const size = statSync(file).size;
  const r = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? '');
  const [start, end] = r ? [Number(r[1]), r[2] ? Math.min(Number(r[2]), size - 1) : size - 1] : [0, size - 1];
  res.writeHead(r ? 206 : 200, {
    'content-type': 'application/octet-stream',
    'accept-ranges': 'bytes',
    'content-length': end - start + 1,
    ...(r ? { 'content-range': `bytes ${start}-${end}/${size}` } : {}),
  });
  if (req.method === 'HEAD') return res.end(), true;
  createReadStream(file, { start, end }).pipe(res);
  return true;
}

async function serveDist() {
  if (!existsSync(path.join(DIST, 'index.html'))) throw new Error(`no built page in ${DIST}: run \`npm run build\`, or pass --url`);
  const server = createHttpServer(async (req, res) => {
    if (serveRegistered(req, res)) return;
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
      // Ahead of Vite's own handlers, whose page fallback would answer the files' URLs.
      plugins: [{ name: 'shaderous-served-files', configureServer: (s) => void s.middlewares.use((req, res, next) => serveRegistered(req, res) || next()) }],
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
    // Not browser.close() alone: on macOS Chrome's crash handler can hold it for good (close.mjs).
    await closeBrowser(browser);
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
  // Served from the source by Vite: the page's own modules can run here too. saveAthdf and
  // uploadAthdf then write and read .athdf files here, where their size is not bounded by the
  // page's memory (a Chromium page holds some 8 GB of arrays in all).
  if (server?.ssrLoadModule) call.loadModule = (id) => server.ssrLoadModule(id);
  // The page can fetch a file of this computer's from the server that serves it (viewAthdf).
  if (server) {
    call.serveFile = (file) => {
      const token = randomBytes(12).toString('hex');
      servedFiles.set(token, file);
      return `${SERVED_PREFIX}${token}/${encodeURIComponent(path.basename(file))}`;
    };
  }
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

/**
 * The run's .athdf file written to `file`, read out of the page in pieces (athdf({ chunked })),
 * so that a file of gigabytes is never one string. `name` is the file's name in the page (the
 * problem and time); `pick(name)` may choose the path from it. Returns { file, name, time, bytes }.
 */
export async function saveAthdf(call, pick) {
  if (call.loadModule) return saveAthdfHere(call, pick);
  const f = await call('athdf', { chunked: true });
  const file = typeof pick === 'function' ? pick(f.name) : pick;
  const CHUNK = 64 * 1024 * 1024;
  const fd = openSync(file, 'w');
  try {
    for (let offset = 0; offset < f.bytes; ) {
      const c = await call('athdfChunk', offset, CHUNK);
      const b = Buffer.from(c.base64, 'base64');
      writeSync(fd, b);
      offset = c.end;
    }
  } finally {
    closeSync(fd);
  }
  return { file, name: f.name, time: f.time, bytes: f.bytes };
}

/**
 * Start the page's run from the .athdf file at `file`, sent in pieces (athdfPut, then
 * loadAthdf()), so that a file of gigabytes is never one string. Returns what loadAthdf returns.
 */
export async function uploadAthdf(call, file) {
  if (call.loadModule) return uploadAthdfHere(call, file);
  await putAthdf(call, file);
  return call('loadAthdf');
}

/** The file at `file` put together in the page in pieces (athdfPut), for loadAthdf or viewAthdf. */
async function putAthdf(call, file) {
  const CHUNK = 48 * 1024 * 1024;
  const total = statSync(file).size;
  const fd = openSync(file, 'r');
  try {
    const buf = Buffer.alloc(Math.min(CHUNK, total));
    for (let offset = 0; offset < total; ) {
      const n = readSync(fd, buf, 0, Math.min(CHUNK, total - offset), offset);
      if (n <= 0) throw new Error(`${file}: short read at byte ${offset}`);
      await call('athdfPut', offset, buf.subarray(0, n).toString('base64'), total);
      offset += n;
    }
  } finally {
    closeSync(fd);
  }
}

/** The .athdf reader and writer (src/io/athdf.ts) and the build of HDF5 whose files are this computer's. */
async function athdfHere(call) {
  const [io, { default: h5 }] = await Promise.all([call.loadModule('/src/io/athdf.ts'), import('h5wasm/node')]);
  await h5.ready;
  return { ...io, h5 };
}

const SNAPSHOT_CHUNK = 64 * 1024 * 1024;

/** Put a snapshot's array in its place: "levels", "locations", "faces" or "data.<variable>". */
function placeArray(snap, key, a) {
  if (key === 'levels' || key === 'locations' || key === 'faces') snap[key] = a;
  else if (key.startsWith('data.')) snap.data[key.slice(5)] = a;
  else throw new Error(`a snapshot has no array "${key}"`);
}

/** A snapshot's arrays by key, as placeArray puts them. */
function arraysOf(snap) {
  const out = [['levels', snap.levels], ['locations', snap.locations]];
  for (const [k, a] of Object.entries(snap.data)) if (a) out.push([`data.${k}`, a]);
  if (snap.faces) out.push(['faces', snap.faces]);
  return out;
}

/** saveAthdf with the file written here: the page hands over the snapshot's arrays in pieces. */
async function saveAthdfHere(call, pick) {
  const { writeAthdfFile, h5 } = await athdfHere(call);
  const out = await call('snapshotOut');
  const file = typeof pick === 'function' ? pick(out.name) : pick;
  const snap = { ...out.header, data: {} };
  try {
    for (const { key, type, length } of out.arrays) {
      const a = type === 'int32' ? new Int32Array(length) : new Float32Array(length);
      const bytes = new Uint8Array(a.buffer);
      for (let offset = 0; offset < bytes.length; ) {
        const c = await call('snapshotChunk', key, offset, SNAPSHOT_CHUNK);
        bytes.set(Buffer.from(c.base64, 'base64'), offset);
        offset = c.end;
      }
      placeArray(snap, key, a);
    }
  } finally {
    await call('snapshotDone');
  }
  // gzip level 1, as the page writes its files.
  await writeAthdfFile(file, snap, h5, 1);
  return { file, name: out.name, time: snap.time, bytes: statSync(file).size };
}

/** uploadAthdf with the file read here: the page is handed the snapshot's arrays in pieces. */
async function uploadAthdfHere(call, file) {
  const { readAthdfFile, h5 } = await athdfHere(call);
  await sendSnapshot(call, await readAthdfFile(path.resolve(file), h5));
  return call('loadSnapshot');
}

/**
 * Open the .athdf file at `file` for viewing only (plans/view-only.md, the page's viewAthdf):
 * the page fetches it from the server that serves it (streamed from where it is), reads it in
 * parts, and shows it with its state or holding one field, its finest levels merged until it
 * fits its GPU memory budget; it does not step. `opts.oneField` picks the mode instead of the
 * page. Returns the page's state and `view`: the level kept, the file's finest, leaf blocks after
 * and before, and whether one field is held. A page this process does not serve is sent the
 * file in pieces.
 */
export async function viewAthdf(call, file, opts = {}) {
  if (call.serveFile) return call('viewAthdf', call.serveFile(path.resolve(file)), opts);
  await putAthdf(call, file);
  return call('viewAthdf', undefined, opts);
}

/** Hand the page a snapshot's arrays in pieces (snapshotIn, snapshotPut), for loadSnapshot. */
async function sendSnapshot(call, snap) {
  const arrays = arraysOf(snap);
  const { levels: _l, locations: _p, data: _d, faces: _f, ...header } = snap;
  await call('snapshotIn', header, arrays.map(([key, a]) => ({ key, type: a instanceof Int32Array ? 'int32' : 'float32', length: a.length })));
  for (const [key, a] of arrays) {
    const bytes = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
    for (let offset = 0; offset < bytes.length; offset += SNAPSHOT_CHUNK) {
      await call('snapshotPut', key, offset, bytes.subarray(offset, Math.min(bytes.length, offset + SNAPSHOT_CHUNK)).toString('base64'));
    }
  }
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

/**
 * Give the page the mesh files a setup document's `mesh` shapes name (plans/wind-tunnel.md 3.3),
 * read from `dir` (the setup document's folder): each by the name the setup gives it, and a
 * .gltf's separate buffers by their own. URLs are left to the page, which fetches them.
 */
export async function sendMeshFiles(call, spec, dir) {
  const { readFileSync } = await import('node:fs');
  const names = new Set();
  for (const p of spec?.parts ?? []) {
    const f = p?.shape?.type === 'mesh' ? p.shape.file : undefined;
    if (typeof f === 'string' && f && !/^(https?:|data:|blob:|builtin:)/.test(f)) names.add(f);
  }
  for (const name of names) {
    const file = path.resolve(dir, name);
    if (!existsSync(file)) throw new Error(`the setup's mesh file "${name}" is not in ${dir}`);
    const bytes = readFileSync(file);
    await call('meshFile', name, bytes.toString('base64'));
    if (name.toLowerCase().endsWith('.gltf')) {
      for (const b of JSON.parse(bytes.toString('utf8')).buffers ?? []) {
        if (!b.uri || b.uri.startsWith('data:')) continue;
        const f = path.resolve(path.dirname(file), b.uri);
        if (existsSync(f)) await call('meshFile', b.uri, readFileSync(f).toString('base64'));
      }
    }
  }
  return [...names];
}

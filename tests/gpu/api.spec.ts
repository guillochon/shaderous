// The scripting API, window.shaderous (src/ui/api.ts, docs/api.md), driven as an agent would:
// a headless page, every call through page.evaluate.
import { expect, test, type Page } from '@playwright/test';
import { decodeArray } from '../../src/io/arrays';

test.describe.configure({ mode: 'serial' });

/** Call window.shaderous[name](...args) in the page. */
const call = (page: Page, name: string, ...args: unknown[]) =>
  page.evaluate(([n, a]) => (window as any).shaderous[n as string](...(a as unknown[])), [name, args] as const);

/** Keep a screenshot for looking at: shown on a page of its own and saved from there. */
async function savePng(page: Page, dataUrl: string, name: string) {
  const p = await page.context().newPage();
  await p.setContent(`<body style="margin:0"><img src="${dataUrl}"></body>`);
  await p.locator('img').screenshot({ path: `test-results/api/${name}.png` });
  await p.close();
}

test('steps to a time, reads the gas back, and round-trips a file', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  const s0 = await call(page, 'load', { problem: 'sedov', dim: 2, n: 256 });
  expect(s0).toMatchObject({ problem: 'sedov', ndim: 2, n: 256, t: 0, steps: 0, running: false });

  // Ten steps, then exactly to t = 0.05, then on to the preset's end time and no further.
  expect(await call(page, 'step', 10)).toMatchObject({ steps: 10, taken: 10, stopped: 'steps' });
  const r = await call(page, 'runTo', 0.05);
  expect(r.stopped).toBe('time');
  expect(r.t).toBeCloseTo(0.05, 6);
  const end = await call(page, 'run', { until: 1 });
  expect(end.stopped).toBe('end');
  expect(end.t).toBeCloseTo(0.06, 6);
  expect((await call(page, 'state')).finished).toBe(true);

  const d = await call(page, 'diagnostics');
  expect(d.mass).toBeCloseTo(1, 5);
  expect(d.eTot).toBeCloseTo(1, 3);
  expect(d.ranges.Density[1]).toBeGreaterThan(3);

  const st = await call(page, 'stats');
  expect(st.variables.rho.mean).toBeCloseTo(1, 5);
  const sl = await call(page, 'slice', { variable: 'rho', res: 64 });
  expect(sl.values).toHaveLength(64 * 64);
  expect(sl.max).toBeGreaterThan(3);
  const pr = await call(page, 'probe', [[0.5, 0.5, 0], [0.95, 0.95, 0]]);
  expect(pr.points[0].rho).toBeLessThan(0.1); // the evacuated center
  expect(pr.points[1].rho).toBeCloseTo(1, 5); // gas the blast has not reached

  await call(page, 'view', { field: 3, colormap: 'viridis' });
  const png = await call(page, 'screenshot', { width: 512 });
  expect(png).toMatch(/^data:image\/png;base64,/);
  expect(png.length).toBeGreaterThan(10_000);
  await savePng(page, png, 'sedov-pressure');

  const file = await call(page, 'athdf');
  expect(file.name).toMatch(/\.athdf$/);
  const back = await call(page, 'loadAthdf', file.base64);
  expect(back).toMatchObject({ problem: 'sedov', ndim: 2, n: 256 });
  expect(back.t).toBeCloseTo(end.t, 6);
  expect((await call(page, 'diagnostics')).mass).toBeCloseTo(1, 5);
  expect(await call(page, 'errors')).toEqual([]);
});

test('an adaptive 3D star settles, steps, and shows its mesh', async ({ page }) => {
  await page.goto('/?headless=1&problem=polytrope&n=64&amr=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  // The first readout waits for the star to be settled in its initial conditions.
  const m0 = (await call(page, 'run', { steps: 0, timeoutMs: 120_000 }), await call(page, 'diagnostics')).mass;
  const r = await call(page, 'run', { until: 0.05, timeoutMs: 120_000 });
  expect(r.stopped).toBe('time');
  const d = await call(page, 'diagnostics');
  expect(d.mass / m0).toBeCloseTo(1, 4);
  const lv = await call(page, 'slice', { variable: 'level', axis: 2, res: 32 });
  expect(lv.max).toBe(1); // the star is refined
  expect(lv.values.every((v: number) => v === 0 || v === 1)).toBe(true);
  await call(page, 'view', { mode: 'slice', showBlocks: true });
  await savePng(page, await call(page, 'screenshot', { width: 640, height: 480 }), 'polytrope-blocks');
  expect(await call(page, 'errors')).toEqual([]);
});

test('a slice packed as float32 bytes has the values of the JSON one', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  for (const setup of [{ problem: 'sedov', dim: 2, n: 128 }, { problem: 'sedov', dim: 2, n: 64, amr: 2 }]) {
    await call(page, 'load', setup);
    await call(page, 'step', 30);
    for (const variable of ['rho', 'press', 'level']) {
      const json = await call(page, 'slice', { variable, res: 96 });
      const packed = await call(page, 'slice', { variable, res: 96, encoding: 'base64' });
      expect(packed.values).toBeUndefined();
      expect(packed.array).toMatchObject({ dtype: 'float32', shape: [96, 96], order: 'C', endian: 'little' });
      const { values, ...head } = json;
      const { array, ...packedHead } = packed;
      expect(packedHead).toEqual(head);
      const back = Array.from(decodeArray(array));
      expect(back).toEqual(values.map(Math.fround));
      expect(back).toEqual(values); // every variable is single precision already
    }
  }
  await expect(call(page, 'slice', { encoding: 'npy' })).rejects.toThrow(/unknown encoding "npy": use json or base64/);
  expect(await call(page, 'errors')).toEqual([]);
});

test('drives the panel controls and reports bad calls', async ({ page }) => {
  await page.goto('/?headless=1&problem=kelvin-helmholtz&dim=2&n=256');
  await page.evaluate(() => (window as any).shaderous.ready);
  const ids = (await call(page, 'controls')).map((c: { id: string }) => c.id);
  expect(ids).toEqual(expect.arrayContaining(['problem', 'riemann', 'limiter', 'cfl', 'reset']));

  await call(page, 'step', 20);
  await call(page, 'control', 'riemann', 'hll');
  await call(page, 'control', 'cfl', 0.3);
  expect(await page.evaluate(() => (window as any).shaderousApp.cfg)).toMatchObject({ riemann: 'hll', cfl: 0.3 });
  expect((await call(page, 'step', 5)).steps).toBe(25);
  await call(page, 'control', 'reset');
  expect(await call(page, 'state')).toMatchObject({ steps: 0, running: false });
  expect(await call(page, 'state').then((s) => s.query)).toContain('riemann=hll');

  await expect(call(page, 'control', 'riemann', 'roe')).rejects.toThrow(/no option "roe"/);
  await expect(call(page, 'load', { problem: 'no-such-thing' })).rejects.toThrow(/unknown problem/);
  await expect(call(page, 'view', { colour: 1 })).rejects.toThrow(/unknown view setting/);
  // A failed call leaves the API serving the next one.
  expect((await call(page, 'step', 1)).taken).toBe(1);
});

test('draws magnetic field lines: contours in 2D, tubes and slices in 3D, on blocks too', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  const cases: [string, Record<string, unknown>, Record<string, unknown>][] = [
    ['orszag-tang-lines', { problem: 'orszag-tang', dim: 2, n: 256 }, { field: 3 }],
    ['blast-3d-tubes', { problem: 'mhd-blast', dim: 3, n: 64 }, { mode: 'volume' }],
    ['blast-3d-blocks-tubes', { problem: 'mhd-blast', dim: 3, n: 128, amr: 1 }, { mode: 'volume' }],
    ['blast-3d-blocks-slice-lines', { problem: 'mhd-blast', dim: 3, n: 128, amr: 1 }, { mode: 'slice' }],
  ];
  for (const [name, setup, view] of cases) {
    await call(page, 'load', setup);
    await call(page, 'runTo', 0.05);
    // On by default with a field; the picture changes when they are turned off.
    expect((await call(page, 'view', view)).fieldLines).toBe(true);
    const on = await call(page, 'screenshot', { width: 512, height: 512 });
    await call(page, 'view', { fieldLines: false });
    const off = await call(page, 'screenshot', { width: 512, height: 512 });
    expect(on).not.toBe(off);
    await call(page, 'view', { fieldLines: true });
    await savePng(page, on, name);
  }
  expect(await call(page, 'errors')).toEqual([]);
});

/** Bytes from base64. */
const fromB64 = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

/** An MP4 box, found by its type anywhere in the file (enough for the few read here). */
function mp4Box(bytes: Uint8Array, type: string): DataView | null {
  const t = [...type].map((c) => c.charCodeAt(0));
  for (let i = 4; i + 4 <= bytes.length; i++) {
    if (t.every((c, k) => bytes[i + k] === c)) {
      const size = new DataView(bytes.buffer, i - 4, 4).getUint32(0);
      return new DataView(bytes.buffer, i - 4, size);
    }
  }
  return null;
}

test('records a movie: an MP4 with the frames and size asked for, and overlays', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  await call(page, 'load', { problem: 'kelvin-helmholtz', dim: 2, n: 128 });
  // The overlays alone, in a PNG.
  await call(page, 'step', 50);
  const png = await call(page, 'screenshot', { width: 640, height: 360, overlays: { colorbar: true, triad: true, status: true } });
  await savePng(page, png, 'overlays-2d');
  await call(page, 'view', { mode: 'volume' });

  const r = await call(page, 'record', { every: 4, frames: 12, width: 641, height: 361, fps: 24, download: false, base64: true });
  expect(r).toMatchObject({ frames: 12, width: 640, height: 360, steps: 44, stopped: 'frames' });
  // (from step 50)
  expect(r.seconds).toBeCloseTo(0.5, 6);
  expect(r.name).toMatch(/\.mp4$/);
  const bytes = fromB64(r.base64);
  expect(bytes.length).toBe(r.bytes);
  expect(String.fromCharCode(...bytes.subarray(4, 8))).toBe('ftyp');
  // One sample per frame, and the track's size (16.16 fixed point, at the end of tkhd).
  const stsz = mp4Box(bytes, 'stsz')!;
  expect(stsz.getUint32(16)).toBe(12);
  const tkhd = mp4Box(bytes, 'tkhd')!;
  expect(tkhd.getUint32(tkhd.byteLength - 8) / 65536).toBe(640);
  expect(tkhd.getUint32(tkhd.byteLength - 4) / 65536).toBe(360);
  expect(await call(page, 'recordingStatus')).toBeNull();
  // The run steps on as before, and the page draws its own view again.
  expect(await call(page, 'step', 3)).toMatchObject({ taken: 3 });
});

test('a movie with a time cadence leaves the run exactly as it would have been', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  for (const setup of [{ problem: 'sedov', dim: 2, n: 128 }, { problem: 'sedov', dim: 2, n: 64, amr: 2, tbins: 2 }]) {
    await call(page, 'load', setup);
    const r = await call(page, 'record', { dt: 2e-4, frames: 25, width: 320, height: 320, download: false });
    expect(r.frames).toBe(25);
    // A frame at the step boundary nearest each multiple of dt: never more frames than steps,
    // and none sooner than every dt (frame times a step jumps over are skipped, so later ones
    // can be late).
    expect(r.steps).toBeGreaterThanOrEqual(24);
    expect(r.t).toBeGreaterThan(0.9 * 24 * 2e-4);
    await call(page, 'step', 20);
    const rec = await call(page, 'slice', { variable: 'press', res: 128 });
    const recState = await call(page, 'state');

    await call(page, 'load', setup);
    await call(page, 'step', r.steps + 20);
    const plain = await call(page, 'slice', { variable: 'press', res: 128 });
    const plainState = await call(page, 'state');
    expect(recState.steps).toBe(plainState.steps);
    expect(recState.t).toBe(plainState.t);
    expect(rec.values).toEqual(plain.values);
  }
});

test('checkpoints are written while a movie records, and while the API steps', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  // A folder held in memory, in place of the one the user would pick.
  await page.evaluate(() => {
    const files: string[] = [];
    (window as any).__ckptFiles = files;
    (window as any).showDirectoryPicker = async () => ({
      name: 'test',
      getFileHandle: async (name: string) => ({
        createWritable: async () => ({ write: async () => void files.push(name), close: async () => {} }),
      }),
    });
  });
  const files = () => page.evaluate(() => [...(window as any).__ckptFiles] as string[]);
  await call(page, 'load', { problem: 'sedov', dim: 2, n: 128 });
  await call(page, 'control', 'ckdt', 0.002);
  await call(page, 'control', 'ckpt', true);
  // The folder is picked asynchronously: wait for the first checkpoint to be armed.
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.tStop !== undefined);

  // run() goes through checkpoints, writing each, to its own stop.
  const r = await call(page, 'run', { until: 0.005 });
  expect(r.stopped).toBe('time');
  expect(r.t).toBeCloseTo(0.005, 6);
  expect((await files()).map((f) => f.match(/ckpt(\d+)/)![1])).toEqual(['0001', '0002']);

  // A recording, with either cadence, does too.
  const m = await call(page, 'record', { every: 2, until: 0.009, width: 320, height: 180, download: false });
  expect(m.stopped).toBe('time');
  expect(m.t).toBeCloseTo(0.009, 6);
  const m2 = await call(page, 'record', { dt: 5e-4, until: 0.013, width: 320, height: 180, download: false });
  expect(m2.stopped).toBe('time');
  expect(m2.frames).toBeGreaterThan(5);
  expect((await files()).map((f) => f.match(/ckpt(\d+)/)![1])).toEqual(['0001', '0002', '0003', '0004', '0005', '0006']);
  expect((await call(page, 'state')).finished).toBe(false);
  expect(await call(page, 'errors')).toEqual([]);
});

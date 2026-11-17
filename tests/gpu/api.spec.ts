// The scripting API, window.shaderous (src/ui/api.ts, docs/api.md), driven as an agent would:
// a headless page, every call through page.evaluate.
import { expect, test, type Page } from '@playwright/test';
import { decodeArray } from '../../src/io/arrays';
import { besideLayout } from '../../src/render/overlays';

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

test('the field tools strengthen, weaken and pull the magnetic field, keeping div B', async ({ page }) => {
  // A page someone is watching (the tools paint in its frame loop), paused.
  await page.goto('/?problem=mhd-rotor&dim=2&n=256');
  await page.evaluate(() => (window as any).shaderous.ready);
  await call(page, 'runTo', 0.02);
  await call(page, 'pause');
  const t0 = (await call(page, 'diagnostics')).t;
  const mag = async () => (await call(page, 'diagnostics')).magnetic as { eMag: number; divB: number };
  const box = (await page.locator('#view').boundingBox())!;
  const [cx, cy] = [box.x + box.width / 2, box.y + box.height / 2];
  // A stroke is applied by the frame loop: wait for a few frames after it.
  const frames = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(r)))));
  const click = async (shift = false) => {
    if (shift) await page.keyboard.down('Shift');
    await page.mouse.move(cx, cy);
    await page.mouse.down();
    await frames();
    await page.mouse.up();
    await frames();
    if (shift) await page.keyboard.up('Shift');
  };
  const m0 = await mag();
  await page.click('#tool-field');
  await click();
  const m1 = await mag();
  await click(true);
  const m2 = await mag();
  expect(m1.eMag).toBeGreaterThan(m0.eMag * 1.002);
  expect(m2.eMag).toBeLessThan(m1.eMag);
  await page.click('#tool-pull');
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  for (let i = 1; i <= 10; i++) {
    await page.mouse.move(cx + 8 * i, cy + 3 * i);
    await frames();
  }
  await page.mouse.up();
  await frames();
  const m3 = await mag();
  expect(m3.eMag).not.toBe(m2.eMag);
  for (const m of [m1, m2, m3]) expect(m.divB).toBeLessThan(1e-6);
  // Paused throughout: the changes are the tools'.
  expect((await call(page, 'diagnostics')).t).toBe(t0);
  // The page's own tools are logged as they are applied.
  const log = await call(page, 'actions');
  const ops = log.entries.filter((e: any) => e.action.kind === 'brush').map((e: any) => e.action.op.kind);
  expect(ops).toContain('field');
  expect(ops).toContain('pull');
  // Runs without a field do not offer them.
  await call(page, 'load', { problem: 'sedov', dim: 2, n: 64 });
  await expect(page.locator('#tool-field')).toBeHidden();
  await expect(page.locator('#tool-pull')).toBeHidden();
});

test('every editing tool works from the API, and is logged', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  await call(page, 'load', { problem: 'mhd-rotor', dim: 2, n: 128 });
  await call(page, 'step', 5);
  const diag = () => call(page, 'diagnostics');
  const at = async (pos: number[]) => (await call(page, 'probe', [pos])).points[0];
  const d0 = await diag();

  await call(page, 'tool', 'mass', { pos: [0.25, 0.25], radius: 0.05, amount: 0.5 });
  expect((await diag()).mass).toBeGreaterThan(d0.mass);

  await call(page, 'tool', 'velocity', { pos: [0.75, 0.25], radius: 0.05, vel: [0.3, -0.2] });
  const v = (await at([0.75, 0.25, 0.5])).vel;
  expect(v[0]).toBeCloseTo(0.3, 3);
  expect(v[1]).toBeCloseTo(-0.2, 3);

  const p0 = (await at([0.75, 0.75, 0.5])).press;
  await call(page, 'tool', 'temperature', { pos: [0.75, 0.75], radius: 0.05, factor: 2 });
  expect((await at([0.75, 0.75, 0.5])).press / p0).toBeCloseTo(2, 2);

  const e0 = (await diag()).magnetic.eMag;
  await call(page, 'tool', 'field', { pos: [0.25, 0.75], factor: 1.5 });
  expect((await diag()).magnetic.eMag).toBeGreaterThan(e0);
  const pull = await call(page, 'tool', 'pull', { from: [0.3, 0.5], to: [0.4, 0.55], grip: 0.5 });
  expect(pull.frames).toBeGreaterThan(1);
  expect((await diag()).magnetic.divB).toBeLessThan(1e-6);

  // An obstacle holds its gas at rest.
  await call(page, 'tool', 'velocity', { pos: [0.1, 0.9], radius: 0.06, vel: [0.5, 0] });
  await call(page, 'tool', 'obstacle', { shape: 'rect', center: [0.1, 0.9], half: [0.03, 0.03] });
  expect((await at([0.1, 0.9, 0.5])).vel[0]).toBe(0);

  await call(page, 'tool', 'source', { pos: [0.5, 0.1], fixed: true });
  const sink = await call(page, 'tool', 'sink', { pos: [0.9, 0.1], fixed: true });
  expect(sink.index).toBe(0);
  expect((await diag()).masses.length).toBe(1);
  await call(page, 'tool', 'clear', { what: 'sinks' });
  await call(page, 'tool', 'clear', { what: 'sources' });
  expect((await diag()).masses.length).toBe(0);

  const q = await call(page, 'tool', 'query', { pos: [0.75, 0.25] });
  expect(q.rho).toBeGreaterThan(0);

  await expect(call(page, 'tool', 'star', { pos: [0.5, 0.5] })).rejects.toThrow(/2D run/);
  await expect(call(page, 'tool', 'brush', {})).rejects.toThrow(/no tool "brush"/);
  await expect(call(page, 'tool', 'field', { pos: [0.5, 0.5], strength: 2 })).rejects.toThrow(/no option "strength"/);

  const log = await call(page, 'actions');
  expect(log.format).toBe('shaderous-session');
  const brushes = log.entries.filter((e: any) => e.action.kind === 'brush').map((e: any) => e.action.op.kind);
  for (const k of ['mass', 'velocity', 'temperature', 'field', 'pull', 'obstacle']) expect(brushes).toContain(k);
  const kinds = log.entries.map((e: any) => e.action.kind);
  expect(kinds).toEqual(expect.arrayContaining(['sources', 'mass', 'remove']));
  // Every action has its step, and none is before the five taken first.
  expect(log.entries.every((e: any) => e.steps >= 5)).toBe(true);
});

test('a session log replays to the bit: tools, a setting, and the adaptive mesh re-meshing', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  await call(page, 'load', { problem: 'mhd-rotor', dim: 2, n: 256, amr: 1 });
  await call(page, 'step', 20);
  await call(page, 'tool', 'pull', { from: [0.4, 0.5], to: [0.5, 0.55] });
  await call(page, 'step', 15);
  await call(page, 'control', 'cfl', 0.3);
  await call(page, 'step', 15);
  await call(page, 'tool', 'mass', { pos: [0.7, 0.3], amount: 0.3 });
  await call(page, 'tool', 'field', { pos: [0.3, 0.3], factor: 1.3 });
  await call(page, 'step', 40);
  const a = await call(page, 'diagnostics');
  const sa = await call(page, 'slice', { variable: 'rho', res: 128 });
  const log = await call(page, 'actions');
  const kinds = log.entries.map((e: any) => e.action.kind);
  expect(kinds).toContain('regrid');
  expect(kinds).toContain('set');

  const r = await call(page, 'replay', log, { untilSteps: a.steps });
  expect(r.stopped).toBeUndefined();
  expect(r.applied).toBe(log.entries.length);
  const b = await call(page, 'diagnostics');
  expect(b.steps).toBe(a.steps);
  expect(b.t).toBe(a.t);
  expect(b.mass).toBe(a.mass);
  expect(b.eTot).toBe(a.eTot);
  expect(b.magnetic.eMag).toBe(a.magnetic.eMag);
  const sb = await call(page, 'slice', { variable: 'rho', res: 128 });
  expect(sb.values).toEqual(sa.values);
  // The replay logs itself, action for action.
  const again = await call(page, 'actions');
  expect(again.entries.map((e: any) => [e.steps, e.action.kind])).toEqual(log.entries.map((e: any) => [e.steps, e.action.kind]));
});

test('a session the page played, edited with the pointer, replays to the bit', async ({ page }) => {
  // The page playing on its own, its adaptive mesh re-meshing when its checks come back.
  await page.goto('/?problem=driven-turbulence-mhd&dim=2&n=256&amr=2');
  await page.evaluate(() => (window as any).shaderous.ready);
  await call(page, 'runTo', 0.05);
  await call(page, 'play');
  const frames = (k: number) => page.evaluate((k) => new Promise((r) => { let i = 0; const f = () => (++i >= k ? r(null) : requestAnimationFrame(f)); requestAnimationFrame(f); }), k);
  await frames(20);
  const box = (await page.locator('#view').boundingBox())!;
  await page.click('#tool-pull');
  const [cx, cy] = [box.x + box.width * 0.4, box.y + box.height * 0.5];
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  for (let i = 1; i <= 40; i++) {
    await page.mouse.move(cx + 3 * i, cy + i);
    await frames(1);
  }
  await page.mouse.up();
  await frames(30);
  await call(page, 'pause');
  const a = await call(page, 'diagnostics');
  const log = await call(page, 'actions');
  const kinds = log.entries.map((e: any) => e.action.kind);
  expect(kinds).toContain('regrid');
  expect(kinds).toContain('brush');

  const r = await call(page, 'replay', log, { untilSteps: a.steps });
  expect(r.stopped).toBeUndefined();
  const b = await call(page, 'diagnostics');
  // Where the replay first went another way, if it did: each action's time, against the log's.
  const again = await call(page, 'actions');
  const first = log.entries.findIndex((e: any, i: number) => again.entries[i]?.t !== e.t || again.entries[i]?.steps !== e.steps);
  if (first >= 0) {
    const show = (l: any, i: number) => (l.entries[i] ? `${l.entries[i].steps} ${l.entries[i].t} ${l.entries[i].action.kind}${l.entries[i].action.op ? ' ' + l.entries[i].action.op.kind : ''}` : '-');
    for (let i = Math.max(0, first - 4); i <= first + 1; i++) console.log(i, 'log:', show(log, i), '| replay:', show(again, i));
  }
  expect([b.steps, b.t, b.mass, b.eTot, b.magnetic.eMag, b.magnetic.bMax]).toEqual([a.steps, a.t, a.mass, a.eTot, a.magnetic.eMag, a.magnetic.bMax]);
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

/** Pixels of PNG data URLs, decoded in the page: [width, height, red channel as numbers]. */
const reds = (page: Page, urls: string[]) =>
  page.evaluate(async (us) => {
    const out: [number, number, number[]][] = [];
    for (const u of us) {
      const img = new Image();
      img.src = u;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = img.width;
      c.height = img.height;
      const g = c.getContext('2d')!;
      g.drawImage(img, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height).data;
      out.push([c.width, c.height, Array.from({ length: c.width * c.height }, (_, k) => d[4 * k])]);
    }
    return out;
  }, urls);

test('a native screenshot has one pixel per cell, the overlays beside it', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  await call(page, 'load', { problem: 'kelvin-helmholtz', dim: 2, n: 128 });
  await call(page, 'step', 40);
  // A linear gray scale over a fixed range: a pixel's gray is its cell's density.
  const [lo, hi] = [0.9, 2.1];
  await call(page, 'view', { colormap: 'gray', autoRange: false, lo, hi, log: false, contours: false, showBlocks: false });
  const n = 128;
  const sl = await call(page, 'slice', { variable: 'rho', res: n });
  const bare = await call(page, 'screenshot', { native: true });
  const all = { colorbar: true, triad: true, status: true };
  const dressed = await call(page, 'screenshot', { native: true, overlays: all });
  await savePng(page, dressed, 'native-2d');
  const [[w, h, px], [W, H, big]] = await reds(page, [bare, dressed]);
  expect([w, h]).toEqual([n, n]);
  // Row 0 of the picture is the top of the box; slice() counts rows from the bottom.
  let worst = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(px[j * n + i] - (255 * (sl.values[(n - 1 - j) * n + i] - lo)) / (hi - lo)));
  expect(worst).toBeLessThan(1.5);
  // With the overlays: a larger picture, the map in it untouched.
  expect(W).toBeGreaterThan(n);
  expect(H).toBeGreaterThan(n);
  const at = besideLayout(n, n, { ticks: 0, title: 0, status: 0 }, all).map;
  let differ = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) if (big[(j + at.y) * W + i + at.x] !== px[j * n + i]) differ++;
  expect(differ).toBe(0);

  // The part in view, zoomed about an off-center point: a crop of the whole box, in whole cells.
  await call(page, 'view', { zoom: 4, cx: 0.3, cy: 0.62 });
  const [[vw, vh, part]] = await reds(page, [await call(page, 'screenshot', { native: 'view' })]);
  expect(vw).toBeLessThan(n);
  expect(vh).toBeLessThan(n);
  let found = false;
  for (let j0 = 0; j0 + vh <= n && !found; j0++) {
    for (let i0 = 0; i0 + vw <= n && !found; i0++) {
      let same = true;
      for (let j = 0; j < vh && same; j++) for (let i = 0; i < vw && same; i++) same = part[j * vw + i] === px[(j + j0) * n + i + i0];
      found = same;
    }
  }
  expect(found).toBe(true);
  await call(page, 'view', { zoom: 1, cx: 0.5, cy: 0.5 });

  // Not with a size; a sized picture may have its overlays beside it too.
  await expect(call(page, 'screenshot', { native: true, width: 64 })).rejects.toThrow(/without `width`/);
  const [[sw, sh]] = await reds(page, [await call(page, 'screenshot', { width: 300, height: 200, layout: 'beside', overlays: { colorbar: true } })]);
  expect(sw).toBeGreaterThan(300);
  expect(sh).toBeGreaterThan(200);
});

test('a native screenshot of an adaptive mesh shows a coarse cell as a square of equal pixels', async ({ page }) => {
  await page.goto('/?headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  await call(page, 'load', { problem: 'kelvin-helmholtz', dim: 2, n: 256, amr: 2 });
  await call(page, 'step', 20);
  await call(page, 'view', { colormap: 'gray', autoRange: true, log: false, contours: false, showBlocks: false });
  const [[w, h, px]] = await reds(page, [await call(page, 'screenshot', { native: true })]);
  expect([w, h]).toEqual([256, 256]);
  // Far from the shear layers, in the smooth band along the top, the mesh is coarse: every
  // aligned 4 x 4 square of pixels there is one cell.
  const squares = (y0: number, y1: number) => {
    let equal = 0;
    let total = 0;
    for (let j = y0; j < y1; j += 4) {
      for (let i = 0; i < w; i += 4) {
        total++;
        let same = true;
        for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) same &&= px[(j + b) * w + i + a] === px[j * w + i];
        if (same) equal++;
      }
    }
    return equal / total;
  };
  expect(squares(0, 16)).toBe(1);
  // In the 3D volume view there is no pixel per cell.
  await call(page, 'load', { problem: 'sedov', dim: 3, n: 32 });
  await call(page, 'view', { mode: 'volume' });
  await expect(call(page, 'screenshot', { native: true })).rejects.toThrow(/volume view/);
  await call(page, 'view', { mode: 'slice' });
  const [[sw, sh]] = await reds(page, [await call(page, 'screenshot', { native: true })]);
  expect([sw, sh]).toEqual([32, 32]);
});

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

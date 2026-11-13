// The scripting API, window.shaderous (src/ui/api.ts, docs/api.md), driven as an agent would:
// a headless page, every call through page.evaluate.
import { expect, test, type Page } from '@playwright/test';

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

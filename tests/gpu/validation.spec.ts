// Runs the in-page validation suite (src/validation/tests.ts) on a real GPU.
import { expect, test } from '@playwright/test';

const TESTS = [
  'sod-axes', 'sod-diagonal', 'strong-shock', 'sedov', 'kelvin-helmholtz', 'low-mach', 'well-balanced', 'advection', 'species', 'advection-accuracy', 'flux-determinism',
  'reflecting-box', 'two-dimensional', 'rayleigh-taylor', 'amr-unrefined', 'amr-shock-tube', 'amr-species', 'amr-sedov', 'amr-instabilities', 'amr-gravity-static', 'amr-gravity-dynamic', 'amr-tidal-disruption', 'amr-time-bins', 'snapshot-roundtrip', 'cooling', 'viscosity', 'conduction', 'point-mass-sink', 'binary-orbit', 'point-mass-encounter', 'gpu-vs-cpu', 'eos', 'sphere-potential', 'jeans', 'polytrope', 'white-dwarf', 'evrard', 'tidal-disruption',
];

test.describe.configure({ mode: 'serial' });

test('the page lists the same tests as this file', async ({ page }) => {
  await page.goto('/validate.html');
  await page.evaluate(() => window.shaderous.ready);
  expect(await page.evaluate(() => window.shaderous.tests)).toEqual(TESTS);
});

for (const id of TESTS) {
  test(id, async ({ page }) => {
    await page.goto('/validate.html');
    await page.evaluate(() => window.shaderous.ready);
    const r = await page.evaluate((t) => window.shaderous.run(t), id);
    console.log(`${id}: ${r.pass ? 'pass' : 'FAIL'} in ${r.seconds.toFixed(1)} s\n` + JSON.stringify(r.metrics, null, 1));
    expect(r.pass, `${r.criterion}\n${JSON.stringify(r.metrics, null, 1)}`).toBe(true);
  });
}

test('the simulator page steps and renders', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/?problem=sedov&n=64');
  await page.waitForFunction(() => (window as any).shaderousApp?.diag?.steps > 20, undefined, { timeout: 60_000 });
  const d = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(d.mass).toBeCloseTo(1, 4);
  expect(d.eTot).toBeCloseTo(1, 2);
  expect(errors).toEqual([]);
});

test('controls rebuild pipelines and switch physics without errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(e.message));
  const steps = () => page.evaluate(() => (window as any).shaderousApp.diag?.steps ?? 0);
  // Wait until the step counter has moved twice, or the run has reached its end time (small
  // grids finish a preset within a second). A reset restarts the counter from zero, so compare
  // for change rather than for growth past the old value.
  const advance = async () => {
    for (let i = 0; i < 2; i++) {
      const before = await steps();
      await page.waitForFunction(
        (b) => {
          const d = (window as any).shaderousApp.diag;
          return d && (d.steps !== b || (d.dt === 0 && d.steps > 0));
        },
        before,
        { timeout: 60_000 },
      );
    }
  };

  await page.goto('/?problem=polytrope&n=32');
  // The Hydro and Gravity sections start collapsed; open them so their controls can be used.
  await page.evaluate(() => document.querySelectorAll('details').forEach((d) => (d.open = true)));
  await page.fill('#tend', '');
  await page.dispatchEvent('#tend', 'change'); // no end time: run until told otherwise
  await advance();
  await page.selectOption('#riemann', 'hll');
  await page.selectOption('#limiter', 'minmod');
  await advance();
  await page.selectOption('#gravity', 'off');
  await advance();
  await page.selectOption('#gravity', 'isolated');
  await page.selectOption('#mode', 'slice');
  await page.selectOption('#field', '1');
  await advance();
  await page.selectOption('#eos', 'gas_radiation');
  await advance();
  await page.selectOption('#problem', 'isothermal-collapse');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'isothermal-collapse');
  await advance();
  await page.selectOption('#res', '64');
  await page.waitForFunction(() => (window as any).shaderousApp.sim?.n === 64);
  await advance();
  const d = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(Number.isFinite(d.eTot)).toBe(true);
  expect(d.mass).toBeCloseTo(1, 3);

  // 2D: only offered where the preset supports it, and switches the grid to one layer.
  expect(await page.locator('#dim option[value="2"]').isDisabled()).toBe(true);
  await page.selectOption('#problem', 'rayleigh-taylor');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'rayleigh-taylor');
  await page.selectOption('#dim', '2');
  await page.waitForFunction(() => (window as any).shaderousApp.sim?.nz === 1);
  await advance();
  const d2 = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(d2.mass).toBeCloseTo(1.5, 3);
  expect(await page.locator('#mode').isDisabled()).toBe(true);

  // Viscosity and conduction: switching them on rebuilds the passes, and the run carries on
  // (onto the adaptive mesh below, too).
  await page.fill('#visc', '0.001');
  await page.dispatchEvent('#visc', 'change');
  await page.fill('#cond', '0.001');
  await page.dispatchEvent('#cond', 'change');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.viscosity === 0.001 && (window as any).shaderousApp.cfg.conductivity === 0.001);
  await advance();
  expect(await page.textContent('#sum-diff')).toBe('viscosity + conduction');
  expect((await page.evaluate(() => (window as any).shaderousApp.diag)).mass).toBeCloseTo(1.5, 3);

  // Adaptive mesh: switches the grid, keeps running, and conserves mass.
  await page.selectOption('#levels', '2');
  await page.waitForFunction(() => (window as any).shaderousApp.sim?.tree !== undefined);
  await advance();
  const d3 = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(d3.mass).toBeCloseTo(1.5, 5);
  expect(await page.evaluate(() => (window as any).shaderousApp.sim.status())).toMatch(/blocks/);
  // Self-gravity runs adaptively too.
  await page.selectOption('#problem', 'polytrope');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'polytrope' && (window as any).shaderousApp.sim);
  await page.selectOption('#res', '64');
  await page.selectOption('#levels', '1');
  await page.waitForFunction(() => (window as any).shaderousApp.sim?.tree !== undefined && (window as any).shaderousApp.cfg.amrLevels === 1, undefined, { timeout: 60_000 });
  await advance();
  const d4 = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(d4.eGrav).toBeLessThan(0);
  expect(Number.isFinite(d4.eTot)).toBe(true);
  expect(errors).toEqual([]);
});

test('a state exported from the page can be imported again', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(e.message));
  const app = <T>(f: string) => page.evaluate(`(() => { const a = window.shaderousApp; return ${f}; })()`) as Promise<T>;

  await page.goto('/?problem=sedov&dim=3&n=32&cfl=0.5');
  await page.waitForFunction(() => (window as any).shaderousApp?.diag?.steps > 10, undefined, { timeout: 60_000 });
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('#export')]);
  expect(download.suggestedFilename()).toMatch(/^shaderous-sedov-3d-32-t.*\.athdf$/);
  const file = await download.path();

  // Move to something else entirely, then load the file: preset, grid, settings and time return.
  await page.selectOption('#problem', 'kelvin-helmholtz');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'kelvin-helmholtz' && (window as any).shaderousApp.sim);
  await page.setInputFiles('#import-file', file);
  await page.waitForFunction(
    () => {
      const a = (window as any).shaderousApp;
      return a.cfg.restart && a.sim && a.diag && a.diag.t === a.cfg.restart.time;
    },
    undefined,
    { timeout: 60_000 },
  );
  expect(await app('[a.cfg.problem.id, a.cfg.ndim, a.cfg.n, a.cfg.cfl]')).toEqual(['sedov', 3, 32, 0.5]);
  expect(await app<number>('a.cfg.restart.time')).toBeGreaterThan(0);
  expect(await app<number>('a.cfg.restart.steps')).toBeGreaterThan(10);
  expect(await page.textContent('#run')).toBe('Run'); // a loaded state waits
  expect(await app<number>('a.diag.mass')).toBeCloseTo(1, 4);
  expect(errors).toEqual([]);
});

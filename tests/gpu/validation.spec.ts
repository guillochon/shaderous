// Runs the in-page validation suite (src/validation/tests.ts) on a real GPU.
import { expect, test } from '@playwright/test';

const TESTS = [
  'sod-axes', 'sod-diagonal', 'strong-shock', 'sedov', 'kelvin-helmholtz', 'advection', 'flux-determinism',
  'reflecting-box', 'two-dimensional', 'rayleigh-taylor', 'amr-unrefined', 'amr-shock-tube', 'amr-sedov', 'amr-instabilities', 'gpu-vs-cpu', 'eos', 'sphere-potential', 'jeans', 'polytrope', 'evrard', 'tidal-disruption',
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

  // Adaptive mesh: switches the grid, keeps running, and conserves mass.
  await page.selectOption('#levels', '2');
  await page.waitForFunction(() => (window as any).shaderousApp.sim?.tree !== undefined);
  await advance();
  const d3 = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(d3.mass).toBeCloseTo(1.5, 5);
  expect(await page.evaluate(() => (window as any).shaderousApp.sim.status())).toMatch(/blocks/);
  // Presets with self-gravity cannot use it.
  await page.selectOption('#problem', 'polytrope');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'polytrope');
  await page.waitForFunction(() => (document.getElementById('levels') as HTMLSelectElement).disabled, undefined, { timeout: 30_000 });
  expect(await page.evaluate(() => (window as any).shaderousApp.cfg.amrLevels)).toBe(0);
  expect(errors).toEqual([]);
});

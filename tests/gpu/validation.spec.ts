// Runs the in-page validation suite (src/validation/tests.ts) on a real GPU.
import { expect, test } from '@playwright/test';

const TESTS = [
  'sod-axes', 'sod-diagonal', 'strong-shock', 'sedov', 'kelvin-helmholtz', 'low-mach', 'well-balanced', 'advection', 'species', 'species-eos', 'shock-bubble', 'detonation', 'advection-accuracy', 'flux-determinism',
  'reflecting-box', 'two-dimensional', 'rayleigh-taylor', 'amr-unrefined', 'amr-shock-tube', 'amr-species', 'amr-species-eos', 'amr-detonation', 'amr-sedov', 'amr-instabilities', 'amr-gravity-static', 'amr-gravity-dynamic', 'amr-tidal-disruption', 'amr-time-bins', 'snapshot-roundtrip', 'cooling', 'viscosity', 'conduction', 'driving', 'point-mass-sink', 'point-mass-many', 'orbit-preview', 'binary-orbit', 'point-mass-encounter', 'gpu-vs-cpu', 'burn-network', 'eos', 'sphere-potential', 'jeans', 'polytrope', 'white-dwarf', 'evrard', 'tidal-disruption',
  'mhd-brio-wu', 'mhd-orszag-tang', 'mhd-field-loop', 'mhd-alfven-wave', 'mhd-blast-3d', 'mhd-slabs', 'mhd-gpu-vs-cpu', 'mhd-restart', 'mhd-replaced', 'mhd-field-brush', 'mhd-amr-unrefined', 'mhd-amr-refined', 'mhd-amr-chunks', 'mhd-amr-moving', 'mhd-amr-bins', 'mhd-gravity', 'mhd-gravity-blocks', 'mhd-star-fields', 'mhd-species', 'mhd-diffusion', 'mhd-resistivity', 'mhd-line-tied', 'moving-wall', 'mhd-tidal-disruption', 'mhd-disk',
  'rot-off', 'rot-inertial', 'rot-balance', 'rot-blob', 'rot-binary', 'rot-mhd',
  'sym-flux', 'sym-sedov', 'sym-half', 'sym-sphere', 'sym-ellipsoid', 'sym-polytrope', 'sym-evrard', 'sym-mhd', 'sym-mass',
  'cosmo-a', 'cosmo-drift', 'cosmo-cool', 'cosmo-eos', 'cosmo-wave', 'cosmo-growth', 'cosmo-pancake',
  'shear-steady', 'shear-mass', 'shear-wave', 'shear-flux', 'shear-mri', 'shear-epicycle',
  'cyl-rest', 'cyl-conserve', 'cyl-sedov', 'cyl-sink', 'cyl-blocks', 'cyl-sphere', 'cyl-evrard', 'cyl-sphere-blocks', 'cyl-evrard-blocks', 'cyl-noh', 'cyl-bondi', 'cyl-torus', 'cyl-mhd', 'cyl-column', 'cyl-star', 'cyl-swirl',
  'sph-rest', 'sph-conserve', 'sph-sedov', 'sph-polytrope', 'sph-bondi', 'sph-blocks', 'sph-polytrope-blocks', 'sph-swirl',
];

// One worker (playwright.config.ts), in order; a failure does not skip the tests after it.
test.describe.configure({ mode: 'default' });

test('the page lists the same tests as this file', async ({ page }) => {
  await page.goto('/validate.html');
  await page.evaluate(() => window.shaderous.ready);
  expect(await page.evaluate(() => window.shaderous.tests)).toEqual(TESTS);
});

for (const id of TESTS) {
  test(id, async ({ page }) => {
    // What a test logs (details of a failure) goes to the report.
    page.on('console', (m) => m.type() === 'log' && console.log(m.text()));
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

  // A setting of the Setup dialog, set as a script sets it (ui/api.ts control): the page does
  // what the dialog's Apply does. A script's change leaves the run held, so play it again.
  const set = async (id: string, value: string | number | boolean) => {
    await page.evaluate(([i, v]) => (window as any).shaderous.control(i, v), [id, value] as const);
    await page.evaluate(() => (window as any).shaderous.play());
  };
  const cfgNow = () => page.evaluate(() => {
    const c = (window as any).shaderousApp.cfg;
    return { id: c.problem.id, ndim: c.ndim, n: c.n, amr: c.amrLevels, riemann: c.riemann, limiter: c.limiter, gravity: c.gravity, eos: c.eos, viscosity: c.viscosity, conductivity: c.conductivity };
  });

  await page.goto('/?problem=polytrope&n=32');
  await page.evaluate(() => (window as any).shaderous.ready);
  await page.evaluate(() => document.querySelectorAll('details').forEach((d) => (d.open = true)));
  await page.fill('#tend', '');
  await page.dispatchEvent('#tend', 'change'); // no end time: run until told otherwise
  await advance();
  await set('riemann', 'hll');
  await set('limiter', 'minmod');
  expect(await cfgNow()).toMatchObject({ riemann: 'hll', limiter: 'minmod' });
  await advance();
  await set('gravity', 'off');
  await advance();
  await set('gravity', 'isolated');
  await page.selectOption('#mode', 'slice');
  await page.selectOption('#field', '1');
  await advance();
  await set('eos', 'gas_radiation');
  expect((await cfgNow()).eos).toBe('gas_radiation');
  await advance();
  await page.selectOption('#problem', 'isothermal-collapse');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'isothermal-collapse');
  await advance();
  await set('res', 64);
  await page.waitForFunction(() => (window as any).shaderousApp.sim?.n === 64);
  await advance();
  const d = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(Number.isFinite(d.eTot)).toBe(true);
  expect(d.mass).toBeCloseTo(1, 3);

  // 2D: only where the setup supports it, and switches the grid to one layer.
  const dims = await page.evaluate(() => (window as any).shaderous.controls().then((c: any[]) => c.find((x) => x.id === 'dim')));
  expect(dims.options.map((o: any) => o.value)).toEqual([3, 2]);
  await page.selectOption('#problem', 'rayleigh-taylor');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'rayleigh-taylor');
  await set('dim', 2);
  await page.waitForFunction(() => (window as any).shaderousApp.sim?.nz === 1);
  await advance();
  const d2 = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(d2.mass).toBeCloseTo(1.5, 3);
  expect(await page.locator('#mode').isDisabled()).toBe(true);

  // Viscosity and conduction: switching them on rebuilds the passes, and the run carries on
  // (onto the adaptive mesh below, too).
  await set('visc', 0.001);
  await set('cond', 0.001);
  expect(await cfgNow()).toMatchObject({ viscosity: 0.001, conductivity: 0.001 });
  await advance();
  expect((await page.evaluate(() => (window as any).shaderousApp.diag)).mass).toBeCloseTo(1.5, 3);

  // Adaptive mesh: switches the grid, keeps the physics chosen, keeps running, and conserves mass.
  await set('levels', 2);
  await page.waitForFunction(() => (window as any).shaderousApp.sim?.tree !== undefined);
  expect(await cfgNow()).toMatchObject({ viscosity: 0.001, amr: 2 });
  await advance();
  const d3 = await page.evaluate(() => (window as any).shaderousApp.diag);
  expect(d3.mass).toBeCloseTo(1.5, 5);
  expect(await page.evaluate(() => (window as any).shaderousApp.sim.status())).toMatch(/blocks/);
  // Self-gravity runs adaptively too.
  await page.selectOption('#problem', 'polytrope');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'polytrope' && (window as any).shaderousApp.sim);
  await set('res', 64);
  await set('levels', 1);
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

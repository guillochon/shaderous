// The Setup dialog and the side bar that goes with it, on a real GPU: every setting of the
// simulation is in the dialog, the bar keeps the preset menus and a setup's featured fields,
// and both edit one document.
import { expect, test, type Page } from './fixtures';

// Node's own modules, without its type package (the project does not install it).
declare const Buffer: { from(text: string): never };
const nodeFs = 'node:fs';
const readText = async (path: string): Promise<string> => ((await import(nodeFs)) as { readFileSync(p: string, enc: string): string }).readFileSync(path, 'utf8');

const app = (page: Page) => ({
  cfg: () =>
    page.evaluate(() => {
      const c = (window as any).shaderousApp.cfg;
      return { id: c.problem.id as string, label: c.problem.label as string, n: c.n as number, ndim: c.ndim as number, viscosity: c.viscosity as number, riemann: c.riemann as string, gravity: c.gravity as string, params: c.params as Record<string, number> };
    }),
  time: () => page.evaluate(() => (window as any).shaderous.state().then((s: any) => s.t as number)),
  run: (steps: number) => page.evaluate((n) => (window as any).shaderous.run({ steps: n }), steps),
  idle: () => page.evaluate(() => (window as any).shaderous.state()),
});

async function open(page: Page, query: string) {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(e.message));
  // Headless: the run steps only when the test asks, so times can be compared.
  await page.goto(`/?${query}&headless=1`);
  await page.evaluate(() => (window as any).shaderous.ready);
  return errors;
}

const dialog = (page: Page) => page.locator('dialog[open]');
const field = (page: Page, label: string) => dialog(page).getByLabel(label, { exact: true }).first();
const tab = (page: Page, name: string) => dialog(page).locator('.sd-tabs').getByRole('tab', { name, exact: true });
const applyButton = (page: Page) => dialog(page).locator('.sd-apply');
/** Unfold a group of the dialog's preset list (all but the running preset's start folded). */
const unfold = (page: Page, group: string) => dialog(page).locator('.sd-group', { has: page.locator('summary', { hasText: group }) }).evaluate((d) => ((d as HTMLDetailsElement).open = true));

test('the side bar holds no simulation setting but the preset menus and the featured fields', async ({ page }) => {
  const errors = await open(page, 'problem=binary&dim=2&n=128');
  // What the Setup dialog holds is not in the bar: the grid, the gas, gravity, the boundaries, the solver.
  for (const id of ['dim', 'res', 'levels', 'eos', 'gamma', 'riemann', 'limiter', 'cfl', 'cooling', 'visc', 'cond', 'gravity', 'G', 'bc-lo-0', 'refhi', 'tbins', 'fluffrho', 'dampr']) {
    await expect(page.locator(`#side #${id}`), id).toHaveCount(0);
  }
  await expect(page.locator('#side #problem')).toBeVisible();
  await expect(page.locator('#side #variant')).toBeAttached();
  await expect(page.locator('#side #setup')).toBeVisible();
  await expect(page.locator('#setup-sum')).toHaveText(/128².*Barotropic/);
  // The binary's featured fields, as its parameters were.
  for (const id of ['pp-q', 'pp-ecc', 'pp-soft']) await expect(page.locator(`#pparams #${id}`)).toBeVisible();
  // Every setting is still there for a script, served from the option table.
  const ids: string[] = await page.evaluate(() => (window as any).shaderous.controls().then((c: any[]) => c.map((x) => x.id)));
  for (const id of ['problem', 'run', 'tend', 'gamma', 'riemann', 'bc', 'n', 'nu', 'parts.0.soft', 'parts.3.damp.rate']) expect(ids, id).toContain(id);
  expect(errors).toEqual([]);
});

test('a preset and a variant picked in the bar load', async ({ page }) => {
  const errors = await open(page, 'problem=sedov&dim=2&n=128');
  await page.selectOption('#problem', 'sod');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'sod-x');
  await page.selectOption('#variant', 'sod-y');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'sod-y' && (window as any).shaderousApp.sim);
  await app(page).idle();
  expect((await app(page).cfg()).n).toBe(128);
  await app(page).run(5);
  expect(await app(page).time()).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test('a featured field changed in the bar shows in the dialog, and one changed in the dialog shows in the bar', async ({ page }) => {
  const errors = await open(page, 'problem=binary&dim=2&n=128');
  await page.fill('#pp-ecc', '0.3');
  await page.dispatchEvent('#pp-ecc', 'change');
  await app(page).idle();
  expect((await app(page).cfg()).params.ecc).toBe(0.3);
  await page.click('#setup');
  await expect(dialog(page)).toBeVisible();
  // The strip at the top of the dialog, and the orbit's own field on the Objects page: one value.
  await expect(dialog(page).locator('.sd-strip').getByLabel('Eccentricity', { exact: true })).toHaveValue('0.3');
  await expect(applyButton(page)).toBeDisabled();
  await dialog(page).locator('.sd-strip').getByLabel('Mass ratio', { exact: true }).fill('0.5');
  await dialog(page).locator('.sd-strip').getByLabel('Mass ratio', { exact: true }).press('Tab');
  await expect(applyButton(page)).toHaveText('Apply and restart');
  await applyButton(page).click();
  await expect(applyButton(page)).toBeDisabled({ timeout: 60_000 });
  const c = await app(page).cfg();
  // Still the binary: its own parameters changed, not its parts.
  expect([c.id, c.params.q, c.params.ecc]).toEqual(['binary', 0.5, 0.3]);
  await expect(page.locator('#pp-q')).toHaveValue('0.5');
  expect(errors).toEqual([]);
});

test('the dialog greys out what cannot be combined and says why', async ({ page }) => {
  const errors = await open(page, 'problem=kelvin-helmholtz&dim=2&n=128');
  await page.click('#setup');
  // Pick the polytrope in the dialog's preset list: a star, with isolated self-gravity.
  await unfold(page, 'Self-gravity');
  await dialog(page).locator('.sd-entry', { hasText: 'Polytrope (hydrostatic star): Standard' }).click();
  await tab(page, 'Gravity').click();
  const periodic = field(page, 'Gravity').locator('option', { hasText: 'Self, periodic' });
  await expect(periodic).toBeDisabled();
  expect(await periodic.getAttribute('title')).toMatch(/Needs Every face periodic.*star/);
  await tab(page, 'Domain and grid').click();
  const twoD = field(page, 'Dimensions').locator('option', { hasText: '2D' });
  await expect(twoD).toBeDisabled();
  expect(await twoD.getAttribute('title')).toMatch(/Self-gravity/);
  // A barotropic gas cannot hold a polytropic star, nor cool.
  await tab(page, 'Gas').click();
  await expect(field(page, 'Equation of state').locator('option[value="barotropic"]')).toBeDisabled();
  // Nothing was applied: the run is still the shear layer.
  expect((await app(page).cfg()).id).toBe('kelvin-helmholtz');
  expect(errors).toEqual([]);
});

test('a live option carries the run on, a compiled one rebuilds it, a part starts it again', async ({ page }) => {
  const errors = await open(page, 'problem=binary&dim=2&n=128');
  await app(page).run(10);
  const t0 = await app(page).time();
  expect(t0).toBeGreaterThan(0);
  await page.click('#setup');

  // Live: applied as it is typed, without Apply, and the clock keeps its time.
  await tab(page, 'Gas').click();
  await field(page, 'Viscosity ν').fill('0.004');
  await field(page, 'Viscosity ν').press('Tab');
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.viscosity === 0.004);
  await expect(applyButton(page)).toBeDisabled();
  expect(await app(page).time()).toBe(t0);

  // Compiled in: waits for Apply, which says what it will do; the clock still keeps its time.
  await tab(page, 'Solver').click();
  await field(page, 'Riemann solver').selectOption('hll');
  await expect(applyButton(page)).toHaveText('Apply and recompile');
  expect((await app(page).cfg()).riemann).toBe('hllc');
  await applyButton(page).click();
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.riemann === 'hll');
  await expect(applyButton(page)).toBeDisabled({ timeout: 60_000 });
  expect(await app(page).time()).toBe(t0);

  // A new part: a third point mass. The run starts again, as a setup of its own.
  await tab(page, 'Objects').click();
  await dialog(page).locator('.sd-add summary').click();
  await dialog(page).locator('.sd-add').getByRole('button', { name: /Point mass/ }).click();
  await expect(applyButton(page)).toHaveText('Apply and restart');
  await applyButton(page).click();
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'custom' && (window as any).shaderousApp.sim);
  await expect(applyButton(page)).toBeDisabled({ timeout: 60_000 });
  const c = await app(page).cfg();
  expect([c.label, c.viscosity, c.riemann]).toEqual(['Binary accretion (modified)', 0.004, 'hll']);
  expect(await app(page).time()).toBe(0);
  expect(await page.evaluate(() => (window as any).shaderousApp.sim.massCount)).toBe(3);
  // The bar's menu shows it among the user's own setups.
  await dialog(page).locator('.sd-close, [aria-label="Close"]').first().click();
  await expect(page.locator('#problem')).toHaveValue('custom');
  await app(page).run(5);
  expect(await app(page).time()).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test('a saved file opens again as the same setup, with the same link', async ({ page }) => {
  const errors = await open(page, 'problem=binary&dim=2&n=128');
  await page.click('#setup');
  // Change the disk, so the setup is no preset.
  await tab(page, 'Objects').click();
  const disk = dialog(page).locator('.sd-part', { hasText: 'disk' }).last();
  if (await disk.locator('.sd-part-body').isHidden()) await disk.locator('.sd-part-head button').first().click();
  await disk.getByLabel('Inner radius', { exact: true }).fill('3');
  await disk.getByLabel('Inner radius', { exact: true }).press('Tab');
  await applyButton(page).click();
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'custom' && (window as any).shaderousApp.sim);
  await expect(applyButton(page)).toBeDisabled({ timeout: 60_000 });
  const [download] = await Promise.all([page.waitForEvent('download'), dialog(page).getByRole('button', { name: 'Save file' }).click()]);
  expect(download.suggestedFilename()).toMatch(/\.shaderous\.json$/);
  const saved = await readText(await download.path());
  const doc = JSON.parse(saved);
  expect(doc.format).toBe('shaderous-setup');
  expect(doc.parts.find((p: any) => p.kind === 'disk').rIn).toBe(3);
  const setup1 = await page.evaluate(() => (window as any).shaderous.state().then((s: any) => s.setup));
  const query1: string = await page.evaluate(() => (window as any).shaderous.state().then((s: any) => s.query));
  expect(query1).toMatch(/problem=custom.*setup=/);

  // A fresh page, the file opened in the dialog and applied: the same document runs.
  await page.goto('/?problem=sedov&dim=2&n=128&headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  await page.click('#setup');
  await dialog(page).locator('input[type="file"]').setInputFiles({ name: 'mine.shaderous.json', mimeType: 'application/json', buffer: Buffer.from(saved) });
  await expect(applyButton(page)).toHaveText('Apply and restart');
  await applyButton(page).click();
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'custom' && (window as any).shaderousApp.sim);
  await expect(applyButton(page)).toBeDisabled({ timeout: 60_000 });
  const setup2 = await page.evaluate(() => (window as any).shaderous.state().then((s: any) => s.setup));
  expect(setup2).toEqual(setup1);

  // And the link alone opens it too.
  await page.waitForFunction(() => (window as any).shaderous.state().then((s: any) => /setup=/.test(s.query)));
  const query2: string = await page.evaluate(() => (window as any).shaderous.state().then((s: any) => s.query));
  expect(new URLSearchParams(query2).get('setup')).toBe(new URLSearchParams(query1).get('setup'));
  await page.goto(`/?${query2}&headless=1`);
  await page.evaluate(() => (window as any).shaderous.ready);
  const setup3 = await page.evaluate(() => (window as any).shaderous.state().then((s: any) => s.setup));
  expect(setup3).toEqual(setup1);
  await app(page).run(3);
  expect(errors).toEqual([]);
});

test('a preset downloads as its whole document', async ({ page }) => {
  await open(page, 'problem=sedov&dim=2&n=128');
  await page.click('#setup');
  const [download] = await Promise.all([page.waitForEvent('download'), dialog(page).getByRole('button', { name: 'Download the preset Sedov-Taylor blast: Radiation dominated' }).click()]);
  expect(download.suggestedFilename()).toBe('sedov-rad.shaderous.json');
  const doc = JSON.parse(await readText(await download.path()));
  // A variant's file as the repository has it, with its parent's content in it: whole, to run alone.
  const own = JSON.parse(await readText('src/presets/sedov-rad.json'));
  const parent = JSON.parse(await readText('src/presets/sedov.json'));
  expect(doc.extends).toBeUndefined();
  expect(doc.physics.eos).toBe(own.physics.eos);
  expect(doc.parts).toEqual(parent.parts);
  expect(doc.id).toBe('sedov-rad');
});

test('a script loads a setup document, and a state file carries it', async ({ page }) => {
  const errors = await open(page, 'problem=sedov&dim=2&n=128');
  // Three point masses in still gas: no preset has this.
  const doc = {
    format: 'shaderous-setup', version: 1, name: 'Three bodies',
    domain: { ndim: 2, n: 128, box: 4, supports2d: true },
    physics: { eos: 'ideal', eosParams: { gamma: 1.4 }, G: 1 },
    background: { rho: 1e-3, p: 1e-4 },
    parts: [
      { kind: 'mass', id: 'a', gm: 1, pos: [0, 0, 0], soft: 0.1 },
      { kind: 'mass', id: 'b', gm: 0.01, pos: [1, 0, 0], vel: [0, 1, 0], soft: 0.1 },
      { kind: 'mass', id: 'c', gm: 0.01, pos: [-1.5, 0, 0], vel: [0, -0.8, 0], soft: 0.1 },
      { kind: 'region', shape: { type: 'sphere', center: [0, 1, 0], radius: 0.3 }, rho: 0.01, s: 1 },
    ],
  };
  const st = await page.evaluate((d) => (window as any).shaderous.load(d), doc);
  expect([st.problem, st.label, st.ndim, st.n]).toEqual(['custom', 'Three bodies', 2, 128]);
  expect(await page.evaluate(() => (window as any).shaderousApp.sim.massCount)).toBe(3);
  await app(page).run(20);
  const d = await page.evaluate(() => (window as any).shaderous.diagnostics());
  expect(d.masses).toHaveLength(3);
  expect(Number.isFinite(d.mass) && d.mass > 0).toBe(true);
  // A wrong document is refused with the reason.
  await expect(page.evaluate(() => (window as any).shaderous.load({ format: 'shaderous-setup', version: 1, parts: [{ kind: 'mass', gm: 'heavy' }] }))).rejects.toThrow(/parts\.0\.gm/);

  // Export, then import on a page that has no such setup: it restarts as itself, at the time
  // and step it was saved at, and goes on as the run it was saved from does.
  const file = await page.evaluate(() => (window as any).shaderous.athdf());
  const saved = await app(page).idle();
  expect(file.time).toBe(saved.t);
  const outcome = () =>
    page.evaluate(async () => {
      const d = await (window as any).shaderous.diagnostics();
      return { t: d.t as number, steps: d.steps as number, mass: d.mass as number, eTot: d.eTot as number, bodies: d.masses.flatMap((m: any) => [...m.pos, ...m.vel]) as number[] };
    });
  await app(page).run(10);
  const straight = await outcome();
  await page.goto('/?problem=sedov&dim=2&n=128&headless=1');
  await page.evaluate(() => (window as any).shaderous.ready);
  const back = await page.evaluate((b) => (window as any).shaderous.loadAthdf(b), file.base64);
  expect([back.problem, back.label, back.setup.parts.length]).toEqual(['custom', 'Three bodies', 4]);
  // The file keeps the time in float32, as the clock has it.
  expect(back.steps).toBe(saved.steps);
  expect(Math.abs(back.t - saved.t)).toBeLessThan(1e-7 * saved.t);
  expect(back.warnings).toEqual([]);
  expect((await app(page).run(10)).taken).toBe(10);
  const resumed = await outcome();
  // The file holds pressure, not internal energy, and the masses in float32: the same run to
  // rounding (in practice the gas comes back bit for bit, the masses to 1e-8).
  expect(resumed.steps).toBe(straight.steps);
  expect(Math.abs(resumed.t - straight.t)).toBeLessThan(1e-7 * straight.t);
  expect(Math.abs(resumed.mass - straight.mass)).toBeLessThan(1e-6 * straight.mass);
  expect(Math.abs(resumed.eTot - straight.eTot)).toBeLessThan(1e-6 * Math.abs(straight.eTot));
  resumed.bodies.forEach((x, i) => expect(Math.abs(x - straight.bodies[i]), `point mass value ${i}`).toBeLessThan(1e-6));
  expect(errors).toEqual([]);
});

test('three stars of a setup of its own settle and hold for a dynamical time', async ({ page }) => {
  const errors = await open(page, 'problem=sedov&dim=2&n=128');
  const star = (id: string, pos: number[], vel: number[]) => ({ kind: 'star', id, type: 'polytrope', n: 1.5, mass: 1, radius: 0.5, pos, vel });
  // Each star 21 cells across: at 4 cells of radius they puff up, and lose 15% of their
  // binding energy in a dynamical time.
  const doc = {
    format: 'shaderous-setup', version: 1, name: 'Triple',
    domain: { ndim: 3, n: 128, amr: 0, box: 6, supports2d: false, bc: { lo: [3, 3, 3], hi: [3, 3, 3] } },
    physics: { eos: 'ideal', eosParams: { gamma: 5 / 3 }, gravity: 'isolated', G: 1, floors: { rho: 1e-10, p: 1e-14 } },
    background: { rho: 1e-6, cut: 1e-4, relativeTo: 'a' },
    parts: [star('a', [-1.5, 0, 0], [0, -0.3, 0]), star('b', [1.5, 0, 0], [0, 0.3, 0]), star('c', [0, 2.2, 0], [0.3, 0, 0])],
  };
  const st = await page.evaluate((d) => (window as any).shaderous.load(d), doc);
  expect([st.problem, st.ndim, st.n]).toEqual(['custom', 3, 128]);
  const diag = () => page.evaluate(() => (window as any).shaderous.diagnostics());
  const d0 = await diag();
  // Three stars of unit mass, less what the grid's cut leaves out.
  expect(d0.mass).toBeGreaterThan(2.5);
  expect(d0.mass).toBeLessThan(3.3);
  // One dynamical time, sqrt(R^3 / G M), of each star.
  const tDyn = Math.sqrt(0.5 ** 3);
  await page.evaluate((t) => (window as any).shaderous.run({ until: t, timeoutMs: 300_000 }), tDyn);
  const d1 = await diag();
  expect(d1.t).toBeCloseTo(tDyn, 6);
  const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(a);
  // Settled stars hold: their peak density and binding energy barely move (1.4% and 0.3%).
  expect(rel(d0.ranges.Density[1], d1.ranges.Density[1])).toBeLessThan(0.05);
  expect(rel(d0.eGrav, d1.eGrav)).toBeLessThan(0.02);
  expect(rel(d0.eTot, d1.eTot)).toBeLessThan(0.01);
  expect(rel(d0.mass, d1.mass)).toBeLessThan(1e-5);
  expect(errors).toEqual([]);
});

test('a binary held on its orbit keeps it, and lets go of it mid-run when asked', async ({ page }) => {
  const errors = await open(page, 'problem=binary&dim=2&n=256&p.hold=1&p.ecc=0.3');
  const masses = () => page.evaluate(() => (window as any).shaderous.diagnostics().then((d: any) => d.masses));
  // The orbit's energy and angular momentum per unit reduced mass, from the two masses alone.
  const orbit = (m: any[]) => {
    const [d, v] = [[0, 1, 2].map((a) => m[1].pos[a] - m[0].pos[a]), [0, 1, 2].map((a) => m[1].vel[a] - m[0].vel[a])];
    return { gm: [m[0].gm, m[1].gm], l: d[0] * v[1] - d[1] * v[0], accreted: m[0].accreted + m[1].accreted };
  };
  const m0 = orbit(await masses());
  await page.evaluate(() => (window as any).shaderous.run({ until: 3 }));
  const m1 = orbit(await masses());
  // The sinks took gas, and it changed neither the masses nor the orbit's angular momentum.
  expect(m1.accreted).toBeGreaterThan(0);
  expect(m1.gm).toEqual(m0.gm);
  expect(Math.abs(m1.l / m0.l - 1)).toBeLessThan(2e-5);

  // Let go, as the bar's checkbox does: the run carries on, and the masses now grow by what they take.
  const t = await app(page).time();
  await page.uncheck('#pp-hold');
  await app(page).idle();
  expect(await app(page).time()).toBe(t);
  await page.evaluate(() => (window as any).shaderous.run({ until: 4 }));
  const m2 = orbit(await masses());
  expect(m2.gm[0] + m2.gm[1]).toBeGreaterThan(m1.gm[0] + m1.gm[1]);
  expect(errors).toEqual([]);
});

test('a disk with a density of its own, from formulas, about a star', async ({ page }) => {
  const errors = await open(page, 'problem=sedov&dim=2&n=128');
  const doc = {
    format: 'shaderous-setup', version: 1, name: 'Star with a ring',
    domain: { ndim: 3, n: 64, amr: 0, box: 8, supports2d: false, bc: { lo: [3, 3, 3], hi: [3, 3, 3] } },
    physics: { eos: 'ideal', eosParams: { gamma: 5 / 3 }, gravity: 'isolated', G: 1, floors: { rho: 1e-10, p: 1e-14 } },
    background: { rho: 1e-7, cut: 1e-4, relativeTo: 'star' },
    parts: [
      { kind: 'star', id: 'star', type: 'polytrope', n: 1.5, mass: 1, radius: 0.6 },
      { kind: 'disk', id: 'ring', about: ['star'], profile: 'balanced', rIn: 1.5, rOut: 3, sigma0: 1e-3, cs: 0.15, floor: 0,
        radialFn: 'sigma0 * exp(-((r - 2.2) / 0.4)^2)', verticalFn: '1 / cosh(z)^2', damp: { from: 3.5, rate: 1 } },
    ],
  };
  const st = await page.evaluate((d) => (window as any).shaderous.load(d), doc);
  expect([st.problem, st.label]).toEqual(['custom', 'Star with a ring']);
  // The star settles with the ring left around it, and the run steps with the generated damping zone.
  await page.evaluate(() => (window as any).shaderous.run({ steps: 10, timeoutMs: 240_000 }));
  const d = await page.evaluate(() => (window as any).shaderous.diagnostics());
  expect(d.mass).toBeGreaterThan(0.9);
  expect(d.mass).toBeLessThan(1.3);
  // Gas at the ring's radius in the midplane, where the star's ambient medium alone would have next to none.
  const ring = await page.evaluate(() => (window as any).shaderous.probe([[0.5 + 2.2 / 8, 0.5, 0.5], [0.5, 0.5 + 2.2 / 8, 0.5], [0.5 + 3.6 / 8, 0.5, 0.5]]));
  const rho = ring.points.map((q: any) => q.rho);
  expect(rho[0]).toBeGreaterThan(3e-4);
  expect(rho[1]).toBeGreaterThan(3e-4);
  expect(rho[2]).toBeLessThan(1e-5);
  // A formula that is not one is refused before anything runs.
  await expect(page.evaluate((d) => (window as any).shaderous.load({ ...d, parts: [d.parts[0], { ...d.parts[1], radialFn: 'sigma0 * nope(r)' }] }), doc)).rejects.toThrow(/"nope" is not a function/);
  expect(errors).toEqual([]);
});

test('a preset whose name does not fit shows it whole at once on hover', async ({ page }) => {
  await open(page, 'problem=sedov&dim=2&n=128');
  await page.click('#setup');
  await unfold(page, 'Point masses');
  await unfold(page, 'Magnetic fields');
  const long = dialog(page).locator('.sd-entry', { hasText: 'Tidal disruption: Partial, random field' });
  const short = dialog(page).locator('.sd-entry', { hasText: /^MHD rotor$/ });
  // Every entry's tooltip says its name, then what pressing it does.
  expect(await short.getAttribute('title')).toBe('MHD rotor\nLoad this preset into the editor (the run is untouched until Apply)');
  expect(await long.evaluate((e) => e.scrollWidth > e.clientWidth)).toBe(true);
  await long.hover();
  // No wait: the dialog's own tip, in place of the browser's.
  await expect(dialog(page).locator('.sd-tip strong')).toHaveText('Tidal disruption: Partial, random field', { timeout: 200 });
  expect(await long.getAttribute('title')).toBeNull();
  // One that fits keeps to the browser's tooltip.
  await short.hover();
  await expect(dialog(page).locator('.sd-tip')).toHaveCount(0);
  expect(await long.getAttribute('title')).toContain('Tidal disruption: Partial, random field');
});

test('the preset list opens folded, but for the group of the preset that runs', async ({ page }) => {
  await open(page, 'problem=binary&dim=2&n=128');
  await page.click('#setup');
  const groups = () => dialog(page).locator('.sd-group').evaluateAll((ds) => ds.map((d) => [d.querySelector('summary')!.textContent, (d as HTMLDetailsElement).open]));
  expect(await groups()).toEqual([['Instabilities', false], ['Shocks and blasts', false], ['Self-gravity', false], ['Point masses', true], ['Magnetic fields', false], ['Tests and sandbox', false]]);
  await expect(dialog(page).locator('.sd-entry.sd-current')).toHaveText('Binary accretion');
  await expect(dialog(page).locator('.sd-entry', { hasText: 'Jeans instability' })).toBeHidden();
  // A group unfolds on a click of its name, and its presets load as before.
  await dialog(page).locator('.sd-group summary', { hasText: 'Instabilities' }).click();
  await dialog(page).locator('.sd-entry', { hasText: 'Jeans instability' }).click();
  await expect(applyButton(page)).toHaveText('Apply and restart');
  // Opened again on another run, it folds to that run's group.
  await applyButton(page).click();
  await page.waitForFunction(() => (window as any).shaderousApp.cfg.problem.id === 'jeans' && (window as any).shaderousApp.sim);
  await dialog(page).locator('.sd-close, [aria-label="Close"]').first().click();
  await page.click('#setup');
  expect((await groups()).filter((g) => g[1]).map((g) => g[0])).toEqual(['Instabilities']);
});

test('a script checks a setup document before running it', async ({ page }) => {
  await open(page, 'problem=sedov&dim=2&n=128');
  const check = (d: unknown) => page.evaluate((x) => (window as any).shaderous.check(x), d);
  // The shipped example: nothing wrong with it.
  const example = JSON.parse(await readText('examples/setups/ms-wd-collision.shaderous.json'));
  const good = await check(example);
  expect([good.ok, good.error, good.notes, good.issues.filter((i: any) => i.level === 'error')]).toEqual([true, undefined, [], []]);
  expect(good.document.parts).toHaveLength(3);
  // A star without the self-gravity that holds it, a softening no cell can see, an unknown key.
  const bad = await check({ ...example, extra: 1, physics: { ...example.physics, gravity: 'off' }, parts: [example.parts[0], { ...example.parts[1], soft: 1e-4 }, example.parts[2]] });
  expect(bad.ok).toBe(false);
  expect(bad.notes.join(' ')).toMatch(/extra: dropped/);
  const text = bad.issues.map((i: any) => `${i.level}: ${i.text}`).join(' | ');
  expect(text).toMatch(/error: Star: needs isolated self-gravity/i);
  expect(text).toMatch(/warning: .*softening.*under two cells/);
  // One that cannot be read says where.
  const broken = await check({ ...example, parts: [{ kind: 'mass', gm: 'heavy' }] });
  expect([broken.ok, broken.issues]).toEqual([false, []]);
  expect(broken.error).toMatch(/parts\.0\.gm/);
  // Nothing ran: the page still holds its own run.
  expect((await app(page).cfg()).id).toBe('sedov');
});

test('the table of the API names its methods, and the headless tools read the same table', async ({ page }) => {
  await open(page, 'problem=sedov&dim=2&n=128');
  const { keys, methods } = await page.evaluate(async () => {
    const api = (window as any).shaderous;
    return { keys: Object.keys(api).filter((k) => typeof api[k] === 'function'), methods: (await api.describe()).methods };
  });
  expect([...keys].sort()).toEqual(methods.map((m: any) => m.name).sort());
  // The committed schema the MCP server reads is this page's (else: npm run gen).
  const schema = JSON.parse(await readText('tools/headless/schema.json'));
  expect(schema.api).toEqual(methods);
});

import { initGpu, type Gpu } from '../gpu/device';
import { runTest, TESTS, type TestResult } from './tests';
import { configFor, Sim } from '../sim';
import { AmrSim } from '../amr/amr_sim';
import { PROBLEMS, problemById } from '../problems';

declare global {
  interface Window {
    shaderous: {
      ready: Promise<void>;
      tests: string[];
      run(id: string): Promise<TestResult>;
      runAll(): Promise<TestResult[]>;
      /** Handles for poking at the simulator from the console. */
      debug: Record<string, unknown>;
    };
  }
}

const fmt = (v: number | string) =>
  typeof v === 'string' ? v : Number.isInteger(v) ? String(v) : Math.abs(v) >= 0.01 && Math.abs(v) < 1e4 ? v.toPrecision(4) : v.toExponential(2);

let gpu: Gpu;
const rows = new Map<string, HTMLTableRowElement>();
const tbody = document.querySelector('#results tbody')!;
/** The last result of each test run on this page. */
const results = new Map<string, TestResult>();
/** The build stamp vite.config.ts defines (as ui/errors.ts reads it; not imported, for its page-wide error handlers). */
declare const __BUILD__: string | undefined;
const NL = String.fromCharCode(10);
const BUILD_STAMP = typeof __BUILD__ === 'string' ? __BUILD__ : 'dev';
const copyFailing = document.getElementById('copy-failing') as HTMLButtonElement;

/**
 * The table's groups, by subject; within each, tests keep the order of TESTS. A test in none of
 * them goes to Other, so a new test is never left off the page.
 */
const GROUPS: { name: string; ids: string[] }[] = [
  { name: 'Hydrodynamics', ids: ['sod-axes', 'sod-diagonal', 'strong-shock', 'sedov', 'kelvin-helmholtz', 'rayleigh-taylor', 'low-mach', 'well-balanced', 'advection', 'advection-accuracy', 'flux-determinism', 'reflecting-box', 'two-dimensional'] },
  { name: 'Species, equations of state and burning', ids: ['species', 'species-eos', 'shock-bubble', 'detonation', 'burn-network', 'eos'] },
  { name: 'Cooling and diffusion', ids: ['cooling', 'viscosity', 'conduction'] },
  { name: 'Gravity and point masses', ids: ['sphere-potential', 'jeans', 'polytrope', 'white-dwarf', 'evrard', 'tidal-disruption', 'point-mass-sink', 'orbit-preview', 'binary-orbit', 'point-mass-encounter'] },
  { name: 'Adaptive mesh', ids: TESTS.map((t) => t.id).filter((id) => id.startsWith('amr-')) },
  { name: 'Magnetic fields', ids: TESTS.map((t) => t.id).filter((id) => id.startsWith('mhd-')) },
  { name: 'Against the reference, and files', ids: ['gpu-vs-cpu', 'snapshot-roundtrip'] },
];
{
  const placed = new Set(GROUPS.flatMap((g) => g.ids));
  const rest = TESTS.map((t) => t.id).filter((id) => !placed.has(id));
  if (rest.length) GROUPS.push({ name: 'Other', ids: rest });
  // Each group in the order of TESTS, and only the tests that exist.
  for (const g of GROUPS) g.ids = TESTS.map((t) => t.id).filter((id) => g.ids.includes(id));
}
/** Each group's header: its tally of results. */
const tallies: { ids: string[]; el: HTMLElement }[] = [];

for (const g of GROUPS) {
  if (!g.ids.length) continue;
  const head = document.createElement('tr');
  head.className = 'group';
  head.innerHTML = `<th colspan="5"><span class="group-name">${g.name}</span> <span class="group-count">${g.ids.length} tests</span> <button class="run-group" title="Run the tests of this group, in order">Run group</button> <span class="group-tally"></span></th>`;
  head.querySelector('.run-group')!.addEventListener('click', () => runMany(g.ids));
  tbody.appendChild(head);
  tallies.push({ ids: g.ids, el: head.querySelector('.group-tally')! });
  for (const id of g.ids) {
    const t = TESTS.find((x) => x.id === id)!;
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class="buttons-cell"><button class="run">Run</button><button class="copy" hidden title="Copy this test's result, with the build, browser and GPU, as text">Copy</button></td><td>${t.name}</td><td class="status">-</td><td class="detail">${t.criterion}</td><td class="time"></td>`;
    tr.querySelector('.run')!.addEventListener('click', () => run(t.id));
    const copy = tr.querySelector('.copy') as HTMLButtonElement;
    copy.addEventListener('click', () => {
      const r = results.get(t.id);
      if (r) void copyText(copy, context() + NL + NL + report(r));
    });
    tbody.appendChild(tr);
    rows.set(t.id, tr);
  }
}

/** Each group's tally: how many of its tests have passed and failed so far. */
function showTallies() {
  for (const { ids, el } of tallies) {
    const done = ids.map((id) => results.get(id)).filter((r): r is TestResult => !!r);
    const failed = done.filter((r) => !r.pass).length;
    el.textContent = done.length ? `${done.length - failed} passed${failed ? `, ${failed} failed` : ''}` : '';
    el.className = 'group-tally' + (failed ? ' fail' : done.length === ids.length ? ' pass' : '');
  }
}

/** Run tests in turn (each waits for the one before, see run()). */
async function runMany(ids: string[]): Promise<TestResult[]> {
  await ready;
  const out: TestResult[] = [];
  for (const id of ids) out.push(await run(id));
  return out;
}

/** Where the tests ran: build, page, browser and GPU. */
function context(): string {
  return [`Shaderous validation, build ${BUILD_STAMP}`, `Page: ${location.href}`, `Browser: ${navigator.userAgent}`, document.getElementById('adapter')!.textContent ?? ''].join(NL);
}

/** A test's result as plain text: what it checks and what it measured. */
function report(r: TestResult): string {
  const lines = [`${r.id}: ${r.pass ? 'pass' : 'FAIL'} in ${r.seconds.toFixed(1)} s`, r.name, `Criterion: ${r.criterion}`];
  for (const [k, v] of Object.entries(r.metrics)) lines.push(`  ${k}: ${fmt(v)}`);
  return lines.join(NL);
}

/** Copy text, and say so on the button for a moment (or that it could not). */
async function copyText(button: HTMLButtonElement, text: string) {
  const label = button.textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copied';
  } catch {
    button.textContent = 'Copy failed';
  }
  setTimeout(() => (button.textContent = label), 1500);
}

/** The button for every failing test: shown while there is one. */
function showCopyFailing() {
  const failing = [...results.values()].filter((r) => !r.pass);
  copyFailing.hidden = failing.length === 0;
  copyFailing.textContent = `Copy all failing tests (${failing.length})`;
}
copyFailing.addEventListener('click', () => {
  // In the order of the table.
  const failing = TESTS.map((t) => results.get(t.id)).filter((r): r is TestResult => !!r && !r.pass);
  void copyText(copyFailing, [context(), ...failing.map(report)].join(NL + NL));
});

let chain: Promise<unknown> = Promise.resolve();

/** Tests run one at a time; each owns the GPU while it runs. */
function run(id: string): Promise<TestResult> {
  const p = chain.then(async () => {
    const tr = rows.get(id)!;
    const status = tr.querySelector('.status')!;
    status.textContent = 'running';
    status.className = 'status running';
    const r = await runTest(gpu, id);
    status.textContent = r.pass ? 'pass' : 'FAIL';
    status.className = 'status ' + (r.pass ? 'pass' : 'fail');
    tr.querySelector('.detail')!.innerHTML =
      `${r.criterion}<dl>` + Object.entries(r.metrics).map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join('') + '</dl>';
    tr.querySelector('.time')!.textContent = r.seconds.toFixed(1) + ' s';
    results.set(id, r);
    (tr.querySelector('.copy') as HTMLButtonElement).hidden = false;
    showCopyFailing();
    showTallies();
    return r;
  });
  chain = p.catch(() => {});
  return p;
}

const ready = (async () => {
  try {
    gpu = await initGpu();
    document.getElementById('adapter')!.textContent =
      `Adapter: ${gpu.info.vendor} ${gpu.info.architecture} ${gpu.info.description}`.trim();
  } catch (e) {
    document.getElementById('adapter')!.textContent = String(e);
    throw e;
  }
})();

window.shaderous = {
  ready,
  tests: TESTS.map((t) => t.id),
  debug: { Sim, AmrSim, configFor, PROBLEMS, problemById, gpu: () => gpu },
  run: async (id) => {
    await ready;
    return run(id);
  },
  // In the table's order, group by group.
  runAll: () => runMany(GROUPS.flatMap((g) => g.ids)),
};

document.getElementById('run-all')!.addEventListener('click', () => window.shaderous.runAll());

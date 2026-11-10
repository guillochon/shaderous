import { initGpu, type Gpu } from '../gpu/device';
import { runTest, TESTS, type TestResult } from './tests';
import { configFor, Sim } from '../sim';
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

for (const t of TESTS) {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><button>Run</button></td><td>${t.name}</td><td class="status">-</td><td class="detail">${t.criterion}</td><td class="time"></td>`;
  tr.querySelector('button')!.addEventListener('click', () => run(t.id));
  tbody.appendChild(tr);
  rows.set(t.id, tr);
}

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
  debug: { Sim, configFor, PROBLEMS, problemById, gpu: () => gpu },
  run: async (id) => {
    await ready;
    return run(id);
  },
  runAll: async () => {
    await ready;
    const out: TestResult[] = [];
    for (const t of TESTS) out.push(await run(t.id));
    return out;
  },
};

document.getElementById('run-all')!.addEventListener('click', () => window.shaderous.runAll());

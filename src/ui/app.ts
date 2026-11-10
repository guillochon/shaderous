// Interactive front end: controls, frame loop with adaptive substepping, diagnostics panel.
import { BUILD, reportError, setAdapterInfo } from './errors';
import { decodeLink, encodeLink, viewDefaults, MAX_ZOOM, type LinkOverrides } from './link';
import { initGpu, type Gpu } from '../gpu/device';
import { configFor, type Diagnostics, type LimiterKind, type RiemannKind, type SimConfig, type Simulation } from '../sim';
import { createSim, maxAmrLevels, supportsAmr } from '../create';
import { EOS_LABELS, type EosKind } from '../eos/eos';
import { FIELDS, PROBLEMS, problemById, type GravityMode } from '../problems';
import { defaultView, Renderer, type ViewState } from '../render/renderer';
import { bakeTransfer, COLORMAPS, defaultOpacity, OpacityEditor, sampleColormap } from '../render/transfer';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const opt = (v: string | number, label: string) => `<option value="${v}">${label}</option>`;

// A labelled control: small caption above the input, laid out on a two- or three-column grid.
const field = (label: string, control: string, id = '') => `<label class="f"${id ? ` id="${id}"` : ''}><span>${label}</span>${control}</label>`;
const select = (id: string, options: string) => `<select id="${id}">${options}</select>`;
const number = (id: string, attrs = '') => `<input id="${id}" type="number" step="any" ${attrs} />`;

const RES_3D = [32, 64, 128, 256, 512];
const RES_2D = [128, 256, 512, 1024, 2048, 4096];
// Largest uniform grids offered; beyond these the mesh must be adaptive.
const UNIFORM_MAX = { 2: 2048, 3: 256 };

const side = $('side');
side.innerHTML = `
  <div id="sheet-bar">
    <button id="sheet-toggle" aria-expanded="false" aria-controls="side" title="Show or hide the controls">
      <span class="grip"></span><span id="sheet-title">Controls</span>
    </button>
    <button id="sheet-run">Pause</button>
  </div>
  <header class="brand"><h1>SHADEROUS</h1><span id="adapter"></span></header>

  <section>
    ${select('problem', PROBLEMS.map((p) => opt(p.id, p.label)).join(''))}
    <p class="blurb" id="blurb"></p>
    <div class="g3">
      ${field('Grid', select('dim', opt(3, '3D') + opt(2, '2D')))}
      ${field('Resolution', select('res', ''))}
      ${field('Refinement', select('levels', opt(0, 'Uniform') + [1, 2, 3].map((l) => opt(l, `${l} level${l > 1 ? 's' : ''}`)).join('')))}
    </div>
    <div class="g3">
      ${field('End time', number('tend', 'placeholder="none"'))}
      <span id="pparams"></span>
    </div>
    <div class="buttons"><button id="run">Pause</button><button id="step">Step</button><button id="reset">Reset</button><button id="share" title="Copy a link that opens the simulator with these settings">Share</button></div>
  </section>

  <details open>
    <summary>View<span class="sum" id="sum-view"></span></summary>
    <div class="g2">
      ${field('Mode', select('mode', opt('volume', 'Volume') + opt('slice', 'Slice')))}
      ${field('Field', select('field', FIELDS.map((f, i) => opt(i, f)).join('')))}
      ${field('Slice normal', select('axis', opt(2, 'z') + opt(1, 'y') + opt(0, 'x')))}
      ${field('Slice position', '<input id="slicepos" type="range" min="0" max="0.999" step="0.001" />')}
      ${field('Range min', number('lo'))}
      ${field('Range max', number('hi'))}
      ${field('Colour map', select('cmap', Object.keys(COLORMAPS).map((c) => opt(c, c)).join('')))}
      ${field('Opacity', '<input id="opacity" type="range" min="1" max="300" step="1" />')}
    </div>
    <div class="checks">
      <label class="check"><input id="log" type="checkbox" />Log scale</label>
      <label class="check"><input id="auto" type="checkbox" checked />Auto range</label>
      <label class="check" title="Outline the blocks of an adaptive mesh in the slice view"><input id="blocks" type="checkbox" checked />Blocks</label>
    </div>
    <canvas id="tf" title="Opacity against field value: drag points, double-click to add or remove"></canvas>
  </details>

  <details>
    <summary>Hydro<span class="sum" id="sum-hydro"></span></summary>
    <div class="g2">
      ${field('Equation of state', select('eos', (Object.keys(EOS_LABELS) as EosKind[]).map((k) => opt(k, EOS_LABELS[k])).join('')))}
      ${field('gamma', number('gamma', 'step="0.01" min="1"'))}
      ${field('K (p = K rho^gamma)', number('K', 'min="0"'), 'row-K')}
      ${field('Radiation constant a', number('arad', 'min="0"'), 'row-arad')}
      ${field('Riemann solver', select('riemann', opt('hllc', 'HLLC') + opt('hll', 'HLL')))}
      ${field('Limiter', select('limiter', opt('koren', 'Third order (Koren)') + opt('mc', 'Monotonized central') + opt('vanleer', 'van Leer') + opt('minmod', 'Minmod')))}
      ${field('CFL number', number('cfl', 'step="0.05" min="0.05" max="1"'))}
    </div>
  </details>

  <details>
    <summary>Gravity<span class="sum" id="sum-grav"></span></summary>
    <div class="g3">
      ${field('Self-gravity', select('gravity', opt('off', 'Off') + opt('periodic', 'Periodic') + opt('isolated', 'Isolated')))}
      ${field('G', number('G', 'min="0"'))}
      ${field('V-cycles', number('vcycles', 'step="1" min="1" max="8"'))}
    </div>
    <div class="g3 jeans">
      <label class="check" title="Pressure floor that keeps the Jeans length resolved (Truelove)"><input id="jeansfloor" type="checkbox" />Jeans floor</label>
      ${field('Cells per Jeans length', number('jeanscells', 'step="1" min="4" max="32"'))}
    </div>
    <div class="checks">
      <label class="check" title="Flux-form gravitational work: conserves total energy"><input id="fluxwork" type="checkbox" />Conserve energy</label>
    </div>
  </details>

  <details open>
    <summary>Diagnostics</summary>
    <div id="diag"><dl></dl><dl></dl></div>
  </details>

  <footer><a href="validate.html">Validation suite</a> &middot; drag to orbit (pan in a slice), wheel or pinch to zoom, double-click to reset a slice<br><span title="Build of the page you are looking at">build ${BUILD}</span></footer>
`;

// On a phone the controls are a sheet under the simulation, closed until asked for.
{
  const toggle = $('sheet-toggle');
  toggle.addEventListener('click', () => {
    const open = side.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
    if (!open) side.scrollTop = 0;
  });
  // The bar's run button stands in for the one inside the closed sheet.
  const run = $('run');
  const mirror = $('sheet-run');
  mirror.addEventListener('click', () => run.click());
  new MutationObserver(() => (mirror.textContent = run.textContent)).observe(run, { childList: true, characterData: true, subtree: true });
  const problem = $<HTMLSelectElement>('problem');
  const title = () => ($('sheet-title').textContent = problem.selectedOptions[0]?.textContent ?? 'Controls');
  problem.addEventListener('change', title);
  title();
}

const canvas = $<HTMLCanvasElement>('view');
const hud = $('hud');

function fail(e: unknown) {
  reportError(e);
}

let gpu: Gpu;
let sim: Simulation | undefined;
let renderer: Renderer;
let cfg: SimConfig;
const view: ViewState = defaultView();
let running = true;
let stepOnce = false;
let building = false;
let finished = false;
let lastDiag: Diagnostics | undefined;

const editor = new OpacityEditor($<HTMLCanvasElement>('tf'), pushTransfer);

function pushTransfer() {
  renderer?.setTransfer(bakeTransfer(editor.colormap, editor.points));
  const cb = document.querySelector<HTMLCanvasElement>('#colorbar canvas')!;
  const g = cb.getContext('2d')!;
  for (let y = 0; y < 256; y++) {
    const [r, gg, b] = sampleColormap(editor.colormap, 1 - y / 255);
    g.fillStyle = `rgb(${r | 0},${gg | 0},${b | 0})`;
    g.fillRect(0, y, 1, 1);
  }
}

const num = (v: number) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? v.toPrecision(5) : v.toExponential(2));

/** Reflect cfg and view into the controls. */
function syncControls() {
  const two = cfg.ndim === 2;
  $<HTMLSelectElement>('problem').value = cfg.problem.id;
  syncProblemParams();
  const dim = $<HTMLSelectElement>('dim');
  dim.value = String(cfg.ndim);
  dim.options[1].disabled = !cfg.problem.supports2d;
  const res = $<HTMLSelectElement>('res');
  const wanted = (two ? RES_2D : RES_3D).map((n) => opt(n, two ? `${n}\u00b2` : `${n}\u00b3`)).join('');
  if (res.dataset.dim !== String(cfg.ndim)) {
    res.innerHTML = wanted;
    res.dataset.dim = String(cfg.ndim);
  }
  res.value = String(cfg.n);
  for (const o of res.options) o.disabled = cfg.amrLevels === 0 && +o.value > UNIFORM_MAX[cfg.ndim];
  const levels = $<HTMLSelectElement>('levels');
  levels.value = String(cfg.amrLevels);
  const amrOk = supportsAmr(cfg.problem);
  levels.disabled = !amrOk;
  levels.title = 'Adaptive mesh: levels of refinement below the coarsest grid';
  for (const o of levels.options) o.disabled = +o.value > maxAmrLevels(cfg.n);
  $<HTMLInputElement>('blocks').checked = view.showBlocks;
  $<HTMLInputElement>('blocks').disabled = cfg.amrLevels === 0;
  // A 2D run is always shown as its plane, and has no self-gravity.
  $<HTMLSelectElement>('mode').disabled = two;
  $<HTMLSelectElement>('axis').disabled = two;
  $<HTMLInputElement>('slicepos').disabled = two;
  for (const id of ['gravity', 'G', 'vcycles', 'jeansfloor']) $<HTMLInputElement>(id).disabled = two;
  $('sum-view').textContent = FIELDS[view.field];
  $('sum-hydro').textContent = `${EOS_LABELS[cfg.eos]}, ${cfg.riemann.toUpperCase()}`;
  const uniform = cfg.gext.some((g) => g !== 0);
  $('sum-grav').textContent = [cfg.gravity === 'off' ? '' : cfg.gravity, uniform ? 'uniform field' : ''].filter(Boolean).join(' + ') || 'off';
  $<HTMLInputElement>('tend').value = Number.isFinite(cfg.tEnd) ? String(cfg.tEnd) : '';
  $('blurb').textContent = cfg.problem.blurb;
  $<HTMLSelectElement>('eos').value = cfg.eos;
  $<HTMLInputElement>('gamma').value = String(+cfg.eosParams.gamma.toFixed(4));
  $<HTMLInputElement>('K').value = String(cfg.eosParams.K);
  $<HTMLInputElement>('arad').value = String(cfg.eosParams.arad);
  $('row-K').style.display = cfg.eos === 'barotropic' ? '' : 'none';
  $('row-arad').style.display = cfg.eos === 'gas_radiation' ? '' : 'none';
  $<HTMLSelectElement>('riemann').value = cfg.riemann;
  $<HTMLSelectElement>('limiter').value = cfg.limiter;
  $<HTMLInputElement>('cfl').value = String(cfg.cfl);
  $<HTMLSelectElement>('gravity').value = cfg.gravity;
  $<HTMLInputElement>('G').value = String(+cfg.G.toPrecision(6));
  $<HTMLInputElement>('vcycles').value = String(cfg.vcycles);
  $<HTMLInputElement>('jeansfloor').checked = cfg.jeansFloor;
  $<HTMLInputElement>('jeanscells').value = String(cfg.jeansCells);
  $<HTMLInputElement>('jeanscells').disabled = two || !cfg.jeansFloor;
  $<HTMLInputElement>('fluxwork').checked = cfg.gravEnergyFlux;
  $<HTMLSelectElement>('mode').value = two ? 'slice' : view.mode;
  $<HTMLSelectElement>('field').value = String(view.field);
  $<HTMLSelectElement>('axis').value = String(two ? 2 : view.sliceAxis);
  $<HTMLInputElement>('slicepos').value = String(view.slicePos);
  $<HTMLInputElement>('log').checked = view.log;
  $<HTMLInputElement>('opacity').value = String(view.opacity);
  $<HTMLSelectElement>('cmap').value = editor.colormap;
  $('run').textContent = running ? 'Pause' : 'Run';
}

/** The preset's own controls (for example mass ratio and impact parameter). */
function syncProblemParams() {
  const box = $('pparams');
  const defs = cfg.problem.params ?? [];
  if (box.dataset.problem !== cfg.problem.id) {
    box.dataset.problem = cfg.problem.id;
    box.innerHTML = defs.map((q) => field(q.label, number(`pp-${q.id}`, `min="${q.min}" max="${q.max}" step="${q.step ?? 'any'}"`))).join('');
    for (const q of defs) {
      const el = $<HTMLInputElement>(`pp-${q.id}`);
      el.addEventListener('change', () => {
        const v = el.value === '' || Number.isNaN(+el.value) ? cfg.params[q.id] : +el.value;
        cfg.params[q.id] = Math.min(q.max, Math.max(q.min, v));
        // A different orbit is a different run: start again.
        if (sim) {
          sim.cfg = cfg;
          sim.reset();
          finished = false;
          running = true;
          lastDiag = undefined;
        }
        syncControls();
      });
    }
  }
  for (const q of defs) {
    const el = $<HTMLInputElement>(`pp-${q.id}`);
    if (document.activeElement !== el) el.value = String(cfg.params[q.id]);
  }
}

// Rebuilds are asynchronous (shader compilation). They run one at a time, in the order the
// controls were touched, so a change made while another is compiling is never dropped.
let queue: Promise<void> = Promise.resolve();
function enqueue(job: () => Promise<void>): Promise<void> {
  queue = queue.then(async () => {
    building = true;
    try {
      await job();
    } catch (e) {
      fail(e);
    } finally {
      building = false;
      if (cfg) syncControls();
    }
  });
  return queue;
}

/** Build a simulation for the selected preset and resolution. */
function build(problemId: string, n: number, ndim: 2 | 3, levels: number, keepPhysics = false, link?: LinkOverrides) {
  return enqueue(async () => {
    const problem = problemById(problemId);
    if (!problem.supports2d) ndim = 3;
    const allowed = ndim === 2 ? RES_2D : RES_3D;
    if (!allowed.includes(n)) n = ndim === 2 ? 512 : 128;
    // Refinement: only where the preset allows it, and no deeper than the resolution permits.
    levels = supportsAmr(problem) ? Math.min(levels, maxAmrLevels(n)) : 0;
    if (levels === 0) n = Math.min(n, UNIFORM_MAX[ndim]);
    const next = configFor(problem, n, { ndim, amrLevels: levels });
    if (!Number.isFinite(next.tEnd)) next.tEnd = Infinity;
    if (keepPhysics && cfg && cfg.problem === problem && cfg.ndim === ndim) {
      // Same preset at a new resolution: keep the user's physics choices.
      const { n: _n, problem: _p, amrLevels: _l, ...rest } = cfg;
      Object.assign(next, rest);
    } else {
      const dv = viewDefaults(problem, ndim);
      Object.assign(view, { field: dv.field, log: dv.log, mode: problem.view?.mode ?? 'volume', sliceAxis: dv.sliceAxis, pitch: dv.pitch, zoom: 1, cx: 0.5, cy: 0.5 });
    }
    if (link) {
      // Settings carried by a shared link, on top of the preset's defaults.
      Object.assign(next, link.cfg);
      Object.assign(next.eosParams, link.eosParams);
      Object.assign(next.params, link.params);
      if (ndim === 2) next.gravity = 'off';
      Object.assign(view, link.view);
      if (link.auto === false) $<HTMLInputElement>('auto').checked = false;
      if (link.colormap || link.opacity) {
        editor.set(link.opacity ?? editor.points, link.colormap ?? editor.colormap);
        pushTransfer();
      }
    }
    const old = sim;
    sim = undefined;
    old?.destroy();
    cfg = next;
    if (ndim === 2) view.sliceAxis = 2;
    const s = await createSim(gpu, cfg);
    await renderer.attach(s);
    sim = s;
    finished = false;
    running = true; // a freshly loaded problem starts playing
    lastDiag = undefined;
    nsub = 1;
  });
}

/** The current settings as a query string (see ui/link.ts). */
function linkQuery(): string {
  return encodeLink({ cfg, view, auto: $<HTMLInputElement>('auto').checked, colormap: editor.colormap, opacity: editor.points }).toString();
}

// The address bar follows the settings, so the page can be shared or reloaded as it stands.
let lastQuery = '';
let linkFrame = 0;
function syncAddress(force = false) {
  if (!cfg || !sim || building) return;
  if (!force && ++linkFrame % 30) return;
  const q = linkQuery();
  if (q === lastQuery) return;
  lastQuery = q;
  history.replaceState(null, '', `${location.pathname}?${q}`);
}

/** Parameter changes that only touch the uniform block. */
function applyParams() {
  if (!sim) return;
  sim.cfg = cfg;
  sim.writeParams();
  finished = false;
}

/** Changes that need shader reassembly. */
function rebuildPipelines(resetAfter = false) {
  return enqueue(async () => {
    if (!sim) return;
    sim.cfg = cfg;
    sim.writeParams();
    await sim.buildHydro();
    await renderer.attach(sim);
    if (resetAfter) {
      sim.reset();
      lastDiag = undefined;
      if (finished) running = true;
    }
    finished = false;
  });
}

function bind() {
  const on = (id: string, fn: (el: HTMLInputElement & HTMLSelectElement) => void, ev = 'change') =>
    $(id).addEventListener(ev, () => fn($(id) as HTMLInputElement & HTMLSelectElement));
  const numOr = (el: HTMLInputElement, fallback: number) => (el.value === '' || Number.isNaN(+el.value) ? fallback : +el.value);

  const picked = () => ({
    id: $<HTMLSelectElement>('problem').value,
    n: +$<HTMLSelectElement>('res').value,
    ndim: (+$<HTMLSelectElement>('dim').value === 2 ? 2 : 3) as 2 | 3,
    levels: +$<HTMLSelectElement>('levels').value,
  });
  on('problem', () => {
    const { id, n, ndim, levels } = picked();
    build(id, n, ndim, levels);
  });
  on('dim', () => {
    const { id, ndim, levels } = picked();
    build(id, 0, ndim, levels); // 0: use the default resolution for that dimensionality
  });
  on('res', () => {
    const { id, n, ndim, levels } = picked();
    build(id, n, ndim, levels, true);
  });
  on('levels', () => {
    const { id, n, ndim, levels } = picked();
    build(id, n, ndim, levels, true);
  });
  on('blocks', (el) => (view.showBlocks = el.checked));
  on('tend', (el) => {
    cfg.tEnd = el.value === '' ? Infinity : +el.value;
    applyParams();
  });
  $('run').addEventListener('click', () => {
    running = !running;
    if (running && finished) {
      // Past the end time: keep going without a limit.
      cfg.tEnd = Infinity;
      applyParams();
    }
    syncControls();
  });
  $('step').addEventListener('click', () => {
    running = false;
    stepOnce = true;
    syncControls();
  });
  $('reset').addEventListener('click', () => {
    if (!sim) return;
    sim.cfg = cfg;
    sim.reset();
    if (finished) running = true; // replay a run that had reached its end time
    finished = false;
    lastDiag = undefined;
    syncControls();
  });

  on('eos', (el) => {
    cfg.eos = el.value as EosKind;
    // The stored internal energy means something different under a new equation of state,
    // so start the problem again.
    rebuildPipelines(true);
  });
  on('gamma', (el) => {
    cfg.eosParams.gamma = Math.max(1, numOr(el, cfg.eosParams.gamma));
    if (cfg.eos !== 'barotropic') cfg.eosParams.gamma = Math.max(1.001, cfg.eosParams.gamma);
    applyParams();
  });
  on('K', (el) => {
    cfg.eosParams.K = numOr(el, cfg.eosParams.K);
    applyParams();
  });
  on('arad', (el) => {
    cfg.eosParams.arad = numOr(el, cfg.eosParams.arad);
    applyParams();
  });
  on('riemann', (el) => {
    cfg.riemann = el.value as RiemannKind;
    rebuildPipelines();
  });
  on('limiter', (el) => {
    cfg.limiter = el.value as LimiterKind;
    rebuildPipelines();
  });
  on('cfl', (el) => {
    cfg.cfl = Math.min(1, Math.max(0.05, numOr(el, cfg.cfl)));
    applyParams();
  });
  const gravityChanged = () => {
    applyParams();
    sim?.refreshPotential();
  };
  on('gravity', (el) => {
    cfg.gravity = el.value as GravityMode;
    gravityChanged();
  });
  on('G', (el) => {
    cfg.G = numOr(el, cfg.G);
    gravityChanged();
  });
  on('vcycles', (el) => {
    cfg.vcycles = Math.min(8, Math.max(1, Math.round(numOr(el, cfg.vcycles))));
    applyParams();
  });
  on('jeansfloor', (el) => {
    cfg.jeansFloor = el.checked;
    applyParams();
    syncControls();
  });
  on('jeanscells', (el) => {
    cfg.jeansCells = Math.min(32, Math.max(4, Math.round(numOr(el, cfg.jeansCells))));
    applyParams();
    syncControls();
  });
  on('fluxwork', (el) => {
    cfg.gravEnergyFlux = el.checked;
    applyParams();
  });

  $('share').addEventListener('click', async () => {
    syncAddress(true);
    const button = $('share');
    try {
      await navigator.clipboard.writeText(location.href);
      button.textContent = 'Copied';
    } catch {
      // No clipboard access: the address bar already holds the link.
      prompt('Copy this link', location.href);
    }
    setTimeout(() => (button.textContent = 'Share'), 2000);
  });

  on('mode', (el) => (view.mode = el.value as ViewState['mode']));
  on('field', (el) => {
    view.field = +el.value;
    syncControls();
  });
  on('axis', (el) => (view.sliceAxis = +el.value));
  on('slicepos', (el) => (view.slicePos = +el.value), 'input');
  on('log', (el) => (view.log = el.checked));
  on('opacity', (el) => (view.opacity = +el.value), 'input');
  on('cmap', (el) => {
    editor.colormap = el.value;
    editor.draw();
    pushTransfer();
  });
  const manual = () => {
    $<HTMLInputElement>('auto').checked = false;
    view.lo = numOr($<HTMLInputElement>('lo'), view.lo);
    view.hi = numOr($<HTMLInputElement>('hi'), view.hi);
  };
  on('lo', manual);
  on('hi', manual);

  // Orbit and zoom: drag with one pointer, pinch with two, or use the wheel. Positions are
  // tracked by hand because touch pointers do not report movementX/Y in every browser.
  const pointers = new Map<number, { x: number; y: number }>();
  const spread = () => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  // A slice (always, in a 2D run) is panned and zoomed in its plane instead of orbited.
  const flat = () => view.mode === 'slice' || cfg?.ndim === 2;
  /** Screen pixels across the whole slice at zoom 1; 0.94 is the margin in slice.wgsl. */
  const span = () => Math.max(1, Math.min(canvas.clientWidth, canvas.clientHeight) * 0.94);
  const clampPan = () => {
    const h = 0.5 / view.zoom;
    view.cx = Math.min(1 - h, Math.max(h, view.cx));
    view.cy = Math.min(1 - h, Math.max(h, view.cy));
  };
  const zoomAt = (clientX: number, clientY: number, factor: number) => {
    const r = canvas.getBoundingClientRect();
    const sx = (clientX - r.left - r.width / 2) / span();
    const sy = -(clientY - r.top - r.height / 2) / span();
    const z = Math.min(MAX_ZOOM, Math.max(1, view.zoom * factor));
    view.cx += sx * (1 / view.zoom - 1 / z);
    view.cy += sy * (1 / view.zoom - 1 / z);
    view.zoom = z;
    clampPan();
  };
  canvas.addEventListener('dblclick', () => {
    if (flat()) [view.zoom, view.cx, view.cy] = [1, 0.5, 0.5];
  });
  canvas.addEventListener('pointerdown', (e) => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    canvas.setPointerCapture(e.pointerId);
  });
  const release = (e: PointerEvent) => pointers.delete(e.pointerId);
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointermove', (e) => {
    const last = pointers.get(e.pointerId);
    if (!last) return;
    const before = pointers.size === 2 ? spread() : 0;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (flat()) {
      // Slice: the point under the pointer (or under the middle of a pinch) stays under it.
      const share = pointers.size === 2 ? 0.5 : 1;
      const k = share / (span() * view.zoom);
      view.cx -= (e.clientX - last.x) * k;
      view.cy += (e.clientY - last.y) * k;
      if (pointers.size === 2 && before > 0) {
        const [a, b] = [...pointers.values()];
        zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, spread() / before);
      } else clampPan();
    } else if (pointers.size === 1) {
      view.yaw -= (e.clientX - last.x) * 0.006;
      view.pitch = Math.min(1.5, Math.max(-1.5, view.pitch + (e.clientY - last.y) * 0.006));
    } else if (pointers.size === 2 && before > 0) {
      view.dist = Math.min(6, Math.max(0.9, (view.dist * before) / Math.max(spread(), 1)));
    }
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      if (flat()) zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015));
      else view.dist = Math.min(6, Math.max(0.9, view.dist * Math.exp(e.deltaY * 0.001)));
    },
    { passive: false },
  );
}

/** Displayed range from the latest diagnostics, unless the user typed one. */
function updateRange(d: Diagnostics) {
  if (!$<HTMLInputElement>('auto').checked) return;
  let [lo, hi] = d.ranges[view.field];
  if (!(hi > lo)) hi = lo + Math.max(Math.abs(lo) * 1e-3, 1e-12);
  if (view.log) {
    hi = Math.max(hi, 1e-30);
    lo = Math.max(lo, hi * 1e-5); // at most five decades
  }
  view.lo = lo;
  view.hi = hi;
  const a = document.activeElement;
  if (a !== $('lo')) setValue($<HTMLInputElement>('lo'), num(lo));
  if (a !== $('hi')) setValue($<HTMLInputElement>('hi'), num(hi));
}

// The live readouts are written into text nodes created once, inside fixed-size boxes with
// layout containment (see style.css), and only when the text actually changes. Updating them
// therefore never rebuilds DOM and never triggers layout of the sidebar or the page.
function setValue(el: HTMLInputElement, v: string) {
  if (el.value !== v) el.value = v;
}
function liveText(parent: HTMLElement): (v: string) => void {
  const node = document.createTextNode('');
  parent.textContent = '';
  parent.appendChild(node);
  let last = '';
  return (v) => {
    if (v !== last) node.data = last = v;
  };
}

// [label, tooltip]; the first seven fill the left column, the rest the right.
const DIAG_ROWS: [string, string][] = [
  ['t', 'Time'], ['dt', 'Timestep'], ['steps', 'Steps taken'], ['mass', 'Total mass'],
  ['Mach', 'Maximum Mach number'], ['floor', 'Cells at the density or pressure floor (emptied voids, usually)'],
  ['Jeans', 'Cells whose thermal Jeans length is under 4 cells. With the Jeans floor on these are the cells it is holding up'],
  ['p x', 'Total x momentum'], ['p y', 'Total y momentum'], ['p z', 'Total z momentum'],
  ['E kin', 'Kinetic energy'], ['E int', 'Internal energy'], ['E grav', 'Gravitational energy'], ['E tot', 'Total energy'],
];
const diagCols = $('diag').querySelectorAll('dl');
const diagCells = DIAG_ROWS.map(([label, title], i) => {
  const dt = document.createElement('dt');
  dt.textContent = label;
  dt.title = title;
  const dd = document.createElement('dd');
  diagCols[i < 7 ? 0 : 1].append(dt, dd);
  return { dd, set: liveText(dd), warn: false };
});
const setHud = liveText(hud);
const setCbHi = liveText($('cb-hi'));
const setCbLo = liveText($('cb-lo'));

function showDiagnostics(d: Diagnostics) {
  const values: [string, boolean][] = [
    [num(d.t), false],
    [num(d.dt), false],
    [String(d.steps), false],
    [num(d.mass), false],
    [num(d.maxMach), false],
    [String(d.floored), d.floored > 0],
    [String(d.jeansUnresolved), d.jeansUnresolved > 0 && !cfg.jeansFloor],
    [d.momentum[0].toExponential(2), false],
    [d.momentum[1].toExponential(2), false],
    [d.momentum[2].toExponential(2), false],
    [num(d.eKin), false],
    [num(d.eInt), false],
    [num(d.eGrav), false],
    [num(d.eTot), false],
  ];
  values.forEach(([text, warn], i) => {
    const cell = diagCells[i];
    cell.set(text);
    if (warn !== cell.warn) {
      cell.warn = warn;
      cell.dd.classList.toggle('warn', warn);
    }
  });
  setCbHi(num(view.hi));
  setCbLo(num(view.lo));
}

// --- frame loop -------------------------------------------------------------------------
// Each frame encodes nsub substeps plus the render into one command buffer, and nsub adapts
// so the GPU just keeps up with the display.
//
// How "keeping up" is detected depends on the browser. Chrome reports completed work
// promptly, so one frame is in flight at a time and its measured GPU time steers nsub.
// Firefox only reports completion on a ~100 ms timer; waiting for it would cap the display
// at 10 fps. There, frames are submitted every display refresh, up to a small backlog, and
// nsub is steered by how many are still outstanding when a batch of completions arrives.
let nsub = 1;
let inflight = 0;
let capped = false; // a refresh was skipped because the backlog limit was reached
let batchPending = false;
let pollMs = 0; // how long the browser takes to report already-finished work
let maxInflight = 1;
let frame = 0;
let lastReadout = 0;
let stepRate = 0; // steps per second of wall time
let stepsSubmitted = 0;
let rateMark = { time: 0, steps: 0 };
let fps = 0;
let lastFrameTime = 0;
let refreshMs = 16.7; // display refresh interval, estimated from requestAnimationFrame
let lastTick = 0;
const MAX_SUB = 64;

async function measurePollLatency() {
  let best = Infinity;
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    gpu.device.queue.submit([]);
    await gpu.device.queue.onSubmittedWorkDone();
    best = Math.min(best, performance.now() - t0);
  }
  pollMs = best;
}

function tick(now: number) {
  requestAnimationFrame(tick);
  syncAddress();
  // Refresh interval: follow the shortest recent gap, so skipped refreshes do not inflate it.
  if (lastTick) {
    const gap = Math.min(Math.max(now - lastTick, 4), 40);
    refreshMs = gap < refreshMs ? gap : refreshMs + 0.02 * (gap - refreshMs);
  }
  lastTick = now;
  // Allow enough frames in flight to cover the completion-report delay.
  maxInflight = pollMs < 8 ? 1 : Math.ceil(pollMs / refreshMs) + 2;
  if (!sim || building) return;
  if (inflight >= maxInflight) {
    capped = true;
    return;
  }
  const s = sim;
  const enc = gpu.device.createCommandEncoder();
  const steps = running && !finished ? nsub : stepOnce ? 1 : 0;
  stepOnce = false;
  // An adaptive run may take fewer steps than asked while it waits for the mesh to catch up.
  const taken = steps ? s.encodeSteps(enc, steps) : 0;
  stepsSubmitted += taken;
  if (taken < steps && running) nsub = Math.max(1, taken);
  renderer.encode(enc, s, view);
  gpu.device.queue.submit([enc.finish()]);
  s.submitted();

  inflight++;
  const adapt = steps > 0 && running;
  const t0 = performance.now();
  gpu.device.queue.onSubmittedWorkDone().then(() => {
    inflight--;
    if (maxInflight === 1) {
      // Prompt completion: steer on the measured GPU time of this frame.
      if (!adapt) return;
      // Aim to use most of one refresh interval.
      const ms = performance.now() - t0;
      if (ms < refreshMs * 0.6) nsub = Math.min(nsub + 1, MAX_SUB);
      else if (ms > refreshMs * 1.1) nsub = Math.max(1, Math.floor(nsub * 0.75));
    } else if (!batchPending) {
      // Completions arrive in batches; judge once per batch, after all of them have landed.
      batchPending = true;
      setTimeout(() => {
        batchPending = false;
        if (running && !finished) {
          if (capped || inflight >= 3) nsub = Math.max(1, Math.floor(nsub * 0.75));
          else if (inflight <= 1) nsub = Math.min(nsub + 1, MAX_SUB);
        }
        capped = false;
      }, 0);
    }
  });

  if (lastFrameTime) fps = 0.9 * fps + 0.1 * (1000 / Math.max(now - lastFrameTime, 1));
  lastFrameTime = now;

  // Readouts refresh five times a second, whatever the display rate.
  frame++;
  if (now - lastReadout >= 200) {
    lastReadout = now;
    s.readDiagnostics().then((d) => {
      if (!d || s !== sim) return;
      lastDiag = d;
      // dt = 0 after some progress means the clock reached the end time.
      if (d.dt === 0 && d.steps > 0 && running && !finished) {
        finished = true;
        running = false;
        syncControls();
      }
      updateRange(d);
      showDiagnostics(d);
    });
    if (now - rateMark.time > 500) {
      stepRate = ((stepsSubmitted - rateMark.steps) * 1000) / (now - rateMark.time);
      rateMark = { time: now, steps: stepsSubmitted };
    }
    const bh = lastDiag?.perturber;
    const where = bh ? `  •  perturber at r = ${Math.hypot(...bh).toFixed(2)}` : '';
    const state = finished ? 'finished' : running ? `${nsub} step${nsub > 1 ? 's' : ''}/frame, ${stepRate.toFixed(0)} steps/s` : 'paused';
    const mesh = s.status();
    setHud(`${cfg.n}${cfg.ndim === 2 ? '²' : '³'}  •  ${fps.toFixed(0)} fps  •  ${state}${where}${mesh ? '  •  ' + mesh : ''}`);
  }
}

(async () => {
  try {
    gpu = await initGpu();
    // Firefox reports no adapter details; the fallback flag is the one thing worth flagging.
    const name = `${gpu.info.vendor} ${gpu.info.architecture}`.trim() || 'WebGPU adapter';
    setAdapterInfo(`${name}${gpu.info.description ? ' (' + gpu.info.description + ')' : ''}`);
    $('adapter').textContent = gpu.info.fallback ? `${name} (software fallback: expect it to be slow)` : name;
    renderer = new Renderer(gpu, canvas);
    await renderer.init();
    await measurePollLatency();
    editor.set(defaultOpacity(), 'inferno');
    pushTransfer();
    bind();
    const link = decodeLink(new URLSearchParams(location.search));
    await build(link.problem, link.n, link.ndim, link.amr, false, link);
    requestAnimationFrame(tick);
  } catch (e) {
    fail(e);
  }
})();

// Handles for the console and for tests.
Object.assign(window, { shaderousApp: { get sim() { return sim; }, get cfg() { return cfg; }, view, get diag() { return lastDiag; }, get nsub() { return nsub; }, get stepRate() { return stepRate; }, get fps() { return fps; }, get pollMs() { return pollMs; } } });

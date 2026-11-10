// Interactive front end: controls, frame loop with adaptive substepping, diagnostics panel.
import { initGpu, type Gpu } from '../gpu/device';
import { configFor, Sim, type Diagnostics, type LimiterKind, type RiemannKind, type SimConfig } from '../sim';
import { EOS_LABELS, type EosKind } from '../eos/eos';
import { FIELDS, PROBLEMS, problemById, type GravityMode } from '../problems';
import { defaultView, Renderer, type ViewState } from '../render/renderer';
import { bakeTransfer, COLORMAPS, defaultOpacity, OpacityEditor, sampleColormap } from '../render/transfer';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const opt = (v: string | number, label: string) => `<option value="${v}">${label}</option>`;

const side = $('side');
side.innerHTML = `
  <h1>SHADEROUS</h1>
  <div class="hint" id="adapter"></div>

  <h2>Problem</h2>
  <div class="row"><label>Preset</label><select id="problem">${PROBLEMS.map((p) => opt(p.id, p.label)).join('')}</select></div>
  <div class="row"><label>Resolution</label><select id="res">${[32, 64, 128, 256].map((n) => opt(n, `${n}³`)).join('')}</select></div>
  <div class="row"><label>End time</label><input id="tend" type="number" step="any" placeholder="run forever" /></div>
  <p class="blurb" id="blurb"></p>
  <div class="buttons"><button id="run">Pause</button><button id="step">Step</button><button id="reset">Reset</button></div>

  <h2>Hydro</h2>
  <div class="row"><label>Equation of state</label><select id="eos">${(Object.keys(EOS_LABELS) as EosKind[]).map((k) => opt(k, EOS_LABELS[k])).join('')}</select></div>
  <div class="row"><label>gamma</label><input id="gamma" type="number" step="0.01" min="1" /></div>
  <div class="row" id="row-K"><label>K (p = K rho^gamma)</label><input id="K" type="number" step="any" min="0" /></div>
  <div class="row" id="row-arad"><label>Radiation const a</label><input id="arad" type="number" step="any" min="0" /></div>
  <div class="row"><label>Riemann solver</label><select id="riemann">${opt('hllc', 'HLLC')}${opt('hll', 'HLL')}</select></div>
  <div class="row"><label>Limiter</label><select id="limiter">${opt('mc', 'Monotonized central')}${opt('vanleer', 'van Leer')}${opt('minmod', 'Minmod')}</select></div>
  <div class="row"><label>CFL number</label><input id="cfl" type="number" step="0.05" min="0.05" max="1" /></div>

  <h2>Gravity</h2>
  <div class="row"><label>Self-gravity</label><select id="gravity">${opt('off', 'Off')}${opt('periodic', 'Periodic')}${opt('isolated', 'Isolated')}</select></div>
  <div class="row"><label>G</label><input id="G" type="number" step="any" min="0" /></div>
  <div class="row"><label>V-cycles / step</label><input id="vcycles" type="number" step="1" min="1" max="8" /></div>
  <label class="check"><input id="jeansfloor" type="checkbox" />Jeans pressure floor (4 cells)</label>
  <label class="check"><input id="fluxwork" type="checkbox" />Energy-conserving gravity source</label>

  <h2>View</h2>
  <div class="row"><label>Mode</label><select id="mode">${opt('volume', 'Volume')}${opt('slice', 'Slice')}</select></div>
  <div class="row"><label>Field</label><select id="field">${FIELDS.map((f, i) => opt(i, f)).join('')}</select></div>
  <div class="row"><label>Slice</label><div class="pair"><select id="axis">${opt(2, 'z')}${opt(1, 'y')}${opt(0, 'x')}</select><input id="slicepos" type="range" min="0" max="0.999" step="0.001" /></div></div>
  <label class="check"><input id="log" type="checkbox" />Log scale</label>
  <label class="check"><input id="auto" type="checkbox" checked />Auto range</label>
  <div class="row"><label>Range</label><div class="pair"><input id="lo" type="number" step="any" /><input id="hi" type="number" step="any" /></div></div>
  <div class="row"><label>Colour map</label><select id="cmap">${Object.keys(COLORMAPS).map((c) => opt(c, c)).join('')}</select></div>
  <div class="row"><label>Opacity</label><input id="opacity" type="range" min="1" max="300" step="1" /></div>
  <canvas id="tf"></canvas>
  <div class="hint">Opacity against field value: drag points, double-click to add or remove. Drag the view to orbit, wheel to zoom.</div>

  <h2>Diagnostics</h2>
  <dl id="diag"></dl>
  <div class="hint"><a href="validate.html">Validation suite</a></div>
`;

const canvas = $<HTMLCanvasElement>('view');
const hud = $('hud');

function fail(e: unknown) {
  const box = $('error');
  box.style.display = 'flex';
  box.textContent = e instanceof Error ? e.message : String(e);
  console.error(e);
}

let gpu: Gpu;
let sim: Sim | undefined;
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

const num = (v: number) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? v.toPrecision(5) : v.toExponential(3));

/** Reflect cfg and view into the controls. */
function syncControls() {
  $<HTMLSelectElement>('problem').value = cfg.problem.id;
  $<HTMLSelectElement>('res').value = String(cfg.n);
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
  $<HTMLInputElement>('fluxwork').checked = cfg.gravEnergyFlux;
  $<HTMLSelectElement>('mode').value = view.mode;
  $<HTMLSelectElement>('field').value = String(view.field);
  $<HTMLSelectElement>('axis').value = String(view.sliceAxis);
  $<HTMLInputElement>('slicepos').value = String(view.slicePos);
  $<HTMLInputElement>('log').checked = view.log;
  $<HTMLInputElement>('opacity').value = String(view.opacity);
  $<HTMLSelectElement>('cmap').value = editor.colormap;
  $('run').textContent = running ? 'Pause' : 'Run';
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
function build(problemId: string, n: number, keepPhysics = false) {
  return enqueue(async () => {
    const problem = problemById(problemId);
    const next = configFor(problem, n);
    if (!Number.isFinite(next.tEnd)) next.tEnd = Infinity;
    if (keepPhysics && cfg) {
      // Same preset at a new resolution: keep the user's physics choices.
      const { n: _n, problem: _p, ...rest } = cfg;
      Object.assign(next, rest);
    } else {
      view.field = problem.view?.field ?? 0;
      view.log = problem.view?.log ?? true;
      view.mode = problem.view?.mode ?? 'volume';
    }
    const old = sim;
    sim = undefined;
    old?.destroy();
    cfg = next;
    const s = await Sim.create(gpu, cfg);
    await renderer.attach(s);
    sim = s;
    finished = false;
    running = true; // a freshly loaded problem starts playing
    lastDiag = undefined;
    nsub = 1;
  });
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

  on('problem', (el) => {
    const n = +$<HTMLSelectElement>('res').value;
    build(el.value, n);
  });
  on('res', (el) => {
    const id = $<HTMLSelectElement>('problem').value;
    build(id, +el.value, id === cfg.problem.id);
  });
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
  });
  on('fluxwork', (el) => {
    cfg.gravEnergyFlux = el.checked;
    applyParams();
  });

  on('mode', (el) => (view.mode = el.value as ViewState['mode']));
  on('field', (el) => (view.field = +el.value));
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

  // Orbit and zoom.
  let dragging = false;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', () => (dragging = false));
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    view.yaw -= e.movementX * 0.006;
    view.pitch = Math.min(1.5, Math.max(-1.5, view.pitch + e.movementY * 0.006));
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      view.dist = Math.min(6, Math.max(0.9, view.dist * Math.exp(e.deltaY * 0.001)));
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

const DIAG_ROWS = [
  'time', 'dt', 'steps', 'mass', 'momentum x', 'momentum y', 'momentum z', 'E kinetic', 'E internal',
  'E gravitational', 'E total', 'max Mach', 'floored cells', 'Jeans < 4 cells',
] as const;
const diagCells = DIAG_ROWS.map((label) => {
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  $('diag').append(dt, dd);
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
    [d.momentum[0].toExponential(3), false],
    [d.momentum[1].toExponential(3), false],
    [d.momentum[2].toExponential(3), false],
    [num(d.eKin), false],
    [num(d.eInt), false],
    [num(d.eGrav), false],
    [num(d.eTot), false],
    [num(d.maxMach), false],
    [String(d.floored), d.floored > 0],
    [String(d.jeansUnresolved), d.jeansUnresolved > 0],
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
  if (steps) s.encodeSteps(enc, steps);
  stepsSubmitted += steps;
  renderer.encode(enc, s, view);
  gpu.device.queue.submit([enc.finish()]);

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
    const state = finished ? 'finished' : running ? `${nsub} step${nsub > 1 ? 's' : ''}/frame, ${stepRate.toFixed(0)} steps/s` : 'paused';
    setHud(`${cfg.n}³  •  ${fps.toFixed(0)} fps  •  ${state}`);
  }
}

(async () => {
  try {
    gpu = await initGpu();
    // Firefox reports no adapter details; the fallback flag is the one thing worth flagging.
    const name = `${gpu.info.vendor} ${gpu.info.architecture}`.trim() || 'WebGPU adapter';
    $('adapter').textContent = gpu.info.fallback ? `${name} (software fallback: expect it to be slow)` : name;
    renderer = new Renderer(gpu, canvas);
    await renderer.init();
    await measurePollLatency();
    editor.set(defaultOpacity(), 'inferno');
    pushTransfer();
    bind();
    const q = new URLSearchParams(location.search);
    const id = PROBLEMS.some((p) => p.id === q.get('problem')) ? q.get('problem')! : PROBLEMS[0].id;
    const n = [32, 64, 128, 256].includes(+q.get('n')!) ? +q.get('n')! : 128;
    await build(id, n);
    requestAnimationFrame(tick);
  } catch (e) {
    fail(e);
  }
})();

// Handles for the console and for tests.
Object.assign(window, { shaderousApp: { get sim() { return sim; }, get cfg() { return cfg; }, view, get diag() { return lastDiag; }, get nsub() { return nsub; }, get stepRate() { return stepRate; }, get fps() { return fps; }, get pollMs() { return pollMs; } } });

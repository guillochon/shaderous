// Interactive front end: controls, frame loop with adaptive substepping, diagnostics panel.
import { BUILD, reportError, setAdapterInfo } from './errors';
import { decodeLink, encodeLink, viewDefaults, MAX_ZOOM, type LinkOverrides } from './link';
import { initGpu, type Gpu } from '../gpu/device';
import { capture, restartFrom } from '../io/capture';
import { maxLevel, type Snapshot } from '../io/snapshot';
import { configFor, MAX_OUTLINE_VERTICES, MAX_SOURCES, type BrushOp, type Diagnostics, type LimiterKind, type Outline, type RiemannKind, type SimConfig, type Simulation, type SourceSpec } from '../sim';
import { createSim, maxAmrLevels, supportsAmr } from '../create';
import { EOS_LABELS, type EosKind } from '../eos/eos';
import { FIELDS, PROBLEMS, problemById, type Bc3, type GravityMode } from '../problems';
import { cameraFrame, defaultView, Renderer, type ViewState } from '../render/renderer';
import { bakeTransfer, COLORMAPS, defaultOpacity, OpacityEditor, sampleColormap } from '../render/transfer';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const opt = (v: string | number, label: string) => `<option value="${v}">${label}</option>`;

// A labelled control: small caption above the input, laid out on a two- or three-column grid.
const field = (label: string, control: string, id = '') => `<label class="f"${id ? ` id="${id}"` : ''}><span>${label}</span>${control}</label>`;
const select = (id: string, options: string) => `<select id="${id}">${options}</select>`;
const number = (id: string, attrs = '') => `<input id="${id}" type="number" step="any" ${attrs} />`;

// What the gas meets at a face of the box, in the order of the codes in problems/types.ts.
const BC_KINDS = [
  { label: 'Periodic', title: 'Gas leaving through this face comes back in through the opposite one (both faces of an axis)' },
  { label: 'Outflow', title: 'Open: the state at the face continues beyond it, so gas may leave or enter' },
  { label: 'Reflecting', title: 'A solid wall' },
  { label: 'Outflow only', title: 'Open for gas leaving; nothing comes in' },
];

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
    <div class="g2 io">
      <button id="export" title="Save the state of the gas as an HDF5 file in the Athena++ layout (.athdf), readable by yt and VisIt">Export state</button>
      <button id="import" title="Load a state from an .athdf file: one saved here, or another code's with density, velocity and pressure">Import state</button>
      <input id="import-file" type="file" accept=".athdf,.h5,.hdf5,.hdf" hidden />
    </div>
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
    <summary>Boundaries<span class="sum" id="sum-bc"></span></summary>
    <div class="g2">
      ${['x', 'y', 'z'].map((ax, a) => (['lo', 'hi'] as const).map((side) =>
        field(`${ax} ${side === 'lo' ? 'lower' : 'upper'} face`, select(`bc-${side}-${a}`, BC_KINDS.map((k, i) => `<option value="${i}" title="${k.title}">${k.label}</option>`).join('')), `row-bc-${side}-${a}`)).join('')).join('')}
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

  <footer><a href="validate.html">Validation suite</a> &middot; <a href="https://github.com/guillochon/shaderous" target="_blank" rel="noopener">Source on GitHub</a><br>drag to orbit (pan in a slice), wheel or pinch to zoom, double-click to reset a slice<br>with an editing tool picked, drag edits the gas and right-drag moves the view; Esc puts the tool away<br><span title="Build of the page you are looking at">build ${BUILD}</span></footer>
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

// --- editing tools ------------------------------------------------------------------------
// A column of tools under the colour bar. Each paints with a soft ball (a disc in 2D) and
// keeps its own size and softness.
type ToolKind = BrushOp['kind'] | 'source';
const icon = (body: string) => `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
const TOOLS: { kind: ToolKind; label: string; title: string; icon: string }[] = [
  {
    kind: 'mass', label: 'Mass',
    title: 'Mass: click or hold to add gas; Shift removes it',
    icon: icon('<circle cx="6.5" cy="9.5" r="4.5" fill="currentColor" stroke="none" /><path d="M12.5 1.5v5M10 4h5" />'),
  },
  {
    kind: 'velocity', label: 'Velocity',
    title: 'Velocity: press, drag in a direction and let go to set the velocity there. A longer drag is faster, up to 5\u00d7 the fastest gas',
    icon: icon('<path d="M2.5 13.5L13 3M7 3h6v6" />'),
  },
  {
    kind: 'temperature', label: 'Temperature',
    title: 'Temperature: click to multiply by 1.5, or hold; Shift divides',
    icon: icon('<path d="M6 9.3V3a2 2 0 0 1 4 0v6.3a3.5 3.5 0 1 1-4 0z" /><path d="M8 6.5v4.5" /><circle cx="8" cy="11.7" r="1.3" fill="currentColor" stroke="none" />'),
  },
  {
    kind: 'obstacle', label: 'Obstacle',
    title: 'Obstacle: draw a solid body for the gas to flow around. In 3D it is cut through the box along the line of sight',
    icon: icon('<path d="M2 13.5h12v-7H8.5v-4H2z" fill="currentColor" stroke="none" />'),
  },
];
TOOLS.push({
  kind: 'source', label: 'Source or sink',
  title: 'Source: click to place one that feeds gas in evenly, or press and drag to aim it (a longer drag is faster). Hold Shift to place a sink, which takes gas away',
  icon: icon('<circle cx="8" cy="8" r="2.3" fill="currentColor" stroke="none" /><path d="M8 1.5v2.3M8 12.2v2.3M1.5 8h2.3M12.2 8h2.3M3.4 3.4l1.6 1.6M11 11l1.6 1.6M12.6 3.4L11 5M5 11l-1.6 1.6" />'),
});
/** Ways of drawing an obstacle. */
type ShapeMode = 'rect' | 'ellipse' | 'polygon';
const SHAPES: { mode: ShapeMode; title: string; icon: string }[] = [
  { mode: 'rect', title: 'Rectangle: drag from corner to corner. Shift makes a square; Alt draws from the centre', icon: icon('<rect x="2.5" y="4" width="11" height="8" />') },
  { mode: 'ellipse', title: 'Ellipse: drag across it. Shift makes a circle; Alt draws from the centre', icon: icon('<ellipse cx="8" cy="8" rx="5.5" ry="4" />') },
  {
    mode: 'polygon',
    title: 'Polygon: click each corner; a double-click, Enter or a click on the first corner closes it. Shift keeps edges at 45\u00b0 steps; Backspace takes the last corner back',
    icon: icon('<path d="M3 12.5L4.5 4l6.5-1.5 2.5 6.5-4.5 4z" />'),
  },
];
let shapeMode: ShapeMode = 'rect';
let obstacleErase = false;
const toolParams: Record<ToolKind, { size: number; softness: number; amount: number }> = {
  mass: { size: 0.06, softness: 0.5, amount: 0.25 },
  velocity: { size: 0.08, softness: 0.5, amount: 0 },
  temperature: { size: 0.06, softness: 0.5, amount: 0 },
  obstacle: { size: 0, softness: 0, amount: 0 },
  source: { size: 0.04, softness: 0.5, amount: 0.25 },
};
let tool: ToolKind | null = null;
/** Mass tool: new gas brings no momentum of its own. */
let massAtRest = true;
const slider = (id: string, label: string, title: string, min: number, max: number) =>
  `<label class="scrub" id="tp-${id}-row" title="${title}"><input id="tp-${id}" type="range" min="${min}" max="${max}" step="0.01" aria-label="${title}" /><span>${label}<output id="tp-${id}-out"></output></span></label>`;
$('tools').innerHTML =
  TOOLS.map((t) => `<button id="tool-${t.kind}" aria-pressed="false" aria-label="${t.label}" title="${t.title}">${t.icon}</button>`).join('') +
  `<div id="tool-params" hidden>
    ${slider('size', 'Size', 'Brush radius, as a share of the box', 0.01, 0.3)}
    ${slider('softness', 'Soft', 'Softness of the edge: 0 is hard, 1 fades all the way from the centre', 0, 1)}
    ${slider('amount', 'Amt', 'Gas added per click, as a share of the densest gas', 0.01, 1)}
    <button id="tool-rest" aria-pressed="true" title="p = 0 for new mass: added gas arrives at rest and shares the momentum already there, slowing the flow. Off: it arrives with the local velocity">p = 0</button>
    <div id="tool-shapes">${SHAPES.map((q) => `<button id="shape-${q.mode}" aria-pressed="false" aria-label="${q.mode}" title="${q.title}">${q.icon}</button>`).join('')}</div>
    <button id="tool-erase" aria-pressed="false" title="Erase: the shape drawn turns obstacle back into gas, at rest">Erase</button>
    <button id="tool-clear" title="Remove every source and sink">Clear</button>
  </div>`;

function syncTools() {
  // A barotropic gas has no temperature of its own to change.
  const noTemp = cfg?.eos === 'barotropic';
  if (noTemp && tool === 'temperature') tool = null;
  for (const t of TOOLS) {
    const b = $<HTMLButtonElement>(`tool-${t.kind}`);
    b.setAttribute('aria-pressed', String(tool === t.kind));
    b.disabled = t.kind === 'temperature' && noTemp;
    b.title = b.disabled ? 'The barotropic equation of state ties the temperature to the density' : t.title;
  }
  canvas.classList.toggle('tool', tool !== null);
  $('tool-params').hidden = tool === null;
  if (tool === null) {
    $('brush').style.display = 'none';
    return;
  }
  const q = toolParams[tool];
  $('tp-amount-row').hidden = tool !== 'mass' && tool !== 'source';
  // The same slider is the amount of a dab of mass and the flux of a source.
  const flux = tool === 'source';
  $('tp-amount-row').title = flux
    ? 'Flux: how fast a source feeds gas in, or a sink takes it away. At 100% a source adds the density of the densest gas in the time the fastest signal takes to cross it'
    : 'Gas added per click, as a share of the densest gas';
  $('tp-amount-row').querySelector('span')!.firstChild!.textContent = flux ? 'Flux' : 'Amt';
  $('tool-clear').hidden = !flux;
  $('tool-clear').textContent = sources.length ? `Clear ${sources.length}` : 'Clear';
  $<HTMLButtonElement>('tool-clear').disabled = sources.length === 0;
  $('tool-rest').hidden = tool !== 'mass';
  $('tool-rest').setAttribute('aria-pressed', String(massAtRest));
  // An obstacle is drawn as a shape, not painted with a brush.
  const shapes = tool === 'obstacle';
  $('tp-size-row').hidden = $('tp-softness-row').hidden = shapes;
  $('tool-shapes').hidden = $('tool-erase').hidden = !shapes;
  for (const sh of SHAPES) $(`shape-${sh.mode}`).setAttribute('aria-pressed', String(shapeMode === sh.mode));
  $('tool-erase').setAttribute('aria-pressed', String(obstacleErase));
  if (shapes) drawBrush();
  else if (!stroke) $('brush').style.display = 'none'; // until the pointer next moves
  for (const k of ['size', 'softness', 'amount'] as const) {
    const el = $<HTMLInputElement>(`tp-${k}`);
    el.value = String(q[k]);
    el.style.setProperty('--p', `${(100 * (q[k] - +el.min)) / (+el.max - +el.min)}%`);
    // Size is the brush radius and Amount a share of the densest gas, both shown in percent.
    $(`tp-${k}-out`).textContent = k === 'softness' ? q[k].toFixed(2) : `${Math.round(q[k] * 100)}%`;
  }
}
for (const t of TOOLS) {
  $(`tool-${t.kind}`).addEventListener('click', () => {
    tool = tool === t.kind ? null : t.kind;
    draft = null;
    syncTools();
  });
}
for (const sh of SHAPES) {
  $(`shape-${sh.mode}`).addEventListener('click', () => {
    shapeMode = sh.mode;
    draft = null;
    syncTools();
  });
}
$('tool-clear').addEventListener('click', () => {
  clearSources();
  syncTools();
});
$('tool-erase').addEventListener('click', () => {
  obstacleErase = !obstacleErase;
  syncTools();
});
for (const k of ['size', 'softness', 'amount'] as const) {
  $(`tp-${k}`).addEventListener('input', () => {
    if (!tool) return;
    toolParams[tool][k] = +$<HTMLInputElement>(`tp-${k}`).value;
    syncTools();
  });
}
$('tool-rest').addEventListener('click', () => {
  massAtRest = !massAtRest;
  syncTools();
});
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || !tool) return;
  // Esc first drops a shape that is being drawn, then puts the tool away.
  if (draft) draft = null;
  else tool = null;
  syncTools();
});

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

/** A press of an editing tool, from pointer down until its last edit has been handed to the GPU. */
interface Stroke {
  id: number;
  kind: ToolKind;
  /** where it began and where the pointer is now, in CSS pixels from the canvas corner */
  x0: number;
  y0: number;
  x: number;
  y: number;
  shift: boolean;
  /** volume view: distance from the eye of the plane, facing the camera, that the stroke paints on */
  depth: number;
  /** false until that plane is known (it takes a readback) */
  ready: boolean;
  released: boolean;
  dabbed: boolean;
  /** when the pointer went down, and up to when the flow of a held press has been applied (ms) */
  t0: number;
  flowed: number;
  /** the densest gas and the reference speed when the stroke began */
  rhoMax: number;
  vRef: number;
}
let stroke: Stroke | null = null;
/** The sources and sinks in the run, oldest first. A reset removes them. */
let sources: SourceSpec[] = [];

/** An obstacle being drawn, in CSS pixels from the canvas corner. */
interface Draft {
  mode: ShapeMode;
  /** the pointer dragging a rectangle or ellipse; -1 for a polygon, which is clicked out */
  id: number;
  /** where the drag began, or the polygon's corners so far */
  pts: [number, number][];
  /** where the pointer is */
  cur: [number, number];
}
let draft: Draft | null = null;
/** Modifier keys, as last seen: Shift constrains a shape, Alt draws it from its centre. */
const mods = { shift: false, alt: false };
/** The edit the next frame applies. */
let pendingBrush: BrushOp | null = null;

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
  $('blurb').textContent = cfg.restart ? `Loaded from a file, at t = ${num(cfg.restart.time)}. Reset returns to that state.` : cfg.problem.blurb;
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
  // Opacity belongs to the volume view: a slice is drawn solid.
  $<HTMLInputElement>('opacity').disabled = two || view.mode === 'slice';
  for (let a = 0; a < 3; a++) {
    $<HTMLSelectElement>(`bc-lo-${a}`).value = String(cfg.bcLo[a]);
    $<HTMLSelectElement>(`bc-hi-${a}`).value = String(cfg.bcHi[a]);
  }
  // A 2D run has no z faces.
  $('row-bc-lo-2').style.display = $('row-bc-hi-2').style.display = two ? 'none' : '';
  const kinds = [...new Set([...cfg.bcLo, ...cfg.bcHi].filter((_, i) => !two || i % 3 !== 2))];
  $('sum-bc').textContent = kinds.length === 1 ? BC_KINDS[kinds[0]].label.toLowerCase() : 'mixed';
  $<HTMLSelectElement>('cmap').value = editor.colormap;
  $('run').textContent = running ? 'Pause' : 'Run';
  syncTools();
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
          clearSources();
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
function build(problemId: string, n: number, ndim: 2 | 3, levels: number, keepPhysics = false, link?: LinkOverrides, snap?: Snapshot) {
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
      const { n: _n, problem: _p, amrLevels: _l, restart: _r, ...rest } = cfg;
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
      if (ndim === 2) {
        next.gravity = 'off';
        next.bcLo = [next.bcLo[0], next.bcLo[1], 0];
        next.bcHi = [next.bcHi[0], next.bcHi[1], 0];
      }
      Object.assign(view, link.view);
      if (link.auto === false) $<HTMLInputElement>('auto').checked = false;
      if (link.colormap || link.opacity) {
        editor.set(link.opacity ?? editor.points, link.colormap ?? editor.colormap);
        pushTransfer();
      }
    }
    // From a file: the same preset, started from the saved state instead of its own.
    if (snap) Object.assign(next, restartFrom(snap, problem, levels > 0, n >> levels));
    const old = sim;
    sim = undefined;
    stroke = null;
    draft = null;
    pendingBrush = null;
    sources = [];
    old?.destroy();
    cfg = next;
    if (ndim === 2) view.sliceAxis = 2;
    const s = await createSim(gpu, cfg);
    await renderer.attach(s);
    sim = s;
    finished = false;
    running = !snap; // a freshly loaded problem starts playing; a loaded state waits to be looked at
    lastDiag = undefined;
    nsub = 1;
  });
}

/** Save the state of the gas to an .athdf file (see io/athdf.ts). */
function exportState() {
  const button = $('export');
  return enqueue(async () => {
    if (!sim) return;
    button.textContent = 'Exporting…';
    try {
      const snap = await capture(sim, { program: 'shaderous', build: BUILD, query: linkQuery() });
      const { writeAthdf } = await import('../io/athdf');
      // gzip level 1: nearly all of the saving of higher levels (files are 8% to 86% of the raw
      // size, depending on how smooth the flow is) for the least time.
      const bytes = await writeAthdf(snap, undefined, 1);
      const name = `shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}-t${+snap.time.toPrecision(4)}.athdf`;
      const url = URL.createObjectURL(new Blob([bytes as Uint8Array<ArrayBuffer>], { type: 'application/x-hdf5' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } finally {
      button.textContent = 'Export state';
    }
  });
}

/** Start a run from an .athdf file: one of ours with its settings, or another code's. */
async function importState(file: File) {
  const button = $('import');
  button.textContent = 'Importing…';
  try {
    const { readAthdf } = await import('../io/athdf');
    const snap = await readAthdf(new Uint8Array(await file.arrayBuffer()));
    const cubic = snap.rootSize[0] === snap.rootSize[1] && (snap.ndim === 2 || snap.rootSize[2] === snap.rootSize[0]);
    const width = snap.hi.map((h, a) => h - snap.lo[a]);
    if (!cubic || Math.abs(width[0] - width[1]) > 1e-4 * width[0] || (snap.ndim === 3 && Math.abs(width[0] - width[2]) > 1e-4 * width[0])) {
      throw new Error('only square or cubic domains can be loaded');
    }
    // Our own file carries its settings. Anyone else's keeps the current ones and takes the
    // file's dimensionality and the nearest resolution on offer.
    const own = snap.meta !== undefined;
    const link = decodeLink(new URLSearchParams(own ? snap.meta!.query : linkQuery()));
    if (!own) {
      const problem = problemById(link.problem);
      if (snap.ndim === 2 && !problem.supports2d) throw new Error(`this file is 2D, and the ${problem.label} preset is 3D only: pick a 2D preset first`);
      const allowed = snap.ndim === 2 ? RES_2D : RES_3D;
      const finest = snap.rootSize[0] << maxLevel(snap);
      const fits = allowed.filter((n) => n <= finest);
      link.ndim = snap.ndim;
      link.n = fits.length ? fits[fits.length - 1] : allowed[0];
      link.amr = maxLevel(snap) > 0 && supportsAmr(problem) ? Math.min(maxLevel(snap), maxAmrLevels(link.n)) : 0;
      if (link.amr === 0) link.n = Math.min(link.n, UNIFORM_MAX[link.ndim]);
      // The run continues to wherever the current settings say; a foreign time means nothing here.
      delete link.cfg.tEnd;
    }
    await build(link.problem, link.n, link.ndim, link.amr, false, link, snap);
  } catch (e) {
    // A file that cannot be used is not a fault of the page: say so and carry on.
    alert(`Could not load ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    button.textContent = 'Import state';
  }
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
      clearSources();
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
    clearSources();
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

  $('export').addEventListener('click', () => exportState());
  const picker = $<HTMLInputElement>('import-file');
  $('import').addEventListener('click', () => picker.click());
  picker.addEventListener('change', () => {
    const file = picker.files?.[0];
    picker.value = ''; // so the same file can be chosen again
    if (file) importState(file);
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

  on('mode', (el) => {
    view.mode = el.value as ViewState['mode'];
    syncControls();
  });
  for (let a = 0; a < 3; a++) {
    for (const side of ['lo', 'hi'] as const) {
      on(`bc-${side}-${a}`, (el) => {
        const lo = [...cfg.bcLo] as Bc3;
        const hi = [...cfg.bcHi] as Bc3;
        const was = lo[a] === 0;
        (side === 'lo' ? lo : hi)[a] = +el.value;
        // Periodic faces come in pairs: the opposite face follows into, or out of, periodic.
        if ((lo[a] === 0) !== (hi[a] === 0)) (side === 'lo' ? hi : lo)[a] = +el.value;
        cfg.bcLo = lo;
        cfg.bcHi = hi;
        if (sim && cfg.amrLevels > 0 && lo[a] === 0 && !was) {
          // On an adaptive mesh a newly periodic axis joins blocks that were never neighbours
          // and may differ by several levels: build the mesh again from the start.
          sim.cfg = cfg;
          sim.reset();
          clearSources();
          finished = false;
          lastDiag = undefined;
        } else applyParams();
        syncControls();
      });
    }
  }
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
    if (tool) {
      // With a tool in hand a double click is two edits, or the end of a polygon.
      if (draft?.mode === 'polygon' && draft.pts.length >= 3) commitDraft();
      return;
    }
    if (flat()) [view.zoom, view.cx, view.cy] = [1, 0.5, 0.5];
  });
  const at = (e: PointerEvent): [number, number] => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const seeMods = (e: { shiftKey: boolean; altKey: boolean }) => {
    mods.shift = e.shiftKey;
    mods.alt = e.altKey;
  };
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    if (tool === 'obstacle' && sim && !building && e.button === 0 && pointers.size === 0) {
      seeMods(e);
      const p = at(e);
      if (shapeMode !== 'polygon') {
        draft = { mode: shapeMode, id: e.pointerId, pts: [p], cur: p };
      } else if (!draft) {
        draft = { mode: 'polygon', id: -1, pts: [p], cur: p };
      } else {
        const q = polygonPoint(draft, p);
        const [fx, fy] = draft.pts[0];
        const [lx, ly] = draft.pts[draft.pts.length - 1];
        if (draft.pts.length >= 3 && Math.hypot(q[0] - fx, q[1] - fy) < 8) commitDraft();
        // (The second press of a double click lands on the corner just placed.)
        else if (Math.hypot(q[0] - lx, q[1] - ly) > 2 && draft.pts.length < MAX_OUTLINE_VERTICES) draft.pts.push(q);
      }
      drawBrush();
      return;
    }
    if (tool && tool !== 'obstacle' && sim && !building && e.button === 0 && pointers.size === 0 && !stroke) {
      beginStroke(e);
      return;
    }
    if (stroke && !stroke.released) {
      // A second finger: this is a pinch after all. Whatever has been painted stays.
      pointers.set(stroke.id, { x: stroke.x + canvas.getBoundingClientRect().left, y: stroke.y + canvas.getBoundingClientRect().top });
      endStroke(false);
    }
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  });
  const release = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    if (draft && draft.id === e.pointerId) {
      // A rectangle or ellipse is finished by letting go.
      if (e.type === 'pointerup') commitDraft();
      else draft = null;
      drawBrush();
    }
    if (stroke?.id === e.pointerId && !stroke.released) endStroke(e.type === 'pointerup');
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  // With a tool in hand the right button moves the view, so it must not open a menu.
  canvas.addEventListener('contextmenu', (e) => {
    if (tool) e.preventDefault();
  });
  canvas.addEventListener('pointerleave', () => {
    if (!stroke) $('brush').style.display = 'none';
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Shift' && stroke) {
      stroke.shift = true;
      drawBrush();
    }
    if (!draft) return;
    if (e.key === 'Shift' || e.key === 'Alt') {
      seeMods(e);
      e.preventDefault(); // Alt alone would otherwise open the browser's menu
    } else if (draft.mode === 'polygon' && e.key === 'Enter' && draft.pts.length >= 3) {
      commitDraft();
    } else if (draft.mode === 'polygon' && e.key === 'Backspace') {
      draft.pts.pop();
      if (!draft.pts.length) draft = null;
      e.preventDefault();
    } else return;
    drawBrush();
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === 'Shift' && stroke) {
      stroke.shift = false;
      drawBrush();
    }
    if (draft && (e.key === 'Shift' || e.key === 'Alt')) {
      seeMods(e);
      e.preventDefault();
      drawBrush();
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (draft && (draft.id === e.pointerId || (draft.id === -1 && pointers.size === 0))) {
      seeMods(e);
      draft.cur = at(e);
      drawBrush();
      return;
    }
    if (stroke && stroke.id === e.pointerId && !stroke.released) {
      const r = canvas.getBoundingClientRect();
      stroke.x = e.clientX - r.left;
      stroke.y = e.clientY - r.top;
      stroke.shift = e.shiftKey;
      drawBrush();
      return;
    }
    if (tool && pointers.size === 0 && !stroke) {
      const r = canvas.getBoundingClientRect();
      drawBrush(e.clientX - r.left, e.clientY - r.top);
    }
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

// --- editing strokes ----------------------------------------------------------------------
type Vec3 = [number, number, number];
/** A slice (always, in a 2D run) rather than the volume view. */
const flatView = () => view.mode === 'slice' || cfg?.ndim === 2;
/** Screen pixels across the whole slice at zoom 1; 0.94 is the margin in slice.wgsl. */
const sliceSpan = () => Math.max(1, Math.min(canvas.clientWidth, canvas.clientHeight) * 0.94);
/** Drag length, in pixels, at which the velocity tool reaches its top speed. */
const fullDrag = () => 0.25 * Math.min(canvas.clientWidth, canvas.clientHeight);

/**
 * The point of the box under a point of the canvas, in fractions of the box, with the
 * directions that screen-right and screen-up correspond to there and the pixels per box length.
 * In the volume view the point lies on the plane facing the camera at the given distance.
 */
function locate(px: number, py: number, depth: number): { pos: Vec3; right: Vec3; up: Vec3; scale: number } {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (flatView()) {
    const a = view.cx + (px - w / 2) / (sliceSpan() * view.zoom);
    const b = view.cy - (py - h / 2) / (sliceSpan() * view.zoom);
    const axis = cfg.ndim === 2 ? 2 : view.sliceAxis;
    // The middle of the layer of cells on show.
    const k = (Math.min(cfg.n - 1, Math.floor(view.slicePos * cfg.n)) + 0.5) / cfg.n;
    const scale = sliceSpan() * view.zoom;
    if (axis === 0) return { pos: [k, a, b], right: [0, 1, 0], up: [0, 0, 1], scale };
    if (axis === 1) return { pos: [a, k, b], right: [1, 0, 0], up: [0, 0, 1], scale };
    return { pos: [a, b, k], right: [1, 0, 0], up: [0, 1, 0], scale };
  }
  const f = cameraFrame(view, w / h);
  const ux = ((2 * px) / w - 1) * (w / h) * f.tan;
  const uy = (1 - (2 * py) / h) * f.tan;
  // Along the ray, fwd + ux right + uy up reaches distance `depth` from the eye, measured along fwd.
  const pos = [0, 1, 2].map((i) => f.eye[i] + depth * (f.fwd[i] + ux * f.right[i] + uy * f.up[i]) + 0.5) as Vec3;
  return { pos, right: f.right as Vec3, up: f.up as Vec3, scale: h / (2 * f.tan * depth) };
}

// --- obstacles ----------------------------------------------------------------------------
/** Centre and half-sizes of the rectangle or ellipse being dragged out. */
function draftBox(d: Draft) {
  const [x0, y0] = d.pts[0];
  let dx = d.cur[0] - x0;
  let dy = d.cur[1] - y0;
  if (mods.shift) {
    // Shift: equal sides (a square, a circle).
    const m = Math.max(Math.abs(dx), Math.abs(dy));
    dx = dx < 0 ? -m : m;
    dy = dy < 0 ? -m : m;
  }
  // Alt: the press is the centre rather than a corner.
  return mods.alt ? { cx: x0, cy: y0, hx: Math.abs(dx), hy: Math.abs(dy) } : { cx: x0 + dx / 2, cy: y0 + dy / 2, hx: Math.abs(dx) / 2, hy: Math.abs(dy) / 2 };
}

/** Where the next corner of a polygon goes: with Shift, at a multiple of 45 degrees from the last. */
function polygonPoint(d: Draft, p: [number, number]): [number, number] {
  if (!mods.shift || !d.pts.length) return p;
  const [lx, ly] = d.pts[d.pts.length - 1];
  const len = Math.hypot(p[0] - lx, p[1] - ly);
  const ang = Math.round(Math.atan2(p[1] - ly, p[0] - lx) / (Math.PI / 4)) * (Math.PI / 4);
  return [lx + len * Math.cos(ang), ly + len * Math.sin(ang)];
}

/**
 * Takes a point of the box, in fractions of it, to canvas pixels (after division by the last
 * component): the inverse of locate(), for every depth at once. Column-major.
 */
function outlineMatrix(): number[] {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  // Rows of the matrix: x and y in pixels times the divisor, and the divisor.
  let rx: number[];
  let ry: number[];
  let rw: number[];
  if (flatView()) {
    const axis = cfg.ndim === 2 ? 2 : view.sliceAxis;
    const [ia, ib] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
    const k = sliceSpan() * view.zoom;
    rx = [0, 0, 0, w / 2 - view.cx * k];
    ry = [0, 0, 0, h / 2 + view.cy * k];
    rx[ia] = k;
    ry[ib] = -k;
    rw = [0, 0, 0, 1];
  } else {
    const f = cameraFrame(view, w / h);
    const origin = f.eye.map((c) => c + 0.5); // the eye, in fractions of the box
    const row = (v: number[]) => [...v, -(v[0] * origin[0] + v[1] * origin[1] + v[2] * origin[2])];
    rw = row(f.fwd); // distance along the view direction
    const right = row(f.right);
    const up = row(f.up);
    const kx = w / 2 / (f.tan * (w / h));
    const ky = h / 2 / f.tan;
    rx = rw.map((d, i) => (w / 2) * d + kx * right[i]);
    ry = rw.map((d, i) => (h / 2) * d - ky * up[i]);
  }
  return [0, 1, 2, 3].flatMap((c) => [rx[c], ry[c], 0, rw[c]]);
}

/** Hand the finished shape to the simulation. */
function commitDraft() {
  const d = draft;
  draft = null;
  if (!d) return;
  let shape: Pick<Outline, 'ellipse' | 'polygon'>;
  if (d.mode === 'polygon') {
    shape = { polygon: d.pts.flat() };
  } else {
    const b = draftBox(d);
    if (b.hx < 1 || b.hy < 1) return; // a click, not a drag
    shape = d.mode === 'ellipse'
      ? { ellipse: [b.cx, b.cy, b.hx, b.hy] }
      : { polygon: [b.cx - b.hx, b.cy - b.hy, b.cx + b.hx, b.cy - b.hy, b.cx + b.hx, b.cy + b.hy, b.cx - b.hx, b.cy + b.hy] };
  }
  pendingBrush = {
    kind: 'obstacle', pos: [0, 0, 0], radius: 1, softness: 0, vel: [0, 0, 0],
    amount: obstacleErase ? -1 : 1,
    outline: { matrix: outlineMatrix(), ...shape },
  };
}

/** The shape being drawn, as an SVG path. */
function draftPath(d: Draft): string {
  if (d.mode === 'polygon') {
    const [fx, fy] = d.pts[0];
    const next = polygonPoint(d, d.cur);
    // The corners so far, a rubber band to the pointer, and a ring on the first corner to close on.
    return `M${[...d.pts, next].map((p) => p.join(' ')).join('L')}` + `M${fx + 4} ${fy}a4 4 0 1 0 -8 0a4 4 0 1 0 8 0`;
  }
  const b = draftBox(d);
  if (d.mode === 'rect') return `M${b.cx - b.hx} ${b.cy - b.hy}h${2 * b.hx}v${2 * b.hy}h${-2 * b.hx}z`;
  return `M${b.cx - b.hx} ${b.cy}a${b.hx} ${b.hy} 0 1 0 ${2 * b.hx} 0a${b.hx} ${b.hy} 0 1 0 ${-2 * b.hx} 0`;
}

/** Remove every source and sink (the simulation drops its own on a reset). */
function clearSources() {
  if (sources.length && sim) sim.setSources([]);
  sources = [];
}

/** Draw the markers of the sources and sinks where they are in the current view. */
let marksDrawn = '';
function drawSources() {
  let html = '';
  if (sources.length && cfg) {
    const m = outlineMatrix();
    const at = (p: number[]) => {
      const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
      return { x: (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w, y: (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w, w };
    };
    for (const q of sources) {
      const c = at(q.pos);
      if (!(c.w > 0)) continue; // behind the camera
      // Size on screen: from a point one radius away along an axis in the plane of the view.
      const e = at(q.pos.map((v, i) => v + q.radius * locate(c.x, c.y, flatView() ? 1 : c.w).right[i]));
      const r = Math.hypot(e.x - c.x, e.y - c.y);
      const speed = Math.hypot(...q.vel);
      let arrow = '';
      if (speed > 0) {
        const tip = at(q.pos.map((v, i) => v + (2.2 * q.radius * q.vel[i]) / speed));
        const [ux, uy] = [tip.x - c.x, tip.y - c.y];
        const len = Math.hypot(ux, uy) || 1;
        const [nx, ny] = [ux / len, uy / len];
        arrow = `<line x1="${c.x}" y1="${c.y}" x2="${tip.x}" y2="${tip.y}" /><path d="M${tip.x + 5 * nx} ${tip.y + 5 * ny}L${tip.x - 4 * ny} ${tip.y + 4 * nx}L${tip.x + 4 * ny} ${tip.y - 4 * nx}z" />`;
      }
      html += `<g class="${q.decay > 0 ? 'sink' : 'source'}"><circle cx="${c.x}" cy="${c.y}" r="${r}" />${arrow}</g>`;
    }
  }
  if (html !== marksDrawn) $('marks').innerHTML = marksDrawn = html;
}

/** Outline of the brush under the pointer, and the arrow of a velocity stroke. */
function drawBrush(px?: number, py?: number) {
  const svg = $('brush');
  if (!tool || !cfg) {
    svg.style.display = 'none';
    return;
  }
  svg.classList.toggle('outline', tool === 'obstacle');
  if (tool === 'obstacle') {
    svg.style.display = 'block';
    svg.classList.toggle('minus', obstacleErase);
    svg.querySelector('path.shape')!.setAttribute('d', draft ? draftPath(draft) : '');
    return;
  }
  const kind = stroke?.kind ?? tool;
  // A velocity, or the aim of a source, is dragged out from where the press began.
  const drag = stroke?.kind === 'velocity' || (stroke?.kind === 'source' && !stroke.shift);
  const x = drag ? stroke!.x0 : (px ?? stroke?.x ?? 0);
  const y = drag ? stroke!.y0 : (py ?? stroke?.y ?? 0);
  const { scale } = locate(x, y, stroke?.ready ? stroke.depth : view.dist);
  svg.style.display = 'block';
  // Pressed: the tool is acting (or, for a velocity, being aimed). Shift takes away.
  const active = !!stroke && !stroke.released;
  svg.classList.toggle('active', active);
  svg.classList.toggle('minus', active && stroke!.shift && kind !== 'velocity');
  svg.classList.toggle('ball', cfg.ndim === 3);
  const radius = toolParams[kind].size * scale;
  const circle = svg.querySelector('circle')!;
  circle.setAttribute('cx', String(x));
  circle.setAttribute('cy', String(y));
  circle.setAttribute('r', String(radius));
  // The ball's equator (its great circle in the x-y plane) as the camera sees it: edge-on from
  // the side, a full circle from above. In a slice it is the outline or a line across it.
  const tilt = !flatView() ? Math.abs(Math.sin(view.pitch)) : view.sliceAxis === 2 ? 1 : 0;
  const ellipse = svg.querySelector('ellipse')!;
  for (const [name, v] of [['cx', x], ['cy', y], ['rx', radius], ['ry', radius * tilt]] as const) ellipse.setAttribute(name, String(v));
  const line = svg.querySelector('line')!;
  line.style.display = drag ? 'block' : 'none';
  if (drag) {
    // The arrow stops growing where the speed stops growing.
    const [dx, dy] = [stroke!.x - x, stroke!.y - y];
    const k = Math.min(1, fullDrag() / Math.max(Math.hypot(dx, dy), 1e-6));
    for (const [name, v] of [['x1', x], ['y1', y], ['x2', x + dx * k], ['y2', y + dy * k]] as const) line.setAttribute(name, String(v));
  }
}

function beginStroke(e: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  const [x, y] = [e.clientX - r.left, e.clientY - r.top];
  const d = lastDiag;
  // Fastest gas now. In a flow that is (nearly) at rest that would leave nothing to set, so the
  // reference never falls below a fifth of the sound speed of the hottest gas.
  const cs = d ? Math.sqrt(Math.max(0, cfg.eosParams.gamma * (cfg.eos === 'gas_radiation' ? cfg.eosParams.R : 1) * d.ranges[1][1])) : 0;
  const s: Stroke = {
    id: e.pointerId, kind: tool!, x0: x, y0: y, x, y, shift: e.shiftKey, depth: view.dist, ready: flatView(),
    released: false, dabbed: false, t0: performance.now(), flowed: 0,
    rhoMax: d?.ranges[0][1] ?? 1, vRef: Math.max(d?.ranges[4][1] ?? 0, 0.2 * cs),
  };
  stroke = s;
  if (!s.ready) {
    // Volume view: paint on the gas that is seen under the pointer, or, where the view is
    // empty, on the plane through the middle of the box.
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const [ux, uy] = [(2 * x) / w - 1, 1 - (2 * y) / h];
    renderer.pick(ux, uy).then(
      (t) => {
        if (t !== null) {
          const f = cameraFrame(view, w / h);
          // Distance along the ray to distance along the view direction.
          s.depth = t / Math.hypot(1, ux * (w / h) * f.tan, uy * f.tan);
        }
        s.ready = true;
      },
      () => (s.ready = true),
    );
  }
  drawBrush();
}

function endStroke(apply: boolean) {
  if (!stroke) return;
  stroke.released = true;
  // A velocity is set on letting go; a cancelled drag sets nothing.
  if (!apply && (stroke.kind === 'velocity' || stroke.kind === 'source')) stroke = null;
  drawBrush();
}

/** Turn the stroke in progress into this frame's edit. Called once per frame. */
function advanceStroke(now: number) {
  const s = stroke;
  if (!s || !sim || !s.ready) return;
  const q = toolParams[s.kind];
  if (s.kind === 'source') {
    if (!s.released) return;
    stroke = null;
    const at = locate(s.x0, s.y0, s.depth);
    const [dx, dy] = [s.x - s.x0, s.y0 - s.y];
    const len = Math.hypot(dx, dy);
    // A drag aims the source; a click (or a drag of a few pixels) leaves it feeding gas in at rest.
    const speed = s.shift || len < 6 ? 0 : Math.min(1, len / fullDrag()) * 5 * s.vRef;
    const vel = [0, 1, 2].map((i) => (speed ? (speed * (dx * at.right[i] + dy * at.up[i])) / len : 0)) as Vec3;
    // The flux is measured against the densest gas and the time a signal takes to cross the source.
    const perTime = (q.amount * (s.vRef || 1)) / (q.size * cfg.problem.box);
    sources.push({ pos: at.pos, radius: q.size, softness: q.softness, vel, rate: s.shift ? 0 : perTime * s.rhoMax, decay: s.shift ? perTime : 0 });
    if (sources.length > MAX_SOURCES) sources.shift(); // the oldest makes room
    sim.setSources(sources);
    syncTools();
    return;
  }
  const base = { kind: s.kind, radius: q.size, softness: q.softness, amount: 0, vel: [0, 0, 0] as Vec3 };
  if (s.kind === 'velocity') {
    if (!s.released) return;
    stroke = null;
    const at = locate(s.x0, s.y0, s.depth);
    const [dx, dy] = [s.x - s.x0, s.y0 - s.y];
    const len = Math.hypot(dx, dy);
    if (len < 4) return; // a click without a drag gives no direction
    const speed = Math.min(1, len / fullDrag()) * 5 * s.vRef;
    const vel = [0, 1, 2].map((i) => (speed * (dx * at.right[i] + dy * at.up[i])) / len) as Vec3;
    pendingBrush = { ...base, pos: at.pos, vel };
    return;
  }
  // Mass and temperature: one full application on the press, then, if the press is held for
  // a quarter of a second, a steady flow of four applications a second.
  let strength = 0;
  if (!s.dabbed) {
    s.dabbed = true;
    s.flowed = s.t0 + 250;
    strength = 1;
  }
  if (!s.released && now > s.flowed) {
    strength += (4 * Math.min(now - s.flowed, 100)) / 1000;
    s.flowed = now;
  }
  if (s.released) {
    stroke = null;
    // Self-gravity has to hear about the new mass.
    const target = sim;
    if (s.kind === 'mass' && cfg.gravity !== 'off' && cfg.ndim === 3) {
      setTimeout(() => target.diagnostics().then((d) => target === sim && target.massChanged(d.mass), fail), 0);
    }
  }
  if (strength === 0) return;
  const sign = s.shift ? -1 : 1;
  const amount = sign * strength * (s.kind === 'mass' ? q.amount * s.rhoMax : Math.log(1.5));
  pendingBrush = { ...base, pos: locate(s.x, s.y, s.depth).pos, amount, atRest: massAtRest };
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
  // Edits go in ahead of the steps, whether or not any are taken this frame.
  advanceStroke(now);
  if (pendingBrush) {
    s.encodeBrush(enc, pendingBrush);
    pendingBrush = null;
  }
  const steps = running && !finished ? nsub : stepOnce ? 1 : 0;
  stepOnce = false;
  // An adaptive run may take fewer steps than asked while it waits for the mesh to catch up.
  const taken = steps ? s.encodeSteps(enc, steps) : 0;
  stepsSubmitted += taken;
  if (taken < steps && running) nsub = Math.max(1, taken);
  renderer.encode(enc, s, view);
  drawSources();
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

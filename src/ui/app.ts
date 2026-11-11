// Interactive front end: controls, frame loop with adaptive substepping, diagnostics panel.
import { BUILD, reportError, setAdapterInfo } from './errors';
import { decodeLink, encodeLink, viewDefaults, MAX_ZOOM, type LinkOverrides } from './link';
import { initGpu, type Gpu } from '../gpu/device';
import { capture, restartFrom } from '../io/capture';
import { maxLevel, type Snapshot } from '../io/snapshot';
import { DiagHistory, drawFull, drawSpark, fmt3 } from './diag_plot';
import { type CellProbe, conducts, configFor, diffusionKey, MAX_MASSES, MAX_OUTLINE_VERTICES, MAX_SOURCES, type BrushOp, type Diagnostics, type LimiterKind, type Outline, type RiemannKind, type SimConfig, type Simulation, type SourceSpec } from '../sim';
import { createSim, maxAmrLevels, supportsAmr } from '../create';
import { EOS_LABELS, makeEos, type EosKind } from '../eos/eos';
import { COOLING_LABELS, type CoolingKind, type CoolingParams } from '../cooling/cooling';
import { familyById, familyOf, FIELD_TIPS, FIELDS, isSelfGravity, LINEAR_FIELDS, SIGNED_FIELDS, MENU, problemById, type Bc3, type GravityMode } from '../problems';
import { cameraFrame, defaultView, Renderer, type ViewState } from '../render/renderer';
import { bakeTransfer, COLORMAPS, defaultOpacity, OpacityEditor } from '../render/transfer';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const opt = (v: string | number, label: string) => `<option value="${v}">${label}</option>`;

// A labeled control: small caption above the input, laid out on a two- or three-column grid.
const field = (label: string, control: string, id = '', tip = '') =>
  `<label class="f"${id ? ` id="${id}"` : ''}${tip ? ` title="${tip}"` : ''}><span>${label}</span>${control}</label>`;
const select = (id: string, options: string, tip = '') => `<select id="${id}"${tip ? ` title="${tip}"` : ''}>${options}</select>`;
/**
 * A number field. A `step` in attrs is the change per click of its arrows (1 without one); a
 * `factor` makes the arrows multiply or divide by it instead (see the handler further down).
 */
const number = (id: string, attrs = '', factor = 0) =>
  `<input id="${id}" type="number" ${/\bstep=/.test(attrs) ? '' : 'step="any" '}${factor ? `data-factor="${factor}" ` : ''}${attrs} />`;

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
    <button id="sheet-run" title="Run or pause the simulation (space)">Pause</button>
  </div>
  <header class="brand"><h1>SHADEROUS</h1><span id="adapter"></span></header>

  <section>
    ${select('problem', MENU.map((g) => `<optgroup label="${g.group}">${g.families.map((f) => opt(f.id, f.label)).join('')}</optgroup>`).join(''), 'The problem to simulate: picking one starts it from its initial conditions')}
    <p class="blurb" id="blurb"></p>
    <div class="g3">
      ${field('Variant', select('variant', ''), 'row-variant', 'Versions of this problem with different setups')}
      <span id="pparams"></span>
    </div>
    <div class="buttons"><button id="run" title="Run or pause the simulation (space)">Pause</button><button id="step" title="Pause and take a single step">Step</button><button id="reset" title="Start the run again from its initial conditions">Reset</button><button id="share" title="Copy a link that opens the simulator with these settings">Share</button></div>
    <div class="io">
      <label class="check" title="Start each problem playing as soon as it loads, and play a finished run again on Reset. Off, a run waits for Play"><input id="autoplay" type="checkbox" />Autoplay</label>
      <button id="export" title="Save the state of the gas as an HDF5 file in the Athena++ layout (.athdf), readable by yt and VisIt">Export state</button>
      <button id="import" title="Load a state from an .athdf file: one saved here, or another code's with density, velocity and pressure">Import state</button>
      <input id="import-file" type="file" accept=".athdf,.h5,.hdf5,.hdf" hidden />
    </div>
  </section>

  <details open>
    <summary title="Dimensions, resolution and the adaptive mesh">Grid<span class="sum" id="sum-grid"></span></summary>
    <div class="g3">
      ${field('Dimensions', select('dim', opt(3, '3D') + opt(2, '2D')), '', 'Two or three dimensions. A 2D run is one layer of cells in x and y')}
      ${field('Resolution', select('res', ''), '', 'Cells across the box; on an adaptive mesh, at its finest level')}
      ${field('Refinement', select('levels', opt(0, 'Uniform') + [1, 2, 3].map((l) => opt(l, `${l} level${l > 1 ? 's' : ''}`)).join('')), '', 'Adaptive mesh: levels of refinement below the coarsest grid. Uniform puts every cell at the full resolution')}
    </div>
    <div class="g2">
      ${field('Refine above', number('refhi', 'step="0.05" min="0.05" max="1"'), '', 'Adaptive mesh: a block is split where the normalized second derivative of density or pressure exceeds this anywhere in it. The indicator is near 1 at a shock or contact and small where the gas varies smoothly. Lower refines more. Changes take effect without a restart')}
      ${field('Merge below', number('reflo', 'step="0.05" min="0" max="1"'), '', 'Adaptive mesh: four sibling blocks are merged where the indicator is below this in all of them. Kept below the refinement threshold. Changes take effect without a restart')}
    </div>
  </details>

  <details>
    <summary title="When the run stops and how long its steps are">Time<span class="sum" id="sum-time"></span></summary>
    <div class="g3">
      ${field('End time', number('tend', 'placeholder="none" min="0"', 1.25), '', 'The run pauses when it reaches this time. Empty runs on without end')}
      ${field('CFL number', number('cfl', 'step="0.05" min="0.05" max="1"'), '', 'Timestep as a fraction of the time a signal takes to cross a cell. Smaller is more stable and slower')}
      ${field('Local timesteps', select('tbins', opt(0, 'Off') + [1, 2, 3].map((k) => opt(k, `Up to ${1 << k}× longer`)).join('')), '', 'Adaptive mesh: blocks of slow gas take longer steps than the fastest block, where that saves work')}
    </div>
  </details>

  <details open>
    <summary title="How the gas is drawn">View<span class="sum" id="sum-view"></span></summary>
    <div class="g2">
      ${field('Mode', select('mode', opt('volume', 'Volume') + opt('slice', 'Slice')), '', 'Volume renders the gas as a glowing fog seen through the box; Slice shows one plane of cells')}
      ${field('Field', select('field', FIELDS.map((f, i) => `<option value="${i}" title="${FIELD_TIPS[i]}">${f}</option>`).join('')), '', 'The quantity the colors show. Fields of either sign (velocity components, heating, vorticity, divergence) are shown on a linear scale centered on zero')}
      ${field('Slice normal', select('axis', opt(2, 'z') + opt(1, 'y') + opt(0, 'x')), '', 'The axis the slice is perpendicular to')}
      ${field('Slice position', '<input id="slicepos" type="range" min="0" max="0.999" step="0.001" />', '', 'Where the slice cuts the box along its normal')}
    </div>
  </details>

  <details>
    <summary title="Equation of state and the hydrodynamics solver">Hydro<span class="sum" id="sum-hydro"></span></summary>
    <div class="g2">
      ${field('Equation of state', select('eos', (Object.keys(EOS_LABELS) as EosKind[]).map((k) => opt(k, EOS_LABELS[k])).join('')), '', 'How pressure follows from density and internal energy. Changing it carries each cell\u2019s pressure over, and the run continues')}
      ${field('gamma', number('gamma', 'step="0.05" min="1"'), '', 'Adiabatic index \u03b3: p = (\u03b3 - 1) \u03c1 e for an ideal gas, p = K \u03c1^\u03b3 for a barotropic one')}
      ${field('K (p = K rho^gamma)', number('K', 'min="0"', 2), 'row-K', 'Barotropic constant K: p = K \u03c1^\u03b3. With \u03b3 = 1 it is the square of the sound speed')}
      ${field('Radiation constant a', number('arad', 'min="0"', 2), 'row-arad', 'Radiation pressure a T\u2074 / 3 is added to the gas pressure')}
      ${field('Riemann solver', select('riemann', opt('hllc', 'HLLC') + opt('hllclm', 'HLLC, low Mach') + opt('hll', 'HLL')), '', 'How the flux between neighboring cells is worked out. HLLC resolves contact waves; its low-Mach version cuts dissipation in slow flow; HLL is the most diffusive and the most robust')}
      ${field('Limiter', select('limiter', opt('koren', 'Third order (Koren)') + opt('mc', 'Monotonized central') + opt('vanleer', 'van Leer') + opt('minmod', 'Minmod')), '', 'How steeply the reconstruction may vary within a cell: Koren is sharpest, Minmod most diffusive and most robust')}
    </div>
  </details>

  <details>
    <summary title="Radiative cooling and heating">Cooling<span class="sum" id="sum-cool"></span></summary>
    <div class="g2">
      ${field('Cooling function', select('cooling', (Object.keys(COOLING_LABELS) as CoolingKind[]).map((k) => opt(k, COOLING_LABELS[k])).join('')), '', 'Optically thin radiative cooling (and heating) of the gas')}
      ${field('Coefficient \u039b\u2080', number('crate', 'min="0"', 2), 'row-crate', 'Power law: the cooling rate per unit volume is \u03c1\u00b2 \u039b\u2080 T^\u03b1')}
      ${field('Exponent of T', number('calpha', 'step="0.25" min="-4" max="4"'), 'row-calpha', 'Power law: the exponent \u03b1 of T in the cooling rate (1/2 is bremsstrahlung)')}
      ${field('Heating rate', number('cheat', 'min="0"', 2), 'row-cheat', 'Heating per unit mass (power law), or the heating rate at unit density (interstellar)')}
      ${field('Kelvin per unit T', number('ctunit', 'min="0"', 2), 'row-ctunit', 'Interstellar: kelvin per code unit of temperature (code density is atoms per cm\u00b3)')}
      ${field('Temperature floor', number('ctfloor', 'min="0"', 2), 'row-ctfloor', 'Gas at or below this temperature does not cool')}
    </div>
  </details>

  <details>
    <summary title="Viscosity and thermal conduction">Diffusion<span class="sum" id="sum-diff"></span></summary>
    <div class="g2">
      ${field('Viscosity \u03bd', number('visc', 'min="0" data-start="0.0001"', 1.25), '', 'Kinematic viscosity: the stress is \u03c1 \u03bd times the shear. Obstacles hold the gas at their walls. 0 is off')}
      ${field('Conductivity \u03ba', number('cond', 'min="0" data-start="0.0001"', 1.25), '', 'Thermal conduction: the heat flux is -\u03ba \u03c1 grad T, so the temperature of an ideal gas diffuses at (\u03b3 - 1) \u03ba. 0 is off')}
    </div>
  </details>

  <details>
    <summary title="What happens to gas at the faces of the box">Boundaries<span class="sum" id="sum-bc"></span></summary>
    <div class="g2">
      ${['x', 'y', 'z'].map((ax, a) => (['lo', 'hi'] as const).map((side) =>
        field(`${ax} ${side === 'lo' ? 'lower' : 'upper'} face`, select(`bc-${side}-${a}`, BC_KINDS.map((k, i) => `<option value="${i}" title="${k.title}">${k.label}</option>`).join('')), `row-bc-${side}-${a}`,
          `What the gas meets at the ${side === 'lo' ? 'lower' : 'upper'} ${ax} face of the box`)).join('')).join('')}
      ${field('Damping from r', number('dampr', 'min="0" step="0.25"'), 'row-dampr', 'Radius from the box center where the damping zone starts. Beyond it the gas is relaxed toward the state it started in, more strongly toward the edge, so waves and gas leave quietly and the boundaries see the initial state')}
      ${field('Damping rate', number('damprate', 'min="0" step="0.5"'), 'row-damprate', 'How many times per local orbital period the gas at the box edge is relaxed toward its initial state. The rate rises from 0 at the inner radius. 0 is off')}
    </div>
  </details>

  <details>
    <summary title="Uniform gravity and self-gravity">Gravity<span class="sum" id="sum-grav"></span></summary>
    <div class="g3">
      ${field('Gravity', select('gravity', [
        ['off', 'Off', 'No gravity (point masses still pull)'],
        ['constant', 'Constant', 'A uniform field of strength G pointing down: -y in 2D, -z in 3D'],
        ['periodic', 'Self, periodic', 'Self-gravity of the gas in a periodic box (3D only)'],
        ['isolated', 'Self, isolated', 'Self-gravity of the gas, isolated: the potential falls off outside the box (3D only)'],
      ].map(([v, l, t]) => `<option value="${v}" title="${t}">${l}</option>`).join('')), '', 'Off, a uniform field, or the self-gravity of the gas. Point masses pull either way')}
      ${field('G', number('G', 'min="0"', 1.5), 'row-G', 'Gravitational constant, or the strength of the uniform field')}
      ${field('V-cycles', number('vcycles', 'step="1" min="1" max="8"'), '', 'Multigrid V-cycles per step for self-gravity. More converge the potential further, at more cost')}
    </div>
    <div class="g3 jeans">
      <label class="check" title="Pressure floor that keeps the Jeans length resolved (Truelove)"><input id="jeansfloor" type="checkbox" />Jeans floor</label>
      ${field('Cells per Jeans length', number('jeanscells', 'step="1" min="4" max="32"'), '', 'The Jeans floor raises the pressure so the Jeans length spans at least this many cells')}
    </div>
    <div class="checks">
      <label class="check" title="Flux-form gravitational work: conserves total energy"><input id="fluxwork" type="checkbox" />Conserve energy</label>
    </div>
  </details>

  <details open>
    <summary title="Totals and extremes over the box, read back five times a second. Hover a number for its whole run">Diagnostics</summary>
    <div id="diag"><dl></dl><dl></dl></div>
    <div class="io diag-io"><button id="export-diag" title="Save every diagnostics readout of this run (one row per readout) as a CSV file">Export diagnostics</button></div>
  </details>

  <footer><a href="validate.html" title="Run the test problems and compare them with exact solutions">Validation suite</a> &middot; <a href="https://github.com/guillochon/shaderous" target="_blank" rel="noopener" title="The code of this simulator">Source on GitHub</a><br>drag to orbit (pan in a slice), wheel or pinch to zoom, double-click to reset a slice, space to play or pause<br>with an editing tool picked, drag edits the gas and right-drag moves the view; Esc puts the tool away<br><span title="Build of the page you are looking at">build ${BUILD}</span></footer>
`;

// The color bar: the range at either end (editable), the opacity curve drawn on the bar, then
// its settings.
$('colorbar').innerHTML = `
  <input id="hi" class="cb-range" type="text" inputmode="decimal" spellcheck="false" title="Top of the displayed range" />
  <canvas id="tf" title="Color bar. In the volume view the line is the opacity: drag its points sideways, double-click to add or remove one"></canvas>
  <input id="lo" class="cb-range" type="text" inputmode="decimal" spellcheck="false" title="Bottom of the displayed range" />
  <div class="cb-checks">
    <label class="check" title="The range follows the data"><input id="auto" type="checkbox" checked />Auto</label>
    <label class="check" title="Logarithmic color scale"><input id="log" type="checkbox" />Log</label>
    <label class="check" title="Outline the blocks of an adaptive mesh in the slice view"><input id="blocks" type="checkbox" checked />Blocks</label>
  </div>
  <select id="cmap" title="Color map">${Object.keys(COLORMAPS).map((c) => opt(c, c)).join('')}</select>
  <label class="scrub" title="Opacity of the gas in the volume view"><input id="opacity" type="range" min="1" max="300" step="1" aria-label="Opacity" /><span>Opac<output id="opacity-out"></output></span></label>
`;

/** Show a preset in the problem and variant menus. */
function showProblem(id: string) {
  const fam = familyOf(id);
  $<HTMLSelectElement>('problem').value = fam.id;
  const variant = $<HTMLSelectElement>('variant');
  if (variant.dataset.family !== fam.id) {
    variant.dataset.family = fam.id;
    variant.innerHTML = fam.variants.map((v) => opt(v.problem.id, v.label)).join('');
  }
  variant.value = id;
  variant.disabled = fam.variants.length < 2;
}
// The page opens on the preset it is about to load, not on the first in the menu.
const firstLink = decodeLink(new URLSearchParams(location.search));
showProblem(firstLink.problem);
$('blurb').textContent = problemById(firstLink.problem).blurb;

// Fields with a factor: a click of the arrows (or an arrow key) multiplies or divides the value
// by it, rounded to three figures, instead of adding one. The browser takes its own step
// first; that tells the direction, and the value it started from is noted on the press.
{
  const from = new WeakMap<HTMLInputElement, number>();
  const factorOf = (t: EventTarget | null) => (t instanceof HTMLInputElement && t.dataset.factor ? +t.dataset.factor : 0);
  const note = (e: Event) => {
    if (factorOf(e.target)) from.set(e.target as HTMLInputElement, +(e.target as HTMLInputElement).value);
  };
  side.addEventListener('pointerdown', note, true);
  side.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') note(e);
  }, true);
  side.addEventListener('input', (e) => {
    const el = e.target as HTMLInputElement;
    const k = factorOf(el);
    // Typing comes with an input type; a step of the arrows does not.
    if (!k || (e as InputEvent).inputType) return;
    const v0 = from.get(el);
    // A field that is off at zero names the value its first step up goes to (data-start), and
    // steps back down to zero from there.
    const start = el.dataset.start ? +el.dataset.start : 0;
    // From zero or an empty field there is otherwise nothing to multiply: the browser's step stands.
    if (v0 === undefined || el.value === '' || (!(v0 > 0) && !start)) return;
    const dir = Math.sign(+el.value - v0);
    if (!dir) return;
    let v = +(v0 * k ** dir).toPrecision(3);
    if (start && !(v0 > 0)) v = dir > 0 ? start : 0;
    else if (start && dir < 0 && v0 <= start) v = 0;
    if (el.min !== '') v = Math.max(+el.min, v);
    if (el.max !== '') v = Math.min(+el.max, v);
    el.value = String(v);
    from.set(el, v);
  }, true);
}

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
// A column of tools under the color bar. Each paints with a soft ball (a disk in 2D) and
// keeps its own size and softness.
type ToolKind = Exclude<BrushOp['kind'], 'eos-out' | 'eos-in'> | 'source' | 'sink' | 'query';
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
  kind: 'source', label: 'Source',
  title: 'Source: click to place one that feeds gas in evenly, or press and drag to aim it (a longer drag is faster)',
  icon: icon('<circle cx="8" cy="8" r="2.3" fill="currentColor" stroke="none" /><path d="M8 1.5v2.3M8 12.2v2.3M1.5 8h2.3M12.2 8h2.3M3.4 3.4l1.6 1.6M11 11l1.6 1.6M12.6 3.4L11 5M5 11l-1.6 1.6" />'),
});
TOOLS.push({
  kind: 'sink', label: 'Sink',
  title: 'Sink: click to place one that takes gas away. Give it a mass and it becomes a point mass: it pulls on the gas, moves, and grows by what it takes in (press and drag to set it moving)',
  icon: icon('<circle cx="8" cy="8" r="5.5" stroke-dasharray="2.2 2.2" /><path d="M2.5 2.5l3 3M5.5 3v2.5H3M13.5 13.5l-3-3M10.5 13v-2.5H13" />'),
});
TOOLS.push({
  kind: 'query', label: 'Query',
  title: 'Query: hover over the gas to see the cell under the pointer: its state and, on an adaptive mesh, its block. Dragging moves the view',
  icon: icon('<circle cx="6.8" cy="6.8" r="4.3" /><path d="M10 10l4.5 4.5" /><path d="M6.8 4.9v.01M6.8 6.6v2.6" />'),
});
/** Point masses the sink tool has placed in this run (the simulation holds four in all). */
let placedMasses = 0;
/** The point masses as last read back, for their markers: refreshed every frame. */
let liveMasses: { of: Simulation; list: Diagnostics['masses'] } | null = null;
/** Why the last tool action did nothing, for the status line. */
let toolNotice = '';
const notice = (text: string) => {
  toolNotice = text;
};
/** Ways of drawing an obstacle. */
type ShapeMode = 'rect' | 'ellipse' | 'polygon';
const SHAPES: { mode: ShapeMode; title: string; icon: string }[] = [
  { mode: 'rect', title: 'Rectangle: drag from corner to corner. Shift makes a square; Alt draws from the center', icon: icon('<rect x="2.5" y="4" width="11" height="8" />') },
  { mode: 'ellipse', title: 'Ellipse: drag across it. Shift makes a circle; Alt draws from the center', icon: icon('<ellipse cx="8" cy="8" rx="5.5" ry="4" />') },
  {
    mode: 'polygon',
    title: 'Polygon: click each corner; a double-click, Enter or a click on the first corner closes it. Shift keeps edges at 45\u00b0 steps; Backspace takes the last corner back',
    icon: icon('<path d="M3 12.5L4.5 4l6.5-1.5 2.5 6.5-4.5 4z" />'),
  },
];
let shapeMode: ShapeMode = 'rect';
let obstacleErase = false;
/** `mass` is the sink tool's: the mass of the sink, as a multiple of the gas in the box. */
const toolParams: Record<ToolKind, { size: number; softness: number; amount: number; mass: number }> = {
  mass: { size: 0.06, softness: 0.5, amount: 0.25, mass: 0 },
  velocity: { size: 0.08, softness: 0.5, amount: 0, mass: 0 },
  temperature: { size: 0.06, softness: 0.5, amount: 0, mass: 0 },
  obstacle: { size: 0, softness: 0, amount: 0, mass: 0 },
  source: { size: 0.04, softness: 0.5, amount: 0.25, mass: 0 },
  sink: { size: 0.04, softness: 0.5, amount: 0.25, mass: 0 },
  query: { size: 0, softness: 0, amount: 0, mass: 0 },
};
/** The sink tool places a point mass rather than a plain sink. */
const sinkHasMass = () => toolParams.sink.mass > 0;
let tool: ToolKind | null = null;
/** Mass tool: new gas brings no momentum of its own. */
let massAtRest = true;
const slider = (id: string, label: string, title: string, min: number, max: number) =>
  `<label class="scrub" id="tp-${id}-row" title="${title}"><input id="tp-${id}" type="range" min="${min}" max="${max}" step="0.01" aria-label="${title}" /><span>${label}<output id="tp-${id}-out"></output></span></label>`;
$('tools').innerHTML =
  TOOLS.map((t) => `<button id="tool-${t.kind}" aria-pressed="false" aria-label="${t.label}" title="${t.title}">${t.icon}</button>`).join('') +
  `<div id="tool-params" hidden>
    ${slider('size', 'Size', 'Brush radius, as a share of the box', 0.01, 0.3)}
    ${slider('softness', 'Soft', 'Softness of the edge: 0 is hard, 1 fades all the way from the center', 0, 1)}
    ${slider('amount', 'Amt', 'Gas added per click, as a share of the densest gas', 0.01, 1)}
    ${slider('mass', 'Mass', 'Mass of the sink, as a multiple of the gas in the box. At 0 it only takes gas away, where it was placed. With a mass it is a point mass with softened gravity: it pulls on the gas, moves, and grows by what it takes in', 0, 2)}
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
  $('tool-params').hidden = tool === null || tool === 'query';
  if (tool !== 'query') hideCellPop();
  if (tool === null || tool === 'query') {
    $('brush').style.display = 'none';
    return;
  }
  const q = toolParams[tool];
  $('tp-amount-row').hidden = tool !== 'mass' && tool !== 'source' && tool !== 'sink';
  $('tp-mass-row').hidden = tool !== 'sink';
  // A point mass has no soft edge to set: its sink always fades from the center.
  $('tp-softness-row').hidden = tool === 'sink' && sinkHasMass();
  // The same slider is the amount of a dab of mass and the flux of a source or sink.
  const flux = tool === 'source' || tool === 'sink';
  $('tp-amount-row').title = tool === 'source'
    ? 'Flux: how fast the source feeds gas in. At 100% it adds the density of the densest gas in the time the fastest signal takes to cross it'
    : tool === 'sink'
      ? 'Flux: how fast the sink takes gas away. Without a mass, 100% empties it in the time the fastest signal takes to cross it; with one, 25% is the orbital rate at its edge'
      : 'Gas added per click, as a share of the densest gas';
  $('tp-amount-row').querySelector('span')!.firstChild!.textContent = flux ? 'Flux' : 'Amt';
  const mine = tool === 'sink' ? sources.filter((q) => q.decay > 0).length + placedMasses : sources.filter((q) => q.decay === 0).length;
  $('tool-clear').hidden = !flux;
  $('tool-clear').title = tool === 'sink' ? 'Remove every sink and point mass placed with this tool' : 'Remove every source';
  $('tool-clear').textContent = mine ? `Clear ${mine}` : 'Clear';
  $<HTMLButtonElement>('tool-clear').disabled = mine === 0;
  $('tool-rest').hidden = tool !== 'mass';
  $('tool-rest').setAttribute('aria-pressed', String(massAtRest));
  // An obstacle is drawn as a shape, not painted with a brush.
  const shapes = tool === 'obstacle';
  $('tp-size-row').hidden = shapes;
  if (shapes) $('tp-softness-row').hidden = true;
  $('tool-shapes').hidden = $('tool-erase').hidden = !shapes;
  for (const sh of SHAPES) $(`shape-${sh.mode}`).setAttribute('aria-pressed', String(shapeMode === sh.mode));
  $('tool-erase').setAttribute('aria-pressed', String(obstacleErase));
  if (shapes) drawBrush();
  else if (!stroke) $('brush').style.display = 'none'; // until the pointer next moves
  for (const k of ['size', 'softness', 'amount', 'mass'] as const) {
    const el = $<HTMLInputElement>(`tp-${k}`);
    el.value = String(q[k]);
    el.style.setProperty('--p', `${(100 * (q[k] - +el.min)) / (+el.max - +el.min)}%`);
    // Size is the brush radius and Amount a share of the densest gas, both shown in percent.
    $(`tp-${k}-out`).textContent = k === 'softness' ? q[k].toFixed(2) : k === 'mass' ? `${q[k].toFixed(2)}\u00d7` : `${Math.round(q[k] * 100)}%`;
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
  // Each tool clears what it placed.
  const sinks = tool === 'sink';
  const keep = sources.filter((q) => (q.decay > 0) !== sinks);
  if (keep.length !== sources.length) {
    sources = keep;
    sim?.setSources(sources);
  }
  if (sinks && placedMasses) {
    sim?.removeAddedMasses();
    placedMasses = 0;
    lastDiag = undefined;
  }
  syncTools();
});
$('tool-erase').addEventListener('click', () => {
  obstacleErase = !obstacleErase;
  syncTools();
});
for (const k of ['size', 'softness', 'amount', 'mass'] as const) {
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
/** Whether a run starts playing by itself when it loads or replays (remembered in this browser). */
let autoplay = true;
try {
  autoplay = localStorage.getItem('shaderous.autoplay') !== 'off';
} catch {}
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
/** Modifier keys, as last seen: Shift constrains a shape, Alt draws it from its center. */
const mods = { shift: false, alt: false };
/** The edit the next frame applies. */
let pendingBrush: BrushOp | null = null;

const editor = new OpacityEditor($<HTMLCanvasElement>('tf'), pushTransfer, true);

function pushTransfer() {
  renderer?.setTransfer(bakeTransfer(editor.colormap, editor.points));
}

/** The opacity control: its fill and its value. */
function syncOpacity() {
  const el = $<HTMLInputElement>('opacity');
  el.style.setProperty('--p', `${(100 * (+el.value - +el.min)) / (+el.max - +el.min)}%`);
  $('opacity-out').textContent = el.value;
}

const num = (v: number) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? v.toPrecision(5) : v.toExponential(2));

/** Reflect cfg and view into the controls. */
function syncControls() {
  const two = cfg.ndim === 2;
  showProblem(cfg.problem.id);
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
  for (const o of levels.options) o.disabled = +o.value > maxAmrLevels(cfg.n);
  $<HTMLInputElement>('blocks').checked = view.showBlocks;
  $<HTMLInputElement>('blocks').disabled = cfg.amrLevels === 0;
  // A 2D run is always shown as its plane, and has no self-gravity.
  $<HTMLSelectElement>('mode').disabled = two;
  $<HTMLSelectElement>('axis').disabled = two;
  $<HTMLInputElement>('slicepos').disabled = two;
  const gravity = $<HTMLSelectElement>('gravity');
  for (const o of gravity.options) o.disabled = two && isSelfGravity(o.value as GravityMode);
  const self = isSelfGravity(cfg.gravity);
  for (const id of ['vcycles', 'jeansfloor']) $<HTMLInputElement>(id).disabled = !self;
  // G is idle without gravity, unless point masses use it.
  const masses = !!cfg.problem.masses || placedMasses > 0;
  $<HTMLInputElement>('G').disabled = cfg.gravity === 'off' && !masses;
  $<HTMLInputElement>('refhi').value = String(cfg.refineHi);
  $<HTMLInputElement>('reflo').value = String(cfg.refineLo);
  $<HTMLInputElement>('refhi').disabled = $<HTMLInputElement>('reflo').disabled = cfg.amrLevels === 0;
  $('sum-grid').textContent = `${cfg.n}${two ? '\u00b2' : '\u00b3'}, ${cfg.amrLevels ? `${cfg.amrLevels} level${cfg.amrLevels > 1 ? 's' : ''}` : 'uniform'}`;
  $('sum-time').textContent = `${Number.isFinite(cfg.tEnd) ? `to t = ${+cfg.tEnd.toPrecision(6)}` : 'no end'}, CFL ${cfg.cfl}`;
  // Fields this run cannot show: a z velocity in 2D, heating without a cooling function, and
  // refinement levels on a uniform grid.
  const fieldSel = $<HTMLSelectElement>('field');
  // Nor a potential without gravity, nor a Jeans length without self-gravity.
  const pulled = cfg.gravity !== 'off' || cfg.gext.some((g) => g !== 0) || !!cfg.problem.masses || placedMasses > 0;
  const noField = (i: number) => (i === 7 && two) || (i === 10 && (cfg.cooling === 'off' || cfg.eos === 'barotropic')) || (i === 14 && cfg.amrLevels === 0)
    || (i === 15 && !pulled) || (i === 17 && !self);
  for (const o of fieldSel.options) o.disabled = noField(+o.value);
  if (noField(view.field)) view.field = 0;
  $<HTMLInputElement>('log').disabled = LINEAR_FIELDS.has(view.field);
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
  // Cooling acts on the internal energy, which a barotropic gas does not have.
  const canCool = cfg.eos !== 'barotropic';
  const cool = canCool ? cfg.cooling : 'off';
  $<HTMLSelectElement>('cooling').value = cool;
  $<HTMLSelectElement>('cooling').disabled = !canCool;
  $('sum-cool').textContent = canCool ? COOLING_LABELS[cool].toLowerCase() : 'needs an energy equation';
  const coolRows: Record<string, [keyof CoolingParams, boolean]> = {
    crate: ['rate', cool === 'power'], calpha: ['alpha', cool === 'power'], cheat: ['heat', cool !== 'off'],
    ctunit: ['tUnit', cool === 'ism'], ctfloor: ['tFloor', cool !== 'off'],
  };
  for (const [id, [key, shown]] of Object.entries(coolRows)) {
    $<HTMLInputElement>(id).value = String(cfg.coolParams[key]);
    $(`row-${id}`).style.display = shown ? '' : 'none';
  }
  // Conduction moves heat, which a barotropic gas does not carry.
  $<HTMLInputElement>('visc').value = String(cfg.viscosity);
  $<HTMLInputElement>('cond').value = String(cfg.conductivity);
  $<HTMLInputElement>('cond').disabled = !canCool;
  $('sum-diff').textContent = [cfg.viscosity > 0 ? 'viscosity' : '', conducts(cfg) ? 'conduction' : ''].filter(Boolean).join(' + ') || 'off';
  $<HTMLSelectElement>('riemann').value = cfg.riemann;
  $<HTMLSelectElement>('limiter').value = cfg.limiter;
  $<HTMLInputElement>('cfl').value = String(cfg.cfl);
  const tbins = $<HTMLSelectElement>('tbins');
  tbins.value = String(cfg.timeBins);
  tbins.disabled = cfg.amrLevels === 0;
  $<HTMLSelectElement>('gravity').value = cfg.gravity;
  $<HTMLInputElement>('G').value = String(+cfg.G.toPrecision(6));
  $<HTMLInputElement>('vcycles').value = String(cfg.vcycles);
  $<HTMLInputElement>('jeansfloor').checked = cfg.jeansFloor;
  $<HTMLInputElement>('jeanscells').value = String(cfg.jeansCells);
  $<HTMLInputElement>('jeanscells').disabled = !self || !cfg.jeansFloor;
  $<HTMLInputElement>('fluxwork').checked = cfg.gravEnergyFlux;
  $<HTMLSelectElement>('mode').value = two ? 'slice' : view.mode;
  $<HTMLSelectElement>('field').value = String(view.field);
  $<HTMLSelectElement>('axis').value = String(two ? 2 : view.sliceAxis);
  $<HTMLInputElement>('slicepos').value = String(view.slicePos);
  $<HTMLInputElement>('log').checked = view.log;
  $<HTMLInputElement>('opacity').value = String(view.opacity);
  // Opacity belongs to the volume view: a slice is drawn solid.
  const solid = two || view.mode === 'slice';
  $<HTMLInputElement>('opacity').disabled = solid;
  syncOpacity();
  editor.setEditable(!solid);
  for (let a = 0; a < 3; a++) {
    $<HTMLSelectElement>(`bc-lo-${a}`).value = String(cfg.bcLo[a]);
    $<HTMLSelectElement>(`bc-hi-${a}`).value = String(cfg.bcHi[a]);
  }
  // A 2D run has no z faces.
  $('row-bc-lo-2').style.display = $('row-bc-hi-2').style.display = two ? 'none' : '';
  const kinds = [...new Set([...cfg.bcLo, ...cfg.bcHi].filter((_, i) => !two || i % 3 !== 2))];
  // The damping zone belongs to presets that define one.
  const damps = !!cfg.problem.damping;
  $('row-dampr').style.display = $('row-damprate').style.display = damps ? '' : 'none';
  $<HTMLInputElement>('dampr').value = String(cfg.dampRadius);
  $<HTMLInputElement>('damprate').value = String(cfg.dampRate);
  $('sum-bc').textContent = (kinds.length === 1 ? BC_KINDS[kinds[0]].label.toLowerCase() : 'mixed') + (damps && cfg.dampRate > 0 ? ' + damping' : '');
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
    box.innerHTML = defs.map((q) => field(q.label, number(`pp-${q.id}`, `min="${q.min}" max="${q.max}"${q.step ? ` step="${q.step}"` : ''}`, q.factor), '', q.title ?? '')).join('');
    for (const q of defs) {
      const el = $<HTMLInputElement>(`pp-${q.id}`);
      el.addEventListener('change', () => {
        const v = el.value === '' || Number.isNaN(+el.value) ? cfg.params[q.id] : +el.value;
        cfg.params[q.id] = Math.min(q.max, Math.max(q.min, v));
        if (q.live && sim) {
          // Only the point masses' sizes: the run carries on.
          sim.cfg = cfg;
          sim.retuneMasses();
        } else {
          // Part of the initial conditions (an orbit, the gas's starting state): start again.
          restartRun(true);
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
      Object.assign(next.coolParams, link.coolParams);
      Object.assign(next.params, link.params);
      if (ndim === 2) {
        if (isSelfGravity(next.gravity)) next.gravity = 'off';
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
    // A freshly loaded problem starts playing, unless it waits for the user (an empty box); a
    // loaded state waits to be looked at.
    running = autoplay && !snap && !problem.startPaused;
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

/**
 * A restart or a rebuild of the shaders is under way: the loading chiclet shows. `shaders`: only
 * the shaders are being rebuilt, and the run carries on afterwards.
 */
let reloading: 'restart' | 'shaders' | null = null;
/** Resolves once the page has painted (so the loading chiclet is on screen before a long, blocking job). */
const painted = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

/**
 * Start the run again from its initial conditions, behind the loading chiclet: setting up a
 * large grid (and settling a star) takes a moment.
 */
function restartRun(replay = false) {
  return enqueue(async () => {
    if (!sim) return;
    reloading = 'restart';
    try {
      await painted();
      sim.cfg = cfg;
      sim.reset();
      clearSources();
      if (replay) running = autoplay;
      finished = false;
      lastDiag = undefined;
    } finally {
      reloading = null;
    }
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
    reloading = resetAfter ? 'restart' : 'shaders';
    try {
      await painted();
      sim.cfg = cfg;
      sim.writeParams();
      await sim.buildHydro();
      await renderer.attach(sim);
      if (resetAfter) {
        sim.reset();
        clearSources();
        lastDiag = undefined;
        if (finished) running = autoplay;
      }
      finished = false;
    } finally {
      reloading = null;
    }
  });
}

/**
 * A new equation of state, mid-run. The stored internal energy means something different under
 * the new one, so each cell's pressure is carried across instead: written out under the old
 * equation of state, the passes rebuilt, and read back in under the new one (see the brush
 * kinds `eos-out` and `eos-in`).
 */
function switchEos(kind: EosKind) {
  const setEos = () => {
    cfg.eos = kind;
    // An energy equation needs gamma above 1 (as the gamma field enforces).
    if (kind !== 'barotropic') cfg.eosParams.gamma = Math.max(1.001, cfg.eosParams.gamma);
  };
  return enqueue(async () => {
    if (!sim || sim.preparing) {
      // A star's balance was worked out for the old equation of state: settle it again.
      setEos();
      rebuildPipelines(true); // queued behind this job
      return;
    }
    reloading = 'shaders';
    try {
      await painted();
      const s = sim;
      const convert = (k: 'eos-out' | 'eos-in') => {
        const enc = gpu.device.createCommandEncoder();
        s.encodeBrush(enc, { kind: k, pos: [0.5, 0.5, 0.5], radius: 1, softness: 0, amount: 0, vel: [0, 0, 0] });
        // One pass per submit: the brush block is written through the queue.
        gpu.device.queue.submit([enc.finish()]);
      };
      convert('eos-out');
      setEos();
      s.cfg = cfg;
      s.writeParams();
      await s.buildHydro();
      await renderer.attach(s);
      convert('eos-in');
      s.refreshPotential();
      lastDiag = undefined;
      finished = false;
    } finally {
      reloading = null;
    }
  });
}

function bind() {
  const on = (id: string, fn: (el: HTMLInputElement & HTMLSelectElement) => void, ev = 'change') =>
    $(id).addEventListener(ev, () => fn($(id) as HTMLInputElement & HTMLSelectElement));
  const numOr = (el: HTMLInputElement, fallback: number) => (el.value === '' || Number.isNaN(+el.value) ? fallback : +el.value);

  const picked = () => ({
    id: $<HTMLSelectElement>('variant').value,
    n: +$<HTMLSelectElement>('res').value,
    ndim: (+$<HTMLSelectElement>('dim').value === 2 ? 2 : 3) as 2 | 3,
    levels: +$<HTMLSelectElement>('levels').value,
  });
  on('problem', (el) => {
    // A new problem opens on its first variant.
    showProblem(familyById(el.value).variants[0].problem.id);
    const { id, n, ndim, levels } = picked();
    build(id, n, ndim, levels);
  });
  on('variant', () => {
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
  $<HTMLInputElement>('autoplay').checked = autoplay;
  on('autoplay', (el) => {
    autoplay = el.checked;
    try {
      localStorage.setItem('shaderous.autoplay', autoplay ? 'on' : 'off');
    } catch {}
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
  // Space plays and pauses, unless a control has the keyboard (where it types, ticks or presses).
  window.addEventListener('keydown', (e) => {
    if (e.key !== ' ' || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.closest('input, select, textarea, button, [contenteditable]') || t.isContentEditable)) return;
    e.preventDefault();
    $('run').click();
  });
  $('step').addEventListener('click', () => {
    running = false;
    stepOnce = true;
    syncControls();
  });
  $('reset').addEventListener('click', () => {
    if (!sim) return;
    restartRun(finished); // replay a run that had reached its end time
    syncControls();
  });

  on('eos', (el) => {
    switchEos(el.value as EosKind);
    syncControls();
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
  on('cooling', (el) => {
    cfg.cooling = el.value as CoolingKind;
    // The cooling function is compiled into the update pass; the run carries on.
    rebuildPipelines().then(syncControls);
  });
  const coolInputs: [string, keyof CoolingParams][] = [['crate', 'rate'], ['calpha', 'alpha'], ['cheat', 'heat'], ['ctunit', 'tUnit'], ['ctfloor', 'tFloor']];
  for (const [id, key] of coolInputs) {
    on(id, (el) => {
      cfg.coolParams[key] = numOr(el, cfg.coolParams[key]);
      applyParams();
    });
  }
  // The diffusive fluxes are compiled in only while they are in use: switching one on or off
  // rebuilds the passes, a new strength is only a parameter. The run carries on either way.
  const diffusion = (set: (v: number) => void, el: HTMLInputElement, was: number) => {
    const before = diffusionKey(cfg);
    set(Math.max(0, numOr(el, was)));
    if (diffusionKey(cfg) === before) applyParams();
    else rebuildPipelines();
    syncControls();
  };
  on('visc', (el) => diffusion((v) => (cfg.viscosity = v), el, cfg.viscosity));
  on('cond', (el) => diffusion((v) => (cfg.conductivity = v), el, cfg.conductivity));
  // The damping zone's radius and rate are parameters: the run carries on.
  on('dampr', (el) => {
    cfg.dampRadius = Math.max(0, numOr(el, cfg.dampRadius));
    applyParams();
    syncControls();
  });
  on('damprate', (el) => {
    cfg.dampRate = Math.max(0, numOr(el, cfg.dampRate));
    applyParams();
    syncControls();
  });
  on('riemann', (el) => {
    cfg.riemann = el.value as RiemannKind;
    rebuildPipelines();
  });
  on('limiter', (el) => {
    cfg.limiter = el.value as LimiterKind;
    rebuildPipelines();
  });
  on('tbins', (el) => {
    cfg.timeBins = Math.min(3, Math.max(0, Math.round(+el.value) || 0));
    rebuildPipelines();
  });
  on('cfl', (el) => {
    cfg.cfl = Math.min(1, Math.max(0.05, numOr(el, cfg.cfl)));
    applyParams();
    syncControls();
  });
  // The refinement thresholds are read at every re-mesh: the run carries on. The merge
  // threshold stays below the refinement one.
  on('refhi', (el) => {
    cfg.refineHi = Math.min(1, Math.max(0.05, numOr(el, cfg.refineHi)));
    cfg.refineLo = Math.min(cfg.refineLo, cfg.refineHi / 2);
    applyParams();
    syncControls();
  });
  on('reflo', (el) => {
    cfg.refineLo = Math.min(cfg.refineHi / 2, Math.max(0, numOr(el, cfg.refineLo)));
    applyParams();
    syncControls();
  });
  const gravityChanged = () => {
    applyParams();
    sim?.refreshPotential();
  };
  on('gravity', (el) => {
    const was = cfg.gravity;
    cfg.gravity = el.value as GravityMode;
    // Turning gravity on starts from G = 1; turning it off leaves G as it was.
    if (was === 'off' && cfg.gravity !== 'off') cfg.G = 1;
    gravityChanged();
    syncControls();
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
  $('export-diag').addEventListener('click', () => exportDiagnostics());
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
          // On an adaptive mesh a newly periodic axis joins blocks that were never neighbors
          // and may differ by several levels: build the mesh again from the start.
          restartRun();
        } else applyParams();
        syncControls();
      });
    }
  }
  on('field', (el) => {
    view.field = +el.value;
    // A density gradient spans decades (shocks against smooth flow), so it reads best on a log
    // scale; refinement levels are a handful of integers.
    if (view.field === 13) view.log = true;
    if (view.field === 14) view.log = false;
    syncControls();
  });
  on('axis', (el) => (view.sliceAxis = +el.value));
  on('slicepos', (el) => (view.slicePos = +el.value), 'input');
  on('log', (el) => (view.log = el.checked));
  on('opacity', (el) => {
    view.opacity = +el.value;
    syncOpacity();
  }, 'input');
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
    if (tool && tool !== 'obstacle' && tool !== 'query' && sim && !building && e.button === 0 && pointers.size === 0 && !stroke) {
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
    hideCellPop();
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
    if (tool === 'query') {
      // While the view is being dragged the cell under the pointer keeps changing; wait.
      if (pointers.size === 0) queryAt(e.clientX, e.clientY);
      else hideCellPop();
    } else if (tool && pointers.size === 0 && !stroke) {
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

// --- query tool ---------------------------------------------------------------------------
// While it is in hand, the cell under the pointer is read back and shown beside it. One read is
// in flight at a time: the latest position is read next, and a running simulation is read again
// every 150 ms while the pointer rests.
const cellPop = $('cell-pop');
let queryPoint: { x: number; y: number } | null = null;
let queryBusy = false;

function queryAt(clientX: number, clientY: number) {
  queryPoint = { x: clientX, y: clientY };
  if (!queryBusy) void runQuery();
}

function hideCellPop() {
  queryPoint = null;
  cellPop.hidden = true;
}

async function runQuery() {
  const q = queryPoint;
  const s = sim;
  if (!q || !s || building) return;
  queryBusy = true;
  try {
    const r = canvas.getBoundingClientRect();
    const px = q.x - r.left;
    const py = q.y - r.top;
    let depth = view.dist;
    if (!flatView()) {
      // Volume view: the gas that is seen under the pointer.
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const [ux, uy] = [(2 * px) / w - 1, 1 - (2 * py) / h];
      const t = await renderer.pick(ux, uy);
      if (t === null) {
        cellPop.hidden = true;
        return;
      }
      const f = cameraFrame(view, w / h);
      depth = t / Math.hypot(1, ux * (w / h) * f.tan, uy * f.tan);
    }
    const f = locate(px, py, depth).pos;
    if (cfg.ndim === 2) f[2] = 0;
    if (f.some((v) => v < 0 || v >= 1)) {
      cellPop.hidden = true;
      return;
    }
    const c = await s.probe(f);
    if (s !== sim || !queryPoint || tool !== 'query') return;
    if (!c) cellPop.hidden = true;
    else showCell(c, queryPoint.x, queryPoint.y);
  } catch {
    cellPop.hidden = true;
  } finally {
    queryBusy = false;
    if (queryPoint && queryPoint !== q) void runQuery();
    else if (queryPoint && running) {
      setTimeout(() => {
        if (queryPoint && !queryBusy) void runQuery();
      }, 150);
    }
  }
}

function showCell(c: CellProbe, x: number, y: number) {
  const dims = cfg.ndim;
  const vec = (v: number[]) => `(${v.slice(0, dims).map((a) => fmt3(a)).join(', ')})`;
  const eos = makeEos(cfg.eos, cfg.eosParams);
  const rho = c.u1[0];
  const vel = [c.u1[1] / rho, c.u1[2] / rho, c.u1[3] / rho];
  const eint = c.u2[1];
  const p = eos.pressure(rho, eint);
  const cs = eos.soundSpeed(rho, p);
  const speed = Math.hypot(...vel.slice(0, dims));
  const rows: [string, string, string?][] = [
    ['Position', vec(c.pos)],
    ['Cell size', fmt3(c.dx)],
  ];
  if (c.block) {
    const b = c.block;
    rows.push(['Level', `${b.level} of ${cfg.amrLevels}`]);
    rows.push(['Block', `(${b.coords.join(', ')}), slot ${b.slot}`]);
    rows.push(['Indicator', b.indicator === null ? 'not yet read' : `${fmt3(b.indicator)} (split ${cfg.refineHi}, merge ${cfg.refineLo})`]);
  }
  if (c.u2[3] > 0.5) rows.push(['Obstacle', 'solid cell; gas held for later', 'solid']);
  rows.push(
    ['Density', fmt3(rho)],
    ['Velocity', vec(vel)],
    ['Speed', fmt3(speed)],
    ['Mach number', fmt3(cs > 0 ? speed / cs : 0)],
    ['Pressure', fmt3(p)],
    ['Temperature', fmt3(eos.temperature(rho, eint))],
    ['Sound speed', fmt3(cs)],
    ['Internal energy', `${fmt3(eint / rho)} per mass`],
    ['Total energy', `${fmt3(c.u2[0])} per volume`],
    ['Scalar', fmt3(c.u2[2] / rho)],
  );
  cellPop.querySelector('.title')!.textContent = `Cell (${c.index.join(', ')})`;
  cellPop.querySelector('dl')!.innerHTML = rows.map(([k, v, cls]) => `<dt>${k}</dt><dd${cls ? ` class="${cls}"` : ''}>${v}</dd>`).join('');
  cellPop.hidden = false;
  // Beside the pointer, kept on screen.
  const pw = cellPop.offsetWidth;
  const ph = cellPop.offsetHeight;
  const left = x + 18 + pw > window.innerWidth - 4 ? x - 18 - pw : x + 18;
  const top = Math.min(window.innerHeight - ph - 4, Math.max(4, y + 18));
  cellPop.style.left = `${Math.round(Math.max(4, left))}px`;
  cellPop.style.top = `${Math.round(top)}px`;
}

// --- obstacles ----------------------------------------------------------------------------
/** Center and half-sizes of the rectangle or ellipse being dragged out. */
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
  // Alt: the press is the center rather than a corner.
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
  placedMasses = 0;
  toolNotice = '';
}

/** Draw the markers of the sources and sinks where they are in the current view. */
let marksDrawn = '';
function drawSources() {
  let html = '';
  // Between resets and the first readback there is nothing to draw (lastDiag is cleared then).
  const live = liveMasses && liveMasses.of === sim ? liveMasses.list : null;
  const masses = (lastDiag && live ? live : lastDiag?.masses) ?? [];
  if ((sources.length || masses.length) && cfg) {
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
    // The point masses, where the last readback found them: the sink's edge and a dot.
    const box = cfg.problem.box;
    for (const q of masses) {
      const pos = q.pos.map((v) => v / box + 0.5);
      const c = at(pos);
      if (!(c.w > 0)) continue;
      const e = at(pos.map((v, i) => v + (q.sinkRadius / box) * locate(c.x, c.y, flatView() ? 1 : c.w).right[i]));
      // The sink's edge, at its true radius however small, and a dot at the mass.
      const r = Math.hypot(e.x - c.x, e.y - c.y);
      html += `<g class="sink point"><circle cx="${c.x}" cy="${c.y}" r="${r}" /><circle class="dot" cx="${c.x}" cy="${c.y}" r="${Math.min(2.5, Math.max(1, r / 2))}" /></g>`;
    }
  }
  if (html !== marksDrawn) $('marks').innerHTML = marksDrawn = html;
}

/** Outline of the brush under the pointer, and the arrow of a velocity stroke. */
function drawBrush(px?: number, py?: number) {
  const svg = $('brush');
  if (!tool || !cfg || tool === 'query') {
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
  const drag = stroke?.kind === 'velocity' || stroke?.kind === 'source' || (stroke?.kind === 'sink' && sinkHasMass());
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
    rhoMax: Math.max(d?.ranges[0][1] ?? 1, cfg.problem.toolScale?.rho ?? 0),
    vRef: Math.max(d?.ranges[4][1] ?? 0, 0.2 * cs, cfg.problem.toolScale?.v ?? 0),
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
  // A velocity is set on letting go; a canceled drag sets nothing.
  if (!apply && (stroke.kind === 'velocity' || stroke.kind === 'source' || stroke.kind === 'sink')) stroke = null;
  drawBrush();
}

/**
 * Place a point mass with the sink tool: at `pos` (in fractions of the box), moving at `vel`.
 * Its mass is the tool's multiple of the gas now in the box, its sink and its softening the
 * tool's radius, and its sink's rate the tool's flux (25% is the orbital rate at the edge).
 */
function placeMass(pos: Vec3, vel: Vec3, q: { size: number; amount: number; mass: number }) {
  if (!sim) return;
  const box = cfg.problem.box;
  // The gas in the box sets the scale; with no readback yet, a box of unit density does.
  const gas = lastDiag?.mass ?? box ** cfg.ndim;
  // Gravity needs a constant: a preset without one gets G = 1, shown in the Gravity section.
  if (!(cfg.G > 0)) cfg.G = 1;
  sim.cfg = cfg;
  const two = cfg.ndim === 2;
  const ok = sim.addMass({
    gm: cfg.G * q.mass * gas,
    pos: pos.map((v, i) => (two && i === 2 ? 0 : (v - 0.5) * box)),
    vel: two ? [vel[0], vel[1], 0] : vel,
    soft: q.size * box,
    sinkRadius: q.size * box,
    sinkRate: 4 * q.amount,
  });
  if (ok) {
    placedMasses++;
    lastDiag = undefined;
    syncControls();
  }
  notice(ok ? '' : `A run holds at most ${MAX_MASSES} point masses`);
}

/** Turn the stroke in progress into this frame's edit. Called once per frame. */
function advanceStroke(now: number) {
  const s = stroke;
  if (!s || !sim || !s.ready) return;
  const q = toolParams[s.kind];
  if (s.kind === 'source' || s.kind === 'sink') {
    if (!s.released) return;
    stroke = null;
    const sink = s.kind === 'sink';
    const massive = sink && sinkHasMass();
    const at = locate(s.x0, s.y0, s.depth);
    const [dx, dy] = [s.x - s.x0, s.y0 - s.y];
    const len = Math.hypot(dx, dy);
    // A drag aims a source, or sets a point mass moving; a click (or a drag of a few pixels)
    // leaves either at rest. A sink without a mass stays where it is put.
    const speed = (sink && !massive) || len < 6 ? 0 : Math.min(1, len / fullDrag()) * 5 * s.vRef;
    const vel = [0, 1, 2].map((i) => (speed ? (speed * (dx * at.right[i] + dy * at.up[i])) / len : 0)) as Vec3;
    if (massive) {
      placeMass(at.pos, vel, q);
      syncTools();
      return;
    }
    // The flux is measured against the densest gas and the time a signal takes to cross the source.
    const perTime = (q.amount * (s.vRef || 1)) / (q.size * cfg.problem.box);
    sources.push({ pos: at.pos, radius: q.size, softness: q.softness, vel, rate: sink ? 0 : perTime * s.rhoMax, decay: sink ? perTime : 0 });
    if (sources.length > MAX_SOURCES) sources.shift(); // the oldest makes room
    sim.setSources(sources);
    syncTools();
    return;
  }
  if (s.kind === 'query') return; // the query tool reads, and never starts a stroke
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
    if (s.kind === 'mass' && isSelfGravity(cfg.gravity) && cfg.ndim === 3) {
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
  // The first fields are ranged by the diagnostics reduction; the rest by the display pass,
  // over what it last drew.
  const r = renderer.range;
  const drawn = view.field >= d.ranges.length;
  if (drawn && (!r || r.field !== view.field)) return;
  let [lo, hi] = drawn ? [r!.lo, r!.hi] : d.ranges[view.field];
  if (drawn) {
    // A few extreme cells (inside a sink, at the corner of an obstacle) would set the whole
    // range: the top is the 99.5th percentile of |value| when that is lower.
    const top = renderer.quantile(view.field, 0.995);
    if (top !== null) [lo, hi] = [Math.max(lo, -top), Math.min(hi, top)];
  }
  const log = view.log && !LINEAR_FIELDS.has(view.field);
  if (SIGNED_FIELDS.has(view.field)) {
    // Centered on zero, so a diverging colormap shows the sign.
    const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-30);
    [lo, hi] = [-m, m];
  }
  if (!(hi > lo)) hi = lo + Math.max(Math.abs(lo) * 1e-3, 1e-12);
  if (log) {
    if (drawn && r!.pos !== null) lo = Math.max(lo, r!.pos);
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

// [label, tooltip, CSV column, a count]; the first seven fill the left column, the rest the right.
const DIAG_ROWS: [string, string, string, boolean?][] = [
  ['t', 'Time', 't'], ['dt', 'Timestep', 'dt'], ['steps', 'Steps taken', 'steps', true], ['mass', 'Total mass', 'mass'],
  ['Mach', 'Maximum Mach number', 'max_mach'], ['floor', 'Cells at the density or pressure floor (emptied voids, usually)', 'floored_cells', true],
  ['Jeans', 'Cells whose thermal Jeans length is under 4 cells. With the Jeans floor on these are the cells it is holding up', 'jeans_unresolved_cells', true],
  ['p x', 'Total x momentum', 'p_x'], ['p y', 'Total y momentum', 'p_y'], ['p z', 'Total z momentum', 'p_z'],
  ['E kin', 'Kinetic energy', 'e_kin'], ['E int', 'Internal energy', 'e_int'], ['E grav', 'Gravitational energy', 'e_grav'], ['E tot', 'Total energy', 'e_tot'],
];
/** The diagnostics in DIAG_ROWS order. */
const diagValues = (d: Diagnostics): number[] => [
  d.t, d.dt, d.steps, d.mass, d.maxMach, d.floored, d.jeansUnresolved, ...d.momentum, d.eKin, d.eInt, d.eGrav, d.eTot,
];
const diagHistory = new DiagHistory(2);
/** The run the history belongs to. */
let diagOf: Simulation | undefined;
const diagCols = $('diag').querySelectorAll('dl');
const diagPop = $('diag-pop');
const diagPopCanvas = diagPop.querySelector('canvas')!;
const diagPopTitle = diagPop.querySelector('.title')!;
/** Row whose whole-run plot is showing, if any. */
let diagHover = -1;
const diagCells = DIAG_ROWS.map(([label, title], i) => {
  const dt = document.createElement('dt');
  dt.textContent = label;
  dt.title = title;
  const dd = document.createElement('dd');
  const spark = document.createElement('canvas');
  spark.className = 'spark';
  diagCols[i < 7 ? 0 : 1].append(dt, dd, spark);
  for (const el of [dd, spark]) {
    el.addEventListener('pointerenter', () => showDiagPlot(i, dd));
    el.addEventListener('pointerleave', () => hideDiagPlot());
  }
  return { dd, spark, set: liveText(dd), warn: false };
});
const setHud = liveText(hud);

/** The popup with row i's whole run, beside its number. */
function showDiagPlot(i: number, anchor: HTMLElement) {
  diagHover = i;
  diagPop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const pw = diagPop.offsetWidth;
  const ph = diagPop.offsetHeight;
  // To the right of the side panel where there is room, otherwise above the row.
  const side = $('side').getBoundingClientRect();
  let x = side.right + 8;
  let y = r.top + r.height / 2 - ph / 2;
  if (x + pw > window.innerWidth - 4) {
    x = Math.min(window.innerWidth - pw - 4, Math.max(4, r.right - pw));
    y = r.top - ph - 6;
    if (y < 4) y = r.bottom + 6;
  }
  diagPop.style.left = `${Math.round(x)}px`;
  diagPop.style.top = `${Math.round(Math.min(window.innerHeight - ph - 4, Math.max(4, y)))}px`;
  drawDiagPlot();
}
function hideDiagPlot() {
  diagHover = -1;
  diagPop.hidden = true;
}
function drawDiagPlot() {
  if (diagHover < 0) return;
  const [label, title, , count] = DIAG_ROWS[diagHover];
  const rows = diagHistory.rows;
  const now = rows.length ? fmt3(rows[rows.length - 1][diagHover], count) : '';
  diagPopTitle.textContent = `${title} (${label})${now ? ` · ${now}` : ''}`;
  const [xs, ys] = diagHistory.all(diagHover);
  drawFull(diagPopCanvas, xs, ys, !!count);
}

function showDiagnostics(d: Diagnostics, of: Simulation) {
  if (of !== diagOf) {
    diagHistory.clear();
    diagOf = of;
  }
  const values = diagValues(d);
  diagHistory.push(values);
  const warns = [false, false, false, false, false, d.floored > 0, d.jeansUnresolved > 0 && !cfg.jeansFloor];
  values.forEach((v, i) => {
    const cell = diagCells[i];
    cell.set(fmt3(v, DIAG_ROWS[i][3]));
    const warn = warns[i] ?? false;
    if (warn !== cell.warn) {
      cell.warn = warn;
      cell.dd.classList.toggle('warn', warn);
    }
    const [xs, ys] = diagHistory.recent(i);
    drawSpark(cell.spark, xs, ys);
  });
  drawDiagPlot();
}

/** Save the diagnostics of the run so far as CSV. */
function exportDiagnostics() {
  const csv = diagHistory.csv(DIAG_ROWS.map((r) => r[2]));
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}-diagnostics.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
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
const loadingEl = $('loading');
const loadingText = $('loading-text');

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
  // Loading: no simulation yet, a restart or rebuild under way, or a star still settling.
  const loading = !sim || !!reloading || sim.preparing;
  if (loadingEl.hidden === loading) loadingEl.hidden = !loading;
  if (loading) {
    const text = reloading === 'shaders' && !sim?.preparing ? 'Recomputing shaders…' : 'Loading simulation…';
    if (loadingText.textContent !== text) loadingText.textContent = text;
  }
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
  renderer.submitted();

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

  // The masses move every step: their markers follow from a readback of their own, each frame.
  if (sim) {
    const of = sim;
    of.readMasses().then((list) => {
      if (of !== sim) return;
      if (list) liveMasses = { of, list };
      else if (!lastDiag?.masses.length) liveMasses = null;
    });
  }

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
      showDiagnostics(d, s);
    });
    if (now - rateMark.time > 500) {
      stepRate = ((stepsSubmitted - rateMark.steps) * 1000) / (now - rateMark.time);
      rateMark = { time: now, steps: stepsSubmitted };
    }
    const pm = lastDiag?.masses ?? [];
    const acc = pm.reduce((a, m) => a + m.accreted, 0);
    const where = pm.length === 1
      ? `  •  point mass at r = ${Math.hypot(...pm[0].pos).toFixed(2)}`
      : pm.length ? `  •  ${pm.length} point masses, separation ${Math.hypot(...pm[0].pos.map((x, i) => x - pm[1].pos[i])).toFixed(2)}` : '';
    const took = acc > 0 ? `, accreted ${acc.toPrecision(3)}` : '';
    const state = finished ? 'finished' : running ? `${nsub} step${nsub > 1 ? 's' : ''}/frame, ${stepRate.toFixed(0)} steps/s` : 'paused';
    const mesh = s.status();
    setHud(`${cfg.n}${cfg.ndim === 2 ? '²' : '³'}  •  ${fps.toFixed(0)} fps  •  ${state}${where}${where ? took : ''}${mesh ? '  •  ' + mesh : ''}${toolNotice ? '  •  ' + toolNotice : ''}`);
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
    await build(firstLink.problem, firstLink.n, firstLink.ndim, firstLink.amr, false, firstLink);
    requestAnimationFrame(tick);
  } catch (e) {
    fail(e);
  }
})();

// Handles for the console and for tests.
Object.assign(window, { shaderousApp: { get sim() { return sim; }, get cfg() { return cfg; }, view, get diag() { return lastDiag; }, get nsub() { return nsub; }, get stepRate() { return stepRate; }, get fps() { return fps; }, get pollMs() { return pollMs; } } });

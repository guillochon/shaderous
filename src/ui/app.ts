// Interactive front end: controls, frame loop with adaptive substepping, diagnostics panel.
import { BUILD, reportError, setAdapterInfo } from './errors';
import { decodeLink, DEFAULT_SETUP, encodeLink, snapshotSettings, viewDefaults, MAX_CONTOURS, MAX_DIST, MAX_ZOOM, MAX_ZOOM_1D, MIN_DIST, type LinkOverrides, type LinkProfiles } from './link';
import { initGpu, type Gpu } from '../gpu/device';
import { COMPILE_STAGE, INIT_STAGE, compileProgress, endLoadStage, forgetCompileSetup, loadStage, resetCompileCount, setLoadStage, watchLoading } from '../gpu/progress';
import { allocationSurvived, autoBudget, GB, lastCrash, memoryBudget, setAdapter, setUserBudget, userBudget, forgetCrashes } from '../gpu/memory';
import { frameTiming, initTiming, setTiming, timingAvailable, type TimingReport } from '../gpu/timing';
import { capture, restartFrom } from '../io/capture';
import { canPickFolder, checkpointIndex, download, downloadBlob, downloadSink, nextCheckpoint, pickFolder, type CheckpointSink } from '../io/checkpoints';
import { maxLevel, type Snapshot } from '../io/snapshot';
import { AmrSim } from '../amr/amr_sim';
import { bakeVariables, fileSpeciesGroups, planView, viewBakedBlockBytes, viewBlockBytes, viewFacesBlockBytes, type ViewBatch } from '../io/view';
import type { ViewFile } from '../io/view_open';
import { DiagHistory, drawFull, drawSpark, fmt3 } from './diag_plot';
import { fieldMeasure, readout, type Measure, type Readout } from './units_view';
import { unitOf, type DisplaySystem } from '../units/units';
import { ASTRO } from '../units/constants';
import { dipoleField, randomField, type StarProfile } from '../mhd/star_fields';
import { initialFaces } from '../mhd/mhd';
import { blockFaces } from '../mhd/amr_mhd';
import type { Block } from '../amr/tree';
import { boxFrac, fracPos, gridDims } from '../domain';
import { clipSegment, lockLine, distanceAt, distances, drawProfilePlot, MAX_PROFILES, plotLength, PROFILE_COLORS, profileCsv, profileDash, resolveField, sampleCount, sampleNear, samplePoints, seriesRange } from './profile';
import { radiationOn, syncRadiationEos } from '../radiation/radiation';
import { cellEquations, importance, rampColor, type Equation } from './terms';
import { opacity as meanOpacity } from '../opacity/opacity';
import { type CellProbe, configFor, refusal, unitConstants, withSymmetry, diffusionKey, MAX_MASSES, MAX_OUTLINE_VERTICES, MAX_SOURCES, type BrushOp, starBrush, burnSpec, type Diagnostics, type Outline, Sim, type SimConfig, type Simulation, type SolveStats, pressureNote, rotationOf, type SourceSpec, cosmologyOf, cylindrical } from '../sim';
import { expandSpecies, switchedSetup } from '../burn/watch';
import { speciesLabel } from '../species/species';
import { mixMoments, problemCoefficients, speciesNuclides } from '../species/mixture';
import { burnRates } from './burn_rates';
import { ReadoutView, type Row } from './readout_view';
import { BurnGraphView, cellReactions } from './burn_graph';
import { hubble } from '../cosmo';
import { amrFits, createSim, fieldShaderFor, gridBytes, maxAmrLevels, minAmrLevels, supportsAmr } from '../create';
import { laneEmdenCached, polytropeMass } from '../problems/lane_emden';
import { defaultFrameDt, MAX_BODIES, sizedProblem } from '../problems/types';
import { EOS_LABELS, eosHasEnergy, makeEos, physicalEosParams, type Eos, type EosKind } from '../eos/eos';
import { prepareEosTable } from '../eos/tables';
import { familyById, FIELD_MENU, MAGNETIC_FIELDS, NUCLEAR_HEATING_FIELD, RADIATION_FIELDS, FIELD_TIPS, FIELDS, isSelfGravity, LINEAR_FIELDS, MASS_FRACTION_FIELD, PERIODIC, SIGNED_FIELDS, MENU, type Problem, type ProblemParam, problemUnits } from '../problems';
import { curvedView, sphFromView, sphToView, spherical } from '../spherical';
import { backgroundCss, cameraFrame, defaultView, mirrorComputed, mirrorShown, Renderer, setFrameTurn, type ProfileValues, type ViewState } from '../render/renderer';
import { MAX_FIELD_LINES } from '../render/fieldlines';
import { fitLumps, previewOrbit, type Lump, type Orbit } from '../gravity/preview';
import type { MhdSpec, PointMass } from '../problems/types';
import { bakeTransfer, COLORMAPS, defaultOpacity, OpacityEditor } from '../render/transfer';
import { installApi, type RecordingStatus } from './api';
import { PRESET_FILES, presetSpec } from '../presets/index';
import { CUSTOM_ID, customSetup, customVersion, setCustomSetup } from '../presets/custom';
import { ensureMeshFiles } from '../geometry/store';
import { familyFor, problemOf } from '../presets/lookup';
import { decodeSetupParam, encodeSetupParam, formatSetup, MAX_LINK, readSetup } from '../setup/codec';
import { compiledFrom, isParamPath, paramFeatures } from '../setup/compile';
import { optionAt, valueAt } from '../setup/options';
import { changeBetween, runningSpec, sameSetup, setSetting, type Setting } from '../setup/running';
import { SessionRecorder, base64ToF32, planeOutline, pullOps, pullStep, queryDiff, type Action, type MassRole, type PlaneShape, type SessionLog, type StarAction } from './session';
import { cloneSetup, featuredId, featuredOf, featuredPaths, getPath, setPath, type SetupSpec } from '../setup/spec';
import { openSetupDialog } from './setup/dialog';
import { showShare } from './share';
import { installSetupSearch } from './setup_search';
import type { SetupHost } from './setup/host';
import { getSetup, listSetups } from './setup/library';

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

// 3D grids past 256^3 are adaptive only (UNIFORM_MAX): the mesh's memory goes by its blocks,
// not by n, so a finer finest level costs only where the setup refines to it. At 4096^3 the
// leaf index (one entry per finest block, amr/amr_sim.ts) is 64 MB, counted by gridBytes.
const RES_3D = [32, 64, 128, 256, 512, 1024, 2048, 4096];
const RES_2D = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];
// 1D runs (plans/one-dimensional.md): a row of cells, uniform up to 2048 (UNIFORM_MAX, a 3D
// texture's side), adaptive beyond.
const RES_1D = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
const resFor = (ndim: 1 | 2 | 3) => (ndim === 1 ? RES_1D : ndim === 2 ? RES_2D : RES_3D);
/**
 * A 1D run's chip (plans/one-dimensional.md): what the view repeats, and why, or null for a run
 * that resolves its plane. The chip over the view and on screenshots and movies says this.
 */
function oneDChip(c: SimConfig | undefined): { text: string; title: string } | null {
  if (!c || c.ndim !== 1) return null;
  const g = c.problem.geometry;
  if (g === 'spherical') return { text: 'spherically symmetric', title: 'A 1D run: only r is simulated. The view repeats each radius around the polar angle, as rings' };
  if (g === 'cylindrical') return { text: 'cylindrically symmetric', title: 'A 1D run: only R is simulated. The view repeats each radius around the polar angle, as rings' };
  return { text: 'repeated along y', title: 'A 1D run: only x is simulated. The view repeats the row of cells along y, which is not part of the run' };
}
/** The deepest the slice view zooms: a 1D run's down to some 16 cells across (MAX_ZOOM_1D). */
function maxZoom(): number {
  return cfg?.ndim === 1 ? Math.min(MAX_ZOOM_1D, Math.max(MAX_ZOOM, cfg.n / 16)) : MAX_ZOOM;
}
/** The grid's size as the page writes it: 512², 128³, or 4096 cells in 1D. */
const gridLabel = (n: number, ndim: 1 | 2 | 3) => (ndim === 1 ? `1D, ${n} cells` : `${n}${ndim === 2 ? '²' : '³'}`);
// Largest uniform grids offered; beyond these the mesh must be adaptive. A phone, or a device
// that crashed before, offers only the ones within its memory budget (gpu/memory.ts).
// (A uniform 1D grid past a 3D texture's side runs on blocks with no levels: sim.ts onBlocks.)
const UNIFORM_MAX = { 1: 262144, 2: 2048, 3: 256 };
function uniformMax(ndim: 1 | 2 | 3): number {
  const fits = resFor(ndim).filter((n) => n <= UNIFORM_MAX[ndim] && n ** ndim * Sim.cellBytes(ndim) <= memoryBudget());
  return fits.length ? fits[fits.length - 1] : resFor(ndim)[0];
}

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
    <div class="f"><span>Preset</span>
    <div class="pick-row">${select('problem', MENU.map((g) => `<optgroup label="${g.group}">${g.families.map((f) => opt(f.id, f.label)).join('')}</optgroup>`).join('') + '<optgroup label="Your setups" id="own-setups" hidden></optgroup>', 'The preset to simulate, or a setup of your own: picking one starts it from its initial conditions')}
    <button id="problem-search" type="button" title="Search the presets and their variants" aria-label="Search setups"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="6.8" cy="6.8" r="4.3" /><path d="M10.1 10.1L14 14" /></svg></button></div>
    </div>
    <div class="g3">
      ${field('Variant', select('variant', ''), 'row-variant', 'Versions of this problem with different setups')}
      <span id="pparams"></span>
      <span id="pfeatured"></span>
    </div>
    <p class="blurb" id="blurb"></p>
    <div id="setup-sum" title="The grid and the gas of this run"></div>
    <div class="setup-row"><button id="setup" title="Every setting of the simulation: the grid, the gas, gravity, the boundaries, the solver and the objects in the box. Presets and your own setups are files that fill it in">Setup…</button><button id="run" title="Run or pause the simulation (space)">Pause</button></div>
    <div class="buttons"><button id="step" title="Pause and take a single step">Step</button><button id="reset" title="Start the run again from its initial conditions">Reset</button><button id="record" hidden title="Record the run from here as an MP4 movie (settings under Movie); press again to stop and save it. The run is 2-3 times slower while recording">Record</button><button id="share" title="Copy a link that opens the simulator with these settings">Share</button><button id="load" hidden title="Autoload is off: load the changes that wait (problem, grid, initial conditions), paused. Run loads them and plays">Load</button></div>
    <div class="io">
      <label class="check" title="Start each problem playing as soon as it loads, and play a finished run again on Reset. Off, a run waits for Play"><input id="autoplay" type="checkbox" />Autoplay</label>
      <label class="check" title="Load the simulation as soon as a setting that needs a new one changes: the problem, the grid, a parameter of the initial conditions. Off, such changes wait until you press Run, so several can be made before a large grid is set up once"><input id="autoload" type="checkbox" />Autoload</label>
    </div>
    <div class="io state-io">
      <button id="import" title="Load a state from an .athdf file: one saved here, or another code's with density, velocity and pressure">Import state</button>
      <button id="export" title="Save the state of the gas as an HDF5 file in the Athena++ layout (.athdf), readable by yt and VisIt">Export state</button>
      <input id="import-file" type="file" accept=".athdf,.h5,.hdf5,.hdf" hidden />
    </div>
    <div class="io state-io">
      <button id="view-file" title="Open an .athdf checkpoint only to look at it, for runs too large to continue on this device: the GPU holds what the display reads (or only the field shown), and if even that does not fit, its finest levels are merged into coarser ones">Open for viewing</button>
      <input id="view-file-input" type="file" accept=".athdf,.h5,.hdf5,.hdf" hidden />
      <button id="save-log" title="Save what was done to this run since it started (the editing tools, the settings changed, the adaptive mesh's re-meshes, each at the step it happened) as a session log, with the run's diagnostics along the way. The API's replay() runs it again step for step">Save session log</button>
    </div>
  </section>

  <details>
    <summary title="When the run stops, and checkpoints along the way">Time<span class="sum" id="sum-time"></span></summary>
    <div class="g2">
      ${field('End time', number('tend', 'placeholder="none" min="0"', 1.25), '', 'The run pauses when it reaches this time. Empty runs on without end')}
    </div>
    <div class="g2 ckpt">
      <label class="check" title="Save the state as an .athdf file (as Export state does) every time the clock reaches a multiple of the interval. The run stops on each of those times exactly while the file is written. You pick a folder for them first; a browser that cannot write to a folder downloads each one"><input id="ckpt" type="checkbox" />Save checkpoints</label>
      ${field('Checkpoint every', number('ckdt', 'min="0"', 2), '', 'Time between checkpoints, in code time units. Each problem starts with an interval of its own')}
    </div>
    <div id="ckpt-note"></div>
  </details>

  <details id="movie">
    <summary title="Record the run as an MP4 movie with the Record button, or save the view as a PNG with Screenshot">Movie and screenshot<span class="sum" id="sum-movie"></span></summary>
    <div class="g3">
      ${field('Frame', select('rec-cadence', opt('time', 'Every Δt') + opt('steps', 'Every n steps')), '', 'When a frame is taken: after every so many steps, or near every multiple of a time interval. A time interval never shortens a step: each frame is taken at the step nearest its time, so the recorded run is the run it would have been')}
      ${field('Steps', number('rec-every', 'min="1" step="1" value="1"'), 'row-rec-every', 'A movie frame after every this many steps. 1 records every step')}
      ${field('Interval', number('rec-dt', 'min="0" value="0.01"', 2), 'row-rec-dt', 'A movie frame near every multiple of this much code time. Each problem starts with a 600th of its dynamical time, so that one dynamical time plays in 10 s at 60 fps')}
    </div>
    <div class="g3">
      ${field('Playback', select('rec-fps', opt(60, '60 fps') + opt(30, '30 fps') + opt(24, '24 fps')), '', 'Frames per second of the movie when played: sets its length, not which steps are recorded')}
      ${field('Size', select('rec-size', opt('view', 'As the view') + opt('1280x720', '720p') + opt('1920x1080', '1080p') + opt('2560x1440', '1440p') + opt('3840x2160', '4K')), '', 'Pixels of each frame. As the view records the box as the page draws it, at its size and framing, with a small margin and room beside it for the axes and the color bar: not the empty sides of a wide window')}
      ${field('Bitrate', select('rec-bitrate', opt('auto', 'Auto') + [4, 8, 16, 32, 64, 100].map((m) => opt(m, `${m} Mbit/s`)).join('')), '', 'Megabits per second of movie: more gives a cleaner picture and a larger file, about 7.5 MB per minute for each Mbit/s. Auto takes 0.2 bits per pixel per frame (25 Mbit/s at 1080p and 60 fps), enough that smooth color gradients do not band')}
    </div>
    <div class="io">
      <label class="check" title="A color bar with the field, its unit and its range over the right edge of each frame. With Auto range on its labels follow the data; a fixed range gives a steadier movie"><input id="rec-colorbar" type="checkbox" checked />Color bar</label>
      <label class="check" title="The axes x, y and z in the lower left corner of each frame, turning with the camera"><input id="rec-triad" type="checkbox" checked />Axes</label>
      <label class="check" title="The status line in the upper left corner of each frame: the grid, the simulation time, point masses and the mesh"><input id="rec-status" type="checkbox" checked />Status</label>
    </div>
    <div class="io shot">
      <button id="shot" title="Save the view as a PNG. A slice, a projection or a 2D run is saved at the grid's own resolution: one pixel per cell of the finest level, with the color bar and axes beside it. The volume view is saved at its size on the page, with them over it">Screenshot</button>
      ${select('shot-region', opt('box', 'Whole box') + opt('view', 'As the view'), 'Which part of the plane the screenshot takes: the whole box, or the part of it in view (zoom and pan), rounded out to whole cells')}
    </div>
  </details>

  <details open>
    <summary title="How the gas is drawn">View<span class="sum" id="sum-view"></span></summary>
    <div class="g2">
      ${field('Mode', select('mode', opt('volume', 'Volume') + opt('slice', 'Slice') + opt('projection', 'Projection')), '', 'Volume renders the gas as a glowing fog seen through the box; Slice shows one plane of cells; Projection looks through the whole box along an axis: density summed into a column density, any other field averaged along the line of sight, weighted by mass')}
      ${field('Field', select('field', FIELD_MENU.map((g) => `<optgroup label="${g.group}">${g.fields.map((i) => `<option value="${i}" title="${FIELD_TIPS[i]}">${FIELDS[i]}</option>`).join('')}</optgroup>`).join('')), '', 'The quantity the colors show. Fields of either sign (velocity components, heating, vorticity, divergence) are shown on a linear scale centered on zero')}
      ${field('Units', select('units', opt('cgs', 'cgs') + opt('si', 'SI') + opt('code', 'Code')), '', 'Units the readouts are shown in: the color bar, the query tool and the diagnostics. Only a setup that stands for something real (a star, say) has physical units; the others are in code units')}
      ${field('Slice normal', select('axis', opt(2, 'z') + opt(1, 'y') + opt(0, 'x')), 'row-axis', 'The axis the slice is perpendicular to')}
      ${field('Slice position', '<input id="slicepos" type="range" min="0" max="0.999" step="0.001" />', 'row-slicepos', 'Where the slice cuts the box along its normal')}
      ${field('Contours', number('ncont', `min="1" max="${MAX_CONTOURS}" step="1"`), 'row-ncont', 'How many contour lines the slice and projection views draw, or isosurfaces the volume view draws, when Contours is on (beside the color bar): evenly spaced in log |value| between the ends of the displayed range, the ends themselves left out. A range reaching zero or below is contoured in |value| over the four decades under its largest magnitude')}
      ${field('Turn with frame', '<input id="turnframe" type="checkbox" />', 'row-turnframe', 'Volume view of a run in a rotating frame: turn the box at the frame’s rate, so the picture is the one a camera at rest sees (a rotor turning and its wake left behind, rather than the rotor standing still)')}
      ${field('Surface opacity', '<input id="isoa" type="range" min="0" max="1" step="0.005" />', 'row-isoa', 'Volume view, with Contours on: how opaque each isosurface is. Below 1 the surfaces behind show through')}
      ${field('Field lines', number('nflines', `min="1" max="${MAX_FIELD_LINES}" step="1"`), 'row-nflines', 'A run with a magnetic field, with Field lines on (beside the color bar): how many lines. In 2D they are contours of the flux function, this many levels: the same magnetic flux passes between any two, so they crowd where the field is strong. In 3D, this many lines traced from seeds that move with the gas, so each stays the same line as the field evolves: thin tubes in the volume view, lines along the field in the plane in the slice view. Colored by the magnetic pressure B^2/2 on a log scale over up to six decades below the strongest, as Guillochon & McCourt (2017) drew theirs: red weak, then yellow, green, cyan, blue, magenta strong')}
    </div>
  </details>

  <details open>
    <summary title="Totals and extremes over the box, read back five times a second. Hover a number for its whole run">Diagnostics</summary>
    <div id="diag"><dl></dl><dl></dl></div>
    <div id="diag-sinks" hidden></div>
    <div class="io diag-io"><button id="export-diag" title="Save every diagnostics readout of this run (one row per readout) as a CSV file, with each sink’s mass, accretion rate, accreted momentum and angular momentum, position and velocity">Export diagnostics</button></div>
  </details>

  <details id="perf">
    <summary title="GPU memory for the simulation, and the GPU time of each of its passes">Performance</summary>
    <div class="g2 gpumem">
      ${field('GPU memory (GB)', number('gpumem', 'step="0.25" min="0.25" inputmode="decimal"'), '', 'Most GPU memory the simulation may take: it sets how many blocks the adaptive mesh can hold and which uniform grids are offered. Browsers do not say how much there is, so it is guessed: 2.5 GB on a desktop, 2 GB where the GPU shares the system’s memory, 1 GB on a phone or tablet, at most a third of the system’s memory where the browser reports it, and lower after the page has crashed. Set more if your GPU can take it. Changing it starts the run again; a crash goes back to the automatic value')}
      <button id="gpumem-auto" type="button" title="Go back to the automatic value, as guessed for this device: a value set here, or one a crash lowered, is dropped">Auto</button>
    </div>
    <div id="gpumem-note"></div>
    <label class="check" title="Time every pipeline of the simulation on the GPU. Each run of one pipeline then gets its own pass, which costs a few percent while this is on and nothing while it is off. Chrome rounds timestamps to 100 µs unless its WebGPU Developer Features flag is on"><input id="gputiming" type="checkbox" />Time GPU passes</label>
    <div id="timing"></div>
  </details>

  <footer><a href="validate.html" title="Run the test problems and compare them with exact solutions">Validation suite</a> &middot; <a href="https://github.com/guillochon/shaderous" target="_blank" rel="noopener" title="The code of this simulator">Source on GitHub</a> &middot; <a href="https://hub.valency.io/works/W_e4sja6m3" target="_blank" rel="noopener" title="The paper that describes this simulator">Paper on Valency</a><br>drag to orbit (pan in a slice), wheel or pinch to zoom, double-click to reset a slice, space to play or pause<br>with an editing tool picked, drag edits the gas and right-drag moves the view, ↑ ↓ move its ball in depth (Shift: 10&times;, Home: back); Esc puts the tool away; each tool's key (shown on hover) picks it up or puts it away<br><span title="Build of the page you are looking at">build ${BUILD}</span></footer>
`;

// On a phone the rail lies over the simulation, so it can be put away (remembered in this
// browser), leaving only the button that brings it back.
{
  const rail = $('rail');
  const toggle = $<HTMLButtonElement>('rail-toggle');
  const show = (open: boolean) => {
    rail.classList.toggle('closed', !open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.textContent = open ? 'Hide ▸' : '◂ Tools';
    toggle.title = open ? 'Hide the color bar and the editing tools' : 'Show the color bar and the editing tools';
  };
  let open = true;
  try {
    open = localStorage.getItem('shaderous.rail') !== 'closed';
  } catch {}
  show(open);
  toggle.addEventListener('click', () => {
    open = !open;
    show(open);
    // A tool left picked would edit the gas with nothing on screen to say so.
    if (!open && tool) {
      draft = null;
      tool = null;
      syncTools();
    }
    try {
      localStorage.setItem('shaderous.rail', open ? 'open' : 'closed');
    } catch {}
  });
}

// The color bar: the range at either end (editable), the opacity curve drawn on the bar, then
// its settings.
$('colorbar').innerHTML = `
  <div id="cb-unit" class="cb-unit" title="Unit of the displayed field"></div>
  <input id="hi" class="cb-range" type="text" inputmode="decimal" spellcheck="false" title="Top of the displayed range" />
  <canvas id="tf" title="Color bar. In the volume view the line is the opacity: drag its points sideways, double-click to add or remove one"></canvas>
  <input id="lo" class="cb-range" type="text" inputmode="decimal" spellcheck="false" title="Bottom of the displayed range" />
  <div class="cb-checks">
    <label class="check" title="The range follows the data"><input id="auto" type="checkbox" checked />Auto</label>
    <label class="check" title="Logarithmic color scale"><input id="log" type="checkbox" />Log</label>
    <label class="check" title="Outline the blocks of an adaptive mesh, in the slice and the volume view"><input id="blocks" type="checkbox" checked />Blocks</label>
    <label class="check" title="Contour lines in the slice and projection views, isosurfaces in the volume view, evenly spaced in log |value| across the displayed range (how many: View, Contours)"><input id="contours" type="checkbox" />Contours</label>
    <label class="check sub" id="fog-check" title="Volume view: draw the gas as a glowing fog. Off, the isosurfaces show alone"><input id="fog" type="checkbox" checked />Fog</label>
    <label class="check sub" id="exact-check" title="Volume view: isosurfaces of the field’s log interpolated between cell centers. Off, of the value smoothed by a cubic B-spline, which places a surface within the cells it crosses and takes away their terraces"><input id="exact" type="checkbox" />Exact</label>
    <label class="check" title="Volume view: outline the box"><input id="box" type="checkbox" checked />Box</label>
    <label class="check" id="flines-check" title="Magnetic field lines: tubes in the volume view, lines in the plane in the slice view and in 2D (how many: View, Field lines)"><input id="flines" type="checkbox" />Field lines</label>
  </div>
  <select id="cmap" title="Color map">${Object.keys(COLORMAPS).map((c) => opt(c, c)).join('')}</select>
  <label class="scrub" title="Opacity of the gas in the volume view"><input id="opacity" type="range" min="0" max="1" step="0.005" aria-label="Opacity" /><span>Opac<output id="opacity-out"></output></span></label>
`;

/** The user's own setups in the problem menu: the custom one the page holds, and the library's. */
function showOwnSetups() {
  const own = $('own-setups');
  const custom = customSetup();
  const names = listSetups();
  const key = `${customVersion()}|${names.join('|')}`;
  if (own.dataset.key !== key) {
    own.dataset.key = key;
    own.replaceChildren();
    const add = (value: string, label: string) => {
      const o = document.createElement('option');
      o.value = value;
      o.textContent = label; // a name from a file: text, never markup
      own.append(o);
    };
    if (custom) add(CUSTOM_ID, custom.problem.label);
    for (const n of names) add(`lib:${n}`, n);
  }
  own.hidden = !custom && names.length === 0;
}

/** Show a preset (or the custom setup) in the problem and variant menus. */
function showProblem(id: string) {
  showOwnSetups();
  const fam = familyFor(id);
  $<HTMLSelectElement>('problem').value = fam.id;
  const variant = $<HTMLSelectElement>('variant');
  if (variant.dataset.family !== fam.id) {
    variant.dataset.family = fam.id;
    variant.innerHTML = fam.variants.map((v) => opt(v.problem.id, v.label)).join('');
  }
  variant.value = id;
  variant.disabled = fam.variants.length < 2;
  // A setup that is not a preset has no variants to show.
  $('row-variant').hidden = id === CUSTOM_ID;
}
// The menu's own popup, with a search box over the list (the browser's menu cannot hold one).
installSetupSearch(
  $<HTMLSelectElement>('problem'),
  $<HTMLButtonElement>('problem-search'),
  () => {
    showOwnSetups();
    return [...$<HTMLSelectElement>('problem').querySelectorAll('optgroup')]
      .filter((g) => !g.hidden)
      .flatMap((g) => [...g.querySelectorAll('option')].map((o) => ({
        value: o.value,
        label: o.textContent ?? o.value,
        group: g.label,
        variants: o.value === CUSTOM_ID || o.value.startsWith('lib:') ? [] : familyById(o.value).variants.map((v) => ({ id: v.problem.id, label: v.label })),
      })));
  },
  (value, variant) => {
    const problem = $<HTMLSelectElement>('problem');
    problem.value = value;
    // A variant other than the first: shown in the menus, then loaded as its own menu would.
    if (variant && variant !== familyById(value).variants[0].problem.id) {
      showProblem(variant);
      $('sheet-title').textContent = problem.selectedOptions[0]?.textContent ?? 'Controls';
      $('variant').dispatchEvent(new Event('change'));
    } else problem.dispatchEvent(new Event('change'));
  },
);

// The page opens on the preset it is about to load, not on the first in the menu.
const firstLink = decodeLink(new URLSearchParams(location.search));
showProblem(firstLink.problem);
$('blurb').textContent = problemOf(firstLink.problem).blurb;

// Enter in a field to type in applies the value and lets go of the field: leaving it fires the
// change the browser has not fired yet, and the page's keys (space runs) work again.
document.addEventListener('keydown', (e) => {
  const el = e.target;
  if (e.key !== 'Enter' || e.isComposing || !(el instanceof HTMLInputElement)) return;
  if (['text', 'number', 'search', 'url', 'email', 'tel', 'password'].includes(el.type)) el.blur();
});

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
type ToolKind = Exclude<BrushOp['kind'], 'eos-out' | 'eos-in'> | 'source' | 'sink' | 'query' | 'profile';
/** The tools that only read the run: no brush, no settings of their own. */
const readsOnly = (t: ToolKind | null) => t === 'query' || t === 'profile';
/** Polytropic indices the star tool offers. */
const STAR_INDICES = [1, 1.5, 3];
const starId = (n: number) => `star-n-${String(n).replace('.', '_')}`;
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
  {
    kind: 'field', label: 'Field strength',
    title: 'Field strength: click to make the magnetic field 1.5\u00d7 stronger, or hold; Shift weakens it. The flux it gains or loses closes around the brush\u2019s edge, so the field stays divergence-free',
    icon: icon('<path d="M1.5 3c3.5 0 4 3.5 6.5 3.5S11 3 14.5 3M1.5 13c3.5 0 4-3.5 6.5-3.5s3 3.5 6.5 3.5" /><path d="M8 7v2.5" />'),
  },
  {
    kind: 'pull', label: 'Pull field',
    title: 'Pull field: drag to carry the magnetic field lines along with the pointer, most at the brush\u2019s center and less toward its edge. The gas stays where it is',
    icon: icon('<path d="M1.5 4h3.5c2 0 2 6 3 6s1-6 3-6h3.5M1.5 13h13" /><path d="M8 13.5v-3.5M6.3 11.7L8 10l1.7 1.7" />'),
  },
];
/** The tools that change the magnetic field, offered in runs that have one. */
const FIELD_TOOLS: ToolKind[] = ['field', 'pull'];
TOOLS.push({
  kind: 'source', label: 'Source',
  title: 'Source: click to place one that feeds gas in evenly, or press and drag to set it moving (the path it would take is drawn): it goes where gravity takes it, and its gas leaves with its velocity. Fixed, it stays put and a drag aims its gas (a longer drag is faster)',
  icon: icon('<circle cx="8" cy="8" r="2.3" fill="currentColor" stroke="none" /><path d="M8 1.5v2.3M8 12.2v2.3M1.5 8h2.3M12.2 8h2.3M3.4 3.4l1.6 1.6M11 11l1.6 1.6M12.6 3.4L11 5M5 11l-1.6 1.6" />'),
});
TOOLS.push({
  kind: 'sink', label: 'Sink',
  title: 'Sink: click to place a particle that takes gas away, or press and drag to set it moving (the path it would take in the gas as it is now is drawn). Without a mass it goes where gravity takes it and pulls on nothing; with one it is a point mass: it pulls on the gas and grows by what it takes in',
  icon: icon('<circle cx="8" cy="8" r="5.5" stroke-dasharray="2.2 2.2" /><path d="M2.5 2.5l3 3M5.5 3v2.5H3M13.5 13.5l-3-3M10.5 13v-2.5H13" />'),
});
TOOLS.push({
  kind: 'star', label: 'Star',
  title: 'Star: click to place a polytropic star, or press and drag to set it moving (a longer drag is faster; the path it would take in the gas as it is now is drawn). Size is its radius; it takes over the gas wherever it is denser, and is settled into balance on the grid as a preset’s star is. Switches self-gravity on',
  icon: icon('<path d="M8 2.1L9.6 6.4L14.2 6.6L10.6 9.4L11.8 13.9L8 11.3L4.2 13.9L5.4 9.4L1.8 6.6L6.4 6.4z" />'),
});
TOOLS.push({
  kind: 'query', label: 'Query',
  title: 'Query: hover over the gas to see the cell under the pointer (its state and, on an adaptive mesh, its block), or over a source, a sink or a point mass to see that too. Click to pin the readout: on a source or sink it follows that one; anywhere else it follows a massless tracer let go there. Dragging moves the view',
  icon: icon('<circle cx="6.8" cy="6.8" r="4.3" /><path d="M10 10l4.5 4.5" /><path d="M6.8 4.9v.01M6.8 6.6v2.6" />'),
});
TOOLS.push({
  kind: 'profile', label: 'Profile',
  title: 'Profile: press where a line starts and drag to where it ends (or click at each end) to plot a field along it. Shift keeps the line at 15° steps. In the volume view the line lies on the plane facing you through the point orbited; Alt puts an end on the gas seen under the pointer instead. Drag an end to move it; Esc lets a line being placed go',
  icon: icon('<path d="M1.5 14.5h13M1.5 14.5V1.5" /><path d="M3.5 11l3-4 2.5 2.5 4-6" />'),
});
/** Each tool's key on the keyboard: it picks the tool up, or puts it away, as a click does. */
const TOOL_KEYS: Record<ToolKind, string> = {
  mass: 'm', velocity: 'v', temperature: 't', obstacle: 'o', field: 'b', pull: 'g',
  source: 's', sink: 'k', star: 'r', query: 'q', profile: 'p',
};
/** A tool's description, with its key. */
const toolTitle = (t: (typeof TOOLS)[number]) => `${t.title} (${TOOL_KEYS[t.kind].toUpperCase()})`;
/**
 * The point masses the tools have placed in this run, in the order they were added: the last
 * of the run's masses (the simulation holds MAX_MASSES in all). A sink is one; a source that
 * moves rides one; a query pinned to a spot follows one (a massless tracer).
 */
type Placed = 'sink' | 'source' | 'query';
let placed: Placed[] = [];
/** The place among the run's masses of the i-th one the tools placed. */
const placedIndex = (i: number) => (sim?.massCount ?? 0) - placed.length + i;
/**
 * Take the placed masses for which `gone` holds out of the run, and with them the sources that
 * rode them; the sources riding the masses that stay follow those to their new places.
 */
function removePlaced(gone: (kind: Placed, index: number) => boolean) {
  if (!sim) return;
  const out = placed.map((k, i) => (gone(k, placedIndex(i)) ? placedIndex(i) : -1)).filter((i) => i >= 0);
  if (!out.length) return;
  placed = placed.filter((k, i) => !out.includes(placedIndex(i)));
  sim.removeMasses(out);
  logAction({ kind: 'remove', indices: out });
  sources = sources
    .filter((q) => q.carrier === undefined || !out.includes(q.carrier))
    .map((q) => (q.carrier === undefined ? q : { ...q, carrier: q.carrier - out.filter((i) => i < q.carrier!).length }));
  pushSources(sources);
  // A pinned query goes with its mass, or follows it to its new place.
  if (pin && 'mass' in pin) {
    const at = pin.mass;
    if (out.includes(at)) dropPin();
    else pin.mass = at - out.filter((i) => i < at).length;
  }
  lastDiag = undefined;
  liveMasses = null;
}
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
/**
 * `mass` and `soften` are the sink tool's: the mass of the sink, as a multiple of the gas in the
 * box, and the softening length of its gravity, as a share of the box.
 */
const toolParams: Record<ToolKind, { size: number; softness: number; amount: number; mass: number; soften: number }> = {
  mass: { size: 0.06, softness: 0.5, amount: 0.25, mass: 0, soften: 0.04 },
  velocity: { size: 0.08, softness: 0.5, amount: 0, mass: 0, soften: 0.04 },
  temperature: { size: 0.06, softness: 0.5, amount: 0, mass: 0, soften: 0.04 },
  obstacle: { size: 0, softness: 0, amount: 0, mass: 0, soften: 0.04 },
  field: { size: 0.08, softness: 0.5, amount: 0, mass: 0, soften: 0.04 },
  pull: { size: 0.1, softness: 0.7, amount: 0.5, mass: 0, soften: 0.04 },
  source: { size: 0.04, softness: 0.5, amount: 0.25, mass: 0, soften: 0.04 },
  sink: { size: 0.04, softness: 0.5, amount: 0.25, mass: 0, soften: 0.04 },
  star: { size: 0.1, softness: 0, amount: 0, mass: 0, soften: 0.04 },
  query: { size: 0, softness: 0, amount: 0, mass: 0, soften: 0.04 },
  profile: { size: 0, softness: 0, amount: 0, mass: 0, soften: 0.04 },
};
/** The sink tool, and the source tool, place ones that stay where they are put. */
let sinkFixed = false;
let sourceFixed = false;
/** The tool in hand places something that moves: a sink or a source that is not fixed. */
const toolMoves = () => (tool === 'sink' && !sinkFixed) || (tool === 'source' && !sourceFixed);
/** The sink tool places a point mass rather than a massless sink. */
const sinkHasMass = () => toolParams.sink.mass > 0;
let tool: ToolKind | null = null;
/**
 * Star tool: polytropic index, and log10 of the central density (1, as in the polytrope preset),
 * which the Mass slider holds: the mass it shows is that of a star of this density and the
 * tool's Size. Its magnetic field, as the polytrope preset's variants have it: none, a dipole
 * along z or a random (Kolmogorov) one, with log10 of the plasma beta at its center.
 */
let starN = 1.5;
let starLogRho = 0;
type StarField = 'none' | 'dipole' | 'random';
const STAR_FIELDS: { kind: StarField; label: string; title: string }[] = [
  { kind: 'none', label: 'B=0', title: 'No magnetic field' },
  { kind: 'dipole', label: 'Dip', title: 'A dipole field along z, confined to the star. The run becomes a magnetized one: it is loaded again from the state it is in, which Reset then returns to' },
  { kind: 'random', label: 'Rand', title: 'A tangled field with a Kolmogorov spectrum, confined to the star. The run becomes a magnetized one: it is loaded again from the state it is in, which Reset then returns to' },
];
let starField: StarField = 'none';
let starLogBeta = 3;
/** Mass of the star tool's star in code units, per unit central density. */
const starMassPerRho = () => polytropeMass(laneEmdenCached(starN), 1, toolParams.star.size * cfg.problem.box);
/**
 * Change the star tool's size or index with `change`, keeping the star's mass: its central
 * density follows (and may leave the Mass slider's range, which then sits at its end).
 */
function keepStarMass(change: () => void) {
  const mass = 10 ** starLogRho * starMassPerRho();
  change();
  starLogRho = Math.log10(mass / starMassPerRho());
}
/** The star tool's size or mass has been set by hand: until then they are the defaults below. */
let starSet = false;
/**
 * The star tool's defaults, for the run as it is: where the setup has physical units, a star of
 * one solar mass and one solar radius (no larger than the Size slider's 30% of the box);
 * otherwise a tenth of the box in radius, with unit central density.
 */
function starDefaults() {
  if (starSet || !cfg) return;
  const pu = problemUnits(cfg.problem, cfg.params);
  toolParams.star.size = pu ? Math.min(0.3, ASTRO.Rsun / (cfg.problem.box * unitOf(pu, 'length'))) : 0.1;
  starLogRho = pu ? Math.log10(ASTRO.Msun / unitOf(pu, 'mass') / starMassPerRho()) : 0;
}
/** Mass tool: new gas brings no momentum of its own. */
let massAtRest = true;
const slider = (id: string, label: string, title: string, min: number, max: number) =>
  `<label class="scrub" id="tp-${id}-row" title="${title}"><input id="tp-${id}" type="range" min="${min}" max="${max}" step="0.01" aria-label="${title}" /><span>${label}<output id="tp-${id}-out"></output></span></label>`;
$('tools').innerHTML =
  TOOLS.map((t) => `<button id="tool-${t.kind}" aria-pressed="false" aria-label="${t.label}" title="${toolTitle(t)}">${t.icon}<kbd class="hotkey" aria-hidden="true">${TOOL_KEYS[t.kind].toUpperCase()}</kbd></button>`).join('') +
  `<div id="tool-params" class="tool-params"><div class="tool-params-in">
    ${slider('size', 'Size', 'Brush radius, as a share of the box', 0.01, 0.3)}
    ${slider('softness', 'Soft', 'Softness of the edge: 0 is hard, 1 fades all the way from the center', 0, 1)}
    ${slider('amount', 'Amt', 'Gas added per click, as a share of the densest gas', 0.01, 1)}
    <label class="scrub" id="tp-rhoc-row" title="Mass of the star (in solar masses where the setup has physical units, with Size in solar radii). It stays as set when Size or n changes. The slider spans central densities from a thousandth to a thousand times the code unit at the present Size; a typed mass may lie outside it"><input id="tp-rhoc" type="range" min="-3" max="3" step="0.05" aria-label="Mass of the star" /><span>Mass<output id="tp-rhoc-out"></output></span></label>
    <div id="tool-index" title="Polytropic index n: p = K ρ^(1 + 1/n). 1.5 is a convective star, 3 a radiative one like the Sun">${STAR_INDICES.map((n) => `<button id="${starId(n)}" aria-pressed="false" title="n = ${n}">n=${n}</button>`).join('')}</div>
    <div id="tool-star-field">${STAR_FIELDS.map((f) => `<button id="star-b-${f.kind}" aria-pressed="false" title="${f.title}">${f.label}</button>`).join('')}</div>
    <label class="scrub" id="tp-beta-row" title="Plasma beta at the star's center: the gas pressure over the magnetic pressure B² / 2 there (a random field's on average). The larger, the weaker the field"><input id="tp-beta" type="range" min="1" max="7" step="0.05" aria-label="Plasma beta at the star's center" /><span>β<output id="tp-beta-out"></output></span></label>
    ${slider('mass', 'Mass', 'Mass of the sink, as a multiple of the gas in the box. At 0 it is a particle without mass: it takes gas away at the Flux rate and moves as gravity takes it, but pulls on nothing and never grows. With a mass it is a point mass with softened gravity: it pulls on the gas and grows by what it takes in', 0, 2)}
    ${slider('soften', 'Soft', 'Softening length of the point mass\u2019s gravity, as a share of the box: within it the pull falls away instead of growing without limit, so gas and other masses can pass close by. Size is the radius of its sink', 0.002, 0.3)}
    <button id="tool-fixed" aria-pressed="false" title="Fixed: it stays where it is placed, whatever pulls on it (and a drag aims a source\u2019s gas). Off: it moves as gravity takes it, and a drag sets it going">Fixed</button>
    <button id="tool-rest" aria-pressed="true" title="p = 0 for new mass: added gas arrives at rest and shares the momentum already there, slowing the flow. Off: it arrives with the local velocity">p = 0</button>
    <div id="tool-shapes">${SHAPES.map((q) => `<button id="shape-${q.mode}" aria-pressed="false" aria-label="${q.mode}" title="${q.title}">${q.icon}</button>`).join('')}</div>
    <button id="tool-erase" aria-pressed="false" title="Erase: the shape drawn turns obstacle back into gas, at rest">Erase</button>
    <button id="tool-clear" title="Remove every source and sink">Clear</button>
  </div></div>`;

function syncTools() {
  starDefaults();
  // A gas without an energy of its own (barotropic, liquid) has no temperature to change.
  const noTemp = !!cfg && !eosHasEnergy(cfg.eos);
  if (noTemp && tool === 'temperature') tool = null;
  // A star needs self-gravity (3D only), and an equation of state that leaves K free.
  const noStar = (cfg?.ndim ?? 3) < 3
    ? `A star needs self-gravity in 3D, which a ${cfg?.ndim === 1 ? '1D' : '2D'} run does not have`
    : noTemp ? 'This equation of state fixes the pressure of the gas, so a star’s cannot be chosen' : '';
  if (noStar && tool === 'star') tool = null;
  const noField = !cfg?.problem.mhd;
  if (noField && tool !== null && FIELD_TOOLS.includes(tool)) tool = null;
  // The MHD fluxes know no solid cell (mhd/mhd.ts mhdUnsupported).
  const noObstacle = cfg?.problem.mhd ? 'Obstacles are not available with a magnetic field yet' : '';
  if (noObstacle && tool === 'obstacle') tool = null;
  // A point mass in 1D would be a sheet or a line: not one the run's gravity knows.
  const noSink = cfg?.ndim === 1 ? 'Point masses are not available in 1D' : '';
  if (noSink && tool === 'sink') tool = null;
  // A spherical run's cells are not along straight lines of the box.
  const noProfile = cfg && spherical(cfg) && cfg.ndim > 1 ? 'Profiles are not available on a spherical grid yet' : '';
  if (noProfile && tool === 'profile') tool = null;
  for (const t of TOOLS) {
    const b = $<HTMLButtonElement>(`tool-${t.kind}`);
    b.hidden = (noField && FIELD_TOOLS.includes(t.kind)) || (t.kind === 'star' && !!noStar);
    b.setAttribute('aria-pressed', String(tool === t.kind));
    b.disabled = (t.kind === 'temperature' && noTemp) || (t.kind === 'star' && !!noStar) || (t.kind === 'obstacle' && !!noObstacle) || (t.kind === 'profile' && !!noProfile) || (t.kind === 'sink' && !!noSink);
    b.title = !b.disabled ? toolTitle(t) : t.kind === 'star' ? noStar : t.kind === 'obstacle' ? noObstacle : t.kind === 'profile' ? noProfile : t.kind === 'sink' ? noSink : 'This equation of state ties the temperature to the density';
  }
  canvas.classList.toggle('tool', tool !== null);
  // The tool's settings slide out under its own button, pushing the buttons below it down.
  const panel = $('tool-params');
  const open = tool !== null && !readsOnly(tool);
  if (open) {
    const b = $(`tool-${tool}`);
    if (b.nextElementSibling !== panel) {
      // From another tool's: shut, move under this one, and open from there. If it was open,
      // a copy of it stays behind and slides shut at the same time, then goes.
      if (panel.classList.contains('open')) {
        const ghost = panel.cloneNode(true) as HTMLElement;
        ghost.removeAttribute('id');
        for (const el of ghost.querySelectorAll('[id]')) el.removeAttribute('id');
        ghost.classList.add('ghost');
        ghost.inert = true;
        panel.before(ghost);
        void ghost.offsetHeight;
        ghost.classList.remove('open');
        setTimeout(() => ghost.remove(), 300);
      }
      panel.classList.remove('open');
      b.after(panel);
      void panel.offsetHeight; // lay out the shut panel first, so the opening animates
    }
  }
  panel.classList.toggle('open', open);
  if (tool !== 'query') hideCellPop();
  if (tool === null || readsOnly(tool)) {
    $('brush').style.display = 'none';
    return;
  }
  const q = toolParams[tool];
  $('tp-amount-row').hidden = tool !== 'mass' && tool !== 'source' && tool !== 'sink' && tool !== 'pull';
  $('tp-mass-row').hidden = tool !== 'sink';
  // Only a point mass has gravity to soften.
  $('tp-soften-row').hidden = !(tool === 'sink' && sinkHasMass());
  // A sink has no soft edge to set: it always fades from the center. A star's edge is its
  // own profile.
  $('tp-softness-row').hidden = tool === 'sink' || tool === 'star';
  $('tp-rhoc-row').hidden = $('tool-index').hidden = $('tool-star-field').hidden = tool !== 'star';
  $('tp-beta-row').hidden = tool !== 'star' || starField === 'none';
  for (const n of STAR_INDICES) $(starId(n)).setAttribute('aria-pressed', String(starN === n));
  for (const f of STAR_FIELDS) $(`star-b-${f.kind}`).setAttribute('aria-pressed', String(starField === f.kind));
  const rc = $<HTMLInputElement>('tp-rhoc');
  rc.value = String(starLogRho);
  rc.style.setProperty('--p', `${Math.min(100, Math.max(0, (100 * (starLogRho + 3)) / 6))}%`);
  const be = $<HTMLInputElement>('tp-beta');
  be.value = String(starLogBeta);
  be.style.setProperty('--p', `${(100 * (starLogBeta - 1)) / 6}%`);
  // The same slider is the amount of a dab of mass and the flux of a source or sink.
  const flux = tool === 'source' || tool === 'sink';
  $('tp-amount-row').title = tool === 'source'
    ? 'Flux: how fast the source feeds gas in. At 100% it adds the density of the densest gas in the time the fastest signal takes to cross it'
    : tool === 'sink'
      ? 'Flux: how fast the sink takes gas away. Without a mass, 100% empties its center in the time the fastest signal takes to cross it; with one, 25% is the orbital rate at its edge. At the far left it is exactly 0: a particle that takes no gas, and only moves'
      : tool === 'pull'
        ? 'Grip: how much of the drag the field at the brush\u2019s center moves by. The rest slips through the brush, so the field it sweeps up ahead of it can at most grow by 1 / (1 \u2212 grip): twice as strong at 50%, ten times at 90%'
        : 'Gas added per click, as a share of the densest gas';
  $('tp-amount-row').querySelector('span')!.firstChild!.textContent = flux ? 'Flux' : tool === 'pull' ? 'Grip' : 'Amt';
  // For a source or a sink the softness is how its rate fades toward its edge, and is called
  // that: "Soft" is the softening of a point mass's gravity, a slider of the sink tool's own.
  $('tp-softness-row').querySelector('span')!.firstChild!.textContent = flux ? 'Edge' : 'Soft';
  $('tp-softness-row').title = flux
    ? 'Edge: how the rate falls off toward the edge. 0 is the same all the way out, 1 fades all the way from the center'
    : 'Softness of the edge: 0 is hard, 1 fades all the way from the center';
  const mine = tool === 'sink' ? sources.filter((q) => q.decay > 0).length + placed.filter((k) => k === 'sink').length : sources.filter((q) => q.decay === 0).length;
  $('tool-clear').hidden = !flux;
  $('tool-clear').title = tool === 'sink' ? 'Remove every sink placed with this tool, with or without a mass' : 'Remove every source';
  $('tool-clear').textContent = mine ? `Clear ${mine}` : 'Clear';
  $<HTMLButtonElement>('tool-clear').disabled = mine === 0;
  $('tool-fixed').hidden = tool !== 'sink' && tool !== 'source';
  $('tool-fixed').setAttribute('aria-pressed', String(tool === 'source' ? sourceFixed : sinkFixed));
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
  const mr = massRange();
  if (mr && tool === 'sink' && q.mass > 0) q.mass = Math.min(mr.hi, Math.max(mr.lo, q.mass));
  for (const k of ['size', 'softness', 'amount', 'mass', 'soften'] as const) {
    const el = $<HTMLInputElement>(`tp-${k}`);
    if (k === 'soften') el.step = '0.002';
    const log = k === 'mass' && mr;
    if (k === 'mass') [el.min, el.max, el.step] = log ? ['0', '1', '0.001'] : ['0', '2', '0.01'];
    // A sink's flux goes all the way down to none: a point mass that only orbits.
    if (k === 'amount') [el.min, el.max] = [tool === 'sink' ? '0' : '0.01', tool === 'pull' ? '0.9' : '1'];
    const pos = log ? massPos(q.mass, mr) : q[k];
    el.value = String(pos);
    el.style.setProperty('--p', `${(100 * (pos - +el.min)) / (+el.max - +el.min)}%`);
  }
  showToolValues();
}

/**
 * With point masses in the run, the sink tool's Mass slider is logarithmic: from 1e-3 of the
 * lightest to 1e3 times the heaviest, with exactly 0 (a sink without a mass) at its left end.
 * Without any it is linear, from 0 to twice the gas in the box. Both in multiples of that gas,
 * as the tool's setting is. The last range read is kept while a new readout is awaited (just
 * after a mass is placed there is none).
 */
let lastMassRange: { lo: number; hi: number } | null = null;
function massRange(): { lo: number; hi: number } | null {
  const d = lastDiag;
  if (!d) return lastMassRange;
  const gas = d.mass > 0 ? d.mass : cfg.problem.box ** cfg.ndim;
  // (A massless particle's token G M is not a mass to measure against.)
  const ms = cfg.G > 0 ? d.masses.filter((m) => !m.massless).map((m) => m.gm / cfg.G / gas).filter((m) => m > 0) : [];
  lastMassRange = ms.length ? { lo: 1e-3 * Math.min(...ms), hi: 1e3 * Math.max(...ms) } : null;
  return lastMassRange;
}
/** Position (0 to 1) of a mass on the logarithmic slider, and the mass at a position. */
const massPos = (x: number, r: { lo: number; hi: number }) =>
  x <= 0 ? 0 : Math.min(1, Math.max(0.001, Math.log(x / r.lo) / Math.log(r.hi / r.lo)));
const massAt = (p: number, r: { lo: number; hi: number }) => (p <= 0 ? 0 : r.lo * (r.hi / r.lo) ** p);

type ToolSlider = 'size' | 'softness' | 'amount' | 'mass' | 'soften' | 'rhoc' | 'beta';
const TOOL_SLIDERS: ToolSlider[] = ['size', 'softness', 'amount', 'mass', 'soften', 'rhoc', 'beta'];
/** The star tool's sliders that hold the log10 of what they set. */
const logSlider = (k: ToolSlider): k is 'rhoc' | 'beta' => k === 'rhoc' || k === 'beta';

/** A tool slider's setting: what the slider holds (the log of the density, or of beta, for a star's). */
const toolSetting = (k: ToolSlider): number => (k === 'rhoc' ? starLogRho : k === 'beta' ? starLogBeta : toolParams[tool!][k]);

/**
 * How a tool slider's value is shown: the number is the setting (10 to its power, for a star's
 * density and beta) times `factor`, followed by `unit`. A star's size and mass are shown in
 * solar radii and masses in physical units, as the polytrope preset's are (in code units: its
 * radius as a share of the box, and its mass). In physical units that is what the tool sets:
 * the brush radius, the gas a dab adds, a source's or sink's flux and a point mass's mass, at
 * the scales the tools measure against now (so it follows the run). In code units, or for a
 * scale-free problem, it is the share the slider sets: size and amount in percent, mass as a
 * multiple of the gas in the box.
 */
function toolReadout(k: ToolSlider): { factor: number; unit: string } {
  const q = toolParams[tool!];
  const u = units();
  const shown = u.physical && unitSys !== 'code';
  const box = cfg.problem.box;
  const { rhoMax, vRef } = toolRefs();
  const phys = (factor: number, m: Measure) => ({ factor: factor * u.scale(m), unit: ` ${u.label(m)}` });
  // The star tool speaks of its star as the polytrope preset does: in solar radii and masses.
  const pu = shown && tool === 'star' ? problemUnits(cfg.problem, cfg.params) : undefined;
  switch (k) {
    case 'size':
      if (pu) return { factor: (box * unitOf(pu, 'length')) / ASTRO.Rsun, unit: ' R☉' };
      return shown ? phys(box, 'length') : { factor: 100, unit: '%' };
    case 'soften': return shown ? phys(box, 'length') : { factor: 100, unit: '%' };
    case 'beta': return { factor: 1, unit: '' };
    case 'softness': return { factor: 1, unit: '' };
    case 'mass': return shown ? phys(lastDiag?.mass ?? box ** cfg.ndim, 'mass') : { factor: 1, unit: '\u00d7' };
    case 'rhoc': return pu ? { factor: (starMassPerRho() * unitOf(pu, 'mass')) / ASTRO.Msun, unit: ' M☉' } : { factor: starMassPerRho(), unit: '' };
    case 'amount': {
      // A point mass's flux is a share of the orbital rate at its edge, in any units.
      if (!shown || tool === 'pull' || (tool === 'sink' && sinkHasMass())) return { factor: 100, unit: '%' };
      // As the source code below measures it: against the densest gas and the time a signal
      // takes to cross the source.
      const perAmount = (vRef || 1) / (q.size * box);
      return tool === 'source' ? phys(perAmount * rhoMax, [1, -3, -1]) : tool === 'sink' ? phys(perAmount, 'rate') : phys(rhoMax, 'density');
    }
  }
}

/** The number a tool slider shows (without its unit). */
const toolShown = (k: ToolSlider) => (logSlider(k) ? 10 ** toolSetting(k) : toolSetting(k)) * toolReadout(k).factor;

function showToolValues() {
  if (!cfg || tool === null || readsOnly(tool)) return;
  for (const k of TOOL_SLIDERS) {
    const { unit } = toolReadout(k);
    const v = toolShown(k);
    $(`tp-${k}-out`).textContent = unit === '%' ? `${Math.round(v)}%` : k === 'softness' ? v.toFixed(2) : `${fmt3(v)}${unit}`;
  }
}

/**
 * A click on a tool slider's number turns it into a field to type the value in, in the units
 * shown. Enter or leaving the field applies it (kept within the slider's range); Escape does not.
 */
function editToolValue(k: ToolSlider) {
  if (tool === null || readsOnly(tool)) return;
  const row = $(`tp-${k}-row`);
  const range = $<HTMLInputElement>(`tp-${k}`);
  const edit = document.createElement('input');
  edit.type = 'text';
  edit.className = 'scrub-edit';
  const v0 = toolShown(k);
  // Very large or small numbers in exponent form ("1.481e16"), others as they are.
  edit.value = v0 !== 0 && (Math.abs(v0) >= 1e5 || Math.abs(v0) < 1e-3) ? (+v0.toPrecision(4)).toExponential().replace('e+', 'e') : String(+v0.toPrecision(4));
  edit.setAttribute('aria-label', `${row.title} (${toolReadout(k).unit.trim() || 'value'})`);
  row.classList.add('editing');
  row.append(edit);
  edit.focus();
  edit.select();
  let done = false;
  const finish = (apply: boolean) => {
    if (done) return;
    done = true;
    const v = parseFloat(edit.value);
    edit.remove();
    row.classList.remove('editing');
    if (!apply || !Number.isFinite(v) || tool === null || readsOnly(tool)) return;
    const x = v / toolReadout(k).factor;
    if (logSlider(k) && !(x > 0)) return;
    const mr = k === 'mass' ? massRange() : null;
    const setting = mr
      ? (x <= 0 ? 0 : Math.min(mr.hi, Math.max(mr.lo, x)))
      : Math.min(+range.max, Math.max(+range.min, logSlider(k) ? Math.log10(x) : x));
    // A star's mass is whatever is typed, within the slider's range or not.
    if (tool === 'star' && (k === 'rhoc' || k === 'size')) starSet = true;
    if (k === 'rhoc') starLogRho = Math.log10(x);
    else if (k === 'beta') starLogBeta = setting;
    else if (tool === 'star' && k === 'size') keepStarMass(() => (toolParams.star.size = Math.min(0.5, x)));
    else toolParams[tool][k] = setting;
    syncTools();
  };
  edit.addEventListener('keydown', (e) => {
    e.stopPropagation(); // the page's shortcuts (space runs) stay out of the field
    if (e.key === 'Enter') finish(true);
    else if (e.key === 'Escape') finish(false);
  });
  edit.addEventListener('blur', () => finish(true));
}
for (const k of TOOL_SLIDERS) {
  // The number lies over the slider: pressed and dragged it moves the slider, as the bar beside
  // it does; only a click that does not move starts typing a value.
  const out = $(`tp-${k}-out`);
  const range = $<HTMLInputElement>(`tp-${k}`);
  out.title = 'Drag to slide; click to type a value';
  let press: { x: number; dragging: boolean } | null = null;
  const slideTo = (x: number) => {
    const r = range.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (x - r.left) / r.width));
    range.value = String(+range.min + f * (+range.max - +range.min));
    range.dispatchEvent(new Event('input'));
  };
  out.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    e.preventDefault(); // the slider does not jump to where the number was pressed
    e.stopPropagation();
    out.setPointerCapture(e.pointerId);
    press = { x: e.clientX, dragging: false };
  });
  out.addEventListener('pointermove', (e) => {
    if (!press) return;
    if (!press.dragging && Math.abs(e.clientX - press.x) < 4) return;
    press.dragging = true;
    slideTo(e.clientX);
  });
  out.addEventListener('pointerup', (e) => {
    const p = press;
    press = null;
    if (out.hasPointerCapture(e.pointerId)) out.releasePointerCapture(e.pointerId);
    if (p && !p.dragging) editToolValue(k);
  });
  out.addEventListener('pointercancel', () => (press = null));
  // The press was handled above: the click that follows is not one on the slider the label holds.
  out.addEventListener('click', (e) => e.preventDefault());
}
for (const t of TOOLS) {
  $(`tool-${t.kind}`).addEventListener('click', () => {
    tool = tool === t.kind ? null : t.kind;
    draft = null;
    syncTools();
  });
}
// The tools' keys. Not with Ctrl, Cmd or Alt (the browser's shortcuts), nor Shift (held while
// drawing), nor where a key means something already: a field, a menu, the setup dialog.
window.addEventListener('keydown', (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || e.isComposing) return;
  const t = e.target as HTMLElement | null;
  if (t && (t.closest('input, select, textarea, [contenteditable], dialog') || t.isContentEditable)) return;
  if (document.querySelector('dialog[open]')) return;
  const hit = TOOLS.find((x) => TOOL_KEYS[x.kind] === e.key.toLowerCase());
  if (!hit) return;
  const b = $<HTMLButtonElement>(`tool-${hit.kind}`);
  if (b.hidden || b.disabled) return;
  e.preventDefault();
  // On a phone the rail may be put away: brought back, so the tool picked is on screen.
  if (tool !== hit.kind && $('rail').classList.contains('closed')) $('rail-toggle').click();
  b.click();
});
for (const sh of SHAPES) {
  $(`shape-${sh.mode}`).addEventListener('click', () => {
    shapeMode = sh.mode;
    draft = null;
    syncTools();
  });
}
/** Each tool clears what it placed: its sources or sinks, and the masses they are or ride. */
function clearPlaced(sinks: boolean) {
  removePlaced((kind) => kind === (sinks ? 'sink' : 'source'));
  const keep = sources.filter((q) => (q.decay > 0) !== sinks);
  if (keep.length !== sources.length) {
    sources = keep;
    pushSources(sources);
  }
  syncTools();
}
$('tool-clear').addEventListener('click', () => clearPlaced(tool === 'sink'));
$('tool-erase').addEventListener('click', () => {
  obstacleErase = !obstacleErase;
  syncTools();
});
for (const k of ['size', 'softness', 'amount', 'mass', 'soften'] as const) {
  $(`tp-${k}`).addEventListener('input', () => {
    if (!tool) return;
    const v = +$<HTMLInputElement>(`tp-${k}`).value;
    const mr = k === 'mass' ? massRange() : null;
    // A star keeps its mass as its size changes: the two are set apart.
    if (tool === 'star' && k === 'size') {
      starSet = true;
      keepStarMass(() => (toolParams.star.size = v));
    } else toolParams[tool][k] = mr ? massAt(v, mr) : v;
    syncTools();
  });
}
$('tp-rhoc').addEventListener('input', () => {
  starSet = true;
  starLogRho = +$<HTMLInputElement>('tp-rhoc').value;
  syncTools();
});
$('tp-beta').addEventListener('input', () => {
  starLogBeta = +$<HTMLInputElement>('tp-beta').value;
  syncTools();
});
for (const f of STAR_FIELDS) {
  $(`star-b-${f.kind}`).addEventListener('click', () => {
    starField = f.kind;
    syncTools();
  });
}
for (const n of STAR_INDICES) {
  $(starId(n)).addEventListener('click', () => {
    keepStarMass(() => (starN = n));
    syncTools();
  });
}
$('tool-fixed').addEventListener('click', () => {
  if (tool === 'source') sourceFixed = !sourceFixed;
  else sinkFixed = !sinkFixed;
  syncTools();
});
$('tool-rest').addEventListener('click', () => {
  massAtRest = !massAtRest;
  syncTools();
});
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || (!tool && !pin && !profiles.length)) return;
  // Esc first drops a shape or a profile line that is being drawn, then a pinned readout, then
  // puts the tool away, then closes the profile (every line).
  if (draft) draft = null;
  else if (profileDraft) profileDraft = null;
  else if (pin) unpin();
  else if (tool) tool = null;
  else closeProfile();
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
/** System the readouts are shown in, for a problem with physical units. */
let unitSys: DisplaySystem = (() => {
  try {
    const v = localStorage.getItem('shaderous.units');
    return v === 'si' || v === 'code' ? v : 'cgs';
  } catch {
    return 'cgs';
  }
})();
/** How the current run's values are shown. */
// (Comoving coordinates: physical values are proper ones at the scale factor last read.)
const units = (): Readout =>
  readout(problemUnits(cfg.problem, cfg.params), cfg.problem.composition, unitSys, cfg.problem.cosmology ? (lastDiag?.scale ?? cfg.problem.cosmology.aStart) : undefined);
/** The name of the field on view. */
const viewFieldName = () =>
  view.field === MASS_FRACTION_FIELD ? `X(${speciesLabel(cfg.problem.species![view.species])})` : projecting() && view.field === 0 ? 'Column density' : FIELDS[view.field];
/** The projection view (a 2D run is always shown as its plane). */
const projecting = () => view.mode === 'projection' && cfg?.ndim === 3;
/** What the field on view measures: density projected is a column density, mass per area. */
const viewMeasure = (): Measure => (projecting() && view.field === 0 ? [1, -2, 0] : fieldMeasure(view.field, cfg.eosParams.gamma));

/**
 * Volume view: the least density drawn, from the slider's place between the smallest and largest
 * density in the box (log scale). 0 when the slider is at the bottom: all the gas is drawn, even
 * as the smallest density changes.
 */
function densityCut(): number {
  const r = lastDiag?.ranges[0];
  if (!(view.minRho > 0) || !r || !(r[1] > 0)) return 0;
  const lo = Math.log10(Math.max(r[0], r[1] * 1e-30));
  const hi = Math.log10(r[1]);
  return 10 ** (lo + view.minRho * (hi - lo));
}

/** The label of the slider: the slice position, or in the volume view the least density drawn. */
function showDensityCut() {
  if (!cfg) return;
  const label = $('row-slicepos').querySelector('span')!;
  let text = 'Slice position';
  if (cfg.ndim === 3 && view.mode === 'volume') {
    const r = lastDiag?.ranges[0];
    const rho = view.minRho > 0 ? densityCut() : r?.[0];
    const u = units();
    text = rho === undefined ? 'Minimum density' : `Minimum density: ${num(rho * u.scale('density'))} ${u.label('density')}`.trimEnd();
  }
  if (label.textContent !== text) label.textContent = text;
}

/** The ends of the color bar, and its unit, in the displayed units. */
function showRange() {
  const u = units();
  const m = viewMeasure();
  const k = u.scale(m);
  const a = document.activeElement;
  if (a !== $('lo')) setValue($<HTMLInputElement>('lo'), num(view.lo * k));
  if (a !== $('hi')) setValue($<HTMLInputElement>('hi'), num(view.hi * k));
  $('cb-unit').textContent = u.label(m);
}
let running = true;
/**
 * `?headless=1`: the page is driven by a script through window.shaderous (ui/api.ts, see
 * docs/api.md). Nothing plays by itself, and the frame loop draws nothing: the view is drawn
 * when a screenshot asks for it, and the GPU is left to the steps the script asks for.
 */
const headless = new URLSearchParams(location.search).get('headless') === '1';
/** The script is stepping the run: the frame loop takes no steps of its own meanwhile. */
let driving = false;
/** A movie is recording at this size (ui/api.ts record()): the frame loop draws nothing meanwhile. */
let recordingSize: [number, number] | null = null;
/** The recording's progress, for the Record button. */
let recordProgress: RecordingStatus | null = null;
/** Whether a run starts playing by itself when it loads or replays (remembered in this browser). */
let autoplay = true;
try {
  autoplay = localStorage.getItem('shaderous.autoplay') !== 'off';
} catch {}
/**
 * Whether a change that needs a new simulation loads it at once (remembered in this browser).
 * Off, the change waits for Run (see `pending`). A script's changes always load.
 */
let autoload = true;
try {
  autoload = headless || localStorage.getItem('shaderous.autoload') !== 'off';
} catch {}
/**
 * With Autoload off: what waits for Run. `pick`: the problem and grid chosen in the menus, to be
 * built; without one, the present run starts again from its initial conditions (a preset
 * parameter changed). `keepPhysics`: as build() takes it.
 */
let pending: {
  pick?: { id: string; n: number; ndim: 1 | 2 | 3; levels: number; params?: Record<string, number>; settings?: { setting: Setting; value: unknown }[] };
  keepPhysics: boolean;
} | null = null;
/** The preset that waits for Run, if it is another than the run's: its own controls are the ones shown. */
const waitingProblem = (): Problem | null => (pending?.pick && cfg && pending.pick.id !== cfg.problem.id ? problemOf(pending.pick.id) : null);
/** That preset's parameters as they will be loaded: its defaults, with what was changed while it waits. */
const waitingParams = (p: Problem): Record<string, number> => (pending!.pick!.params ??= Object.fromEntries((p.params ?? []).map((q) => [q.id, q.value])));
let stepOnce = false;
let building = false;
let finished = false;
let lastDiag: Diagnostics | undefined;
/**
 * Checkpoints: off until the user asks. While on, cfg.tStop holds the clock at the next one
 * until it is written (see armCheckpoints and the diagnostics readout in tick).
 */
const ckpt = {
  on: false,
  every: 1,
  sink: null as CheckpointSink | null,
  busy: false,
  written: 0,
  error: '',
};

/** Where the query tool was pressed, until it is let go (a click pins the readout). */
let queryPress: { x: number; y: number } | null = null;

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
  /** pull: where the field under the pointer has been carried to so far (fractions of the box) */
  pulled?: Vec3;
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

// --- session log (ui/session.ts) ---------------------------------------------------------------
// What the tools and the settings do to the run, step by step: the API's actions() and replay(),
// and the Save log button.
const session = new SessionRecorder();
/** The settings as a link without the view, as last logged (noteSettings()). */
let sessionSettings = '';
/** A replay is under way (ui/api.ts): the mesh re-meshes where the log says, not on its own checks. */
let replaying = false;
const settingsQuery = () =>
  encodeLink({ cfg, view: viewDefaults(cfg.problem, cfg.ndim), auto: true, colormap: 'inferno', opacity: defaultOpacity(), setup: customLink?.text || undefined }).toString();

/** The steps taken and the time, as diagnostics() has them, once the work submitted so far is done. */
function stampNow(s: Simulation): Promise<{ steps: number; t: number }> {
  const d = gpu.device;
  const buf = d.createBuffer({ label: 'session stamp', size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(s.clockBuffer, 0, buf, 0, 16);
  d.queue.submit([enc.finish()]);
  return buf.mapAsync(GPUMapMode.READ).then(() => {
    const [dt, t, , banked] = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    buf.destroy();
    return { steps: banked + (dt > 0 ? 1 : 0), t: t + dt };
  });
}

/** Log an action applied to the run now (before any step submitted after this call). */
function logAction(action: Action) {
  if (!sim || !session.active) return;
  session.add(action, stampNow(sim));
}

/** A run starts (a load, a reset): its log starts over. */
function beginSession(replayable: boolean, why?: string) {
  if (!sim || !cfg) return session.stop();
  session.start(linkQuery(), BUILD, replayable, why);
  sessionSettings = settingsQuery();
  watchRegrids(sim);
}

/** Log the re-meshes the run's own refinement checks decide (in a replay, it makes none), and changes of the clock's stop. */
function watchRegrids(s: Simulation) {
  s.onRegrid = (steps) => {
    if (s === sim) logAction({ kind: 'regrid', steps });
  };
  if (s.autoRegrid !== undefined) s.autoRegrid = !replaying;
  // Pauses of per-block timesteps likewise: decided live, taken from the log in a replay.
  s.onBins = (pause) => {
    if (s === sim) logAction({ kind: 'bins', pause });
  };
  if (s.autoBins !== undefined) s.autoBins = !replaying;
  sessionClock = clockOf();
  s.onParams = () => {
    if (s !== sim || !session.active) return;
    const now = clockOf();
    if (now.stop === sessionClock.stop && now.frame === sessionClock.frame) return;
    sessionClock = now;
    logAction({ kind: 'clock', ...now });
  };
}

/** The clock's stop as the uniform block has it (sim.ts packParams), and its frame time. */
function clockOf(): { stop: number | null; frame: number | null } {
  const stop = Math.min(Number.isFinite(cfg.tEnd) ? cfg.tEnd : Infinity, cfg.tStop ?? Infinity);
  return { stop: Number.isFinite(stop) ? stop : null, frame: cfg.tFrame ?? null };
}
/** The clock's stop as last logged. */
let sessionClock: { stop: number | null; frame: number | null } = { stop: null, frame: null };

/** Log the settings if they changed since last logged. */
function noteSettings() {
  if (!sim || !cfg || !session.active) return;
  const q = settingsQuery();
  if (q === sessionSettings) return;
  const changes = queryDiff(sessionSettings, q);
  sessionSettings = q;
  logAction({ kind: 'set', changes, spec: runningSpec(cfg) });
}

/** The sources and sinks as set now, to the run and to the log. */
function pushSources(list: SourceSpec[]) {
  if (!sim) return;
  sim.setSources(list);
  logAction({ kind: 'sources', list });
}

/** Add a point mass a tool placed, to the run and to the log; false if the run holds no more. */
function addPlaced(s: Simulation, spec: PointMass, role: MassRole): boolean {
  const ok = s.addMass(spec);
  if (ok) {
    placed.push(role);
    if (s === sim) logAction({ kind: 'mass', spec, role });
  }
  return ok;
}

/** Apply a brush now (not at the next frame), and log it. */
function brushNow(op: BrushOp) {
  if (!sim) return;
  const enc = gpu.device.createCommandEncoder();
  // A re-mesh that waits goes first, and logs itself first (see the frame loop).
  sim.encodeBrush(enc, op);
  logAction({ kind: 'brush', op });
  gpu.device.queue.submit([enc.finish()]);
}

/** Self-gravity hears about the gas a tool added or removed; the log too. */
async function tellMass(target: Simulation) {
  const d = await target.diagnostics();
  if (target !== sim) return;
  target.massChanged(d.mass);
  logAction({ kind: 'massChanged', mass: d.mass });
}

const editor = new OpacityEditor($<HTMLCanvasElement>('tf'), pushTransfer, true);

function pushTransfer() {
  renderer?.setTransfer(bakeTransfer(editor.colormap, editor.points));
}

/**
 * The opacity sliders are logarithmic: a slider's position, 0 to 1, runs from the least value
 * to the greatest in equal ratios, so that the faint end has as much of the slider as the
 * opaque end. The gas's opacity runs from 1 to 300, an isosurface's from 0.01 to 1.
 */
const OPACITY_RANGE = { lo: 1, hi: 300 };
const SURFACE_RANGE = { lo: 0.01, hi: 1 };
const logPos = (v: number, r: { lo: number; hi: number }) => Math.min(1, Math.max(0, Math.log(v / r.lo) / Math.log(r.hi / r.lo)));
const logAt = (p: number, r: { lo: number; hi: number }) => +(r.lo * (r.hi / r.lo) ** p).toPrecision(3);

/** The opacity control: its fill and its value. */
function syncOpacity() {
  const el = $<HTMLInputElement>('opacity');
  el.style.setProperty('--p', `${100 * +el.value}%`);
  $('opacity-out').textContent = String(+view.opacity.toPrecision(3));
}

const num = (v: number) => (v === 0 ? '0' : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? v.toPrecision(5) : v.toExponential(2));

/** A screenshot is being taken or saved. */
let shooting = false;

/** The Record button: Record, or Stop while a movie records. */
function syncRecordButton() {
  const b = $('record');
  // The movie so far is on the status line.
  const text = recordingSize ? 'Stop' : 'Record';
  if (b.textContent !== text) b.textContent = text;
  b.classList.toggle('recording', !!recordingSize);
  $<HTMLButtonElement>('shot').disabled = !!recordingSize || shooting;
}


/** The Screenshot button: the view as a PNG, a plane at the grid's own resolution (api.shoot()). */
async function takeScreenshot() {
  if (recordingSize || shooting || !sim) return;
  const flat = flatView();
  const region = $<HTMLSelectElement>('shot-region').value === 'view' ? 'view' : 'box';
  const overlays = { colorbar: $<HTMLInputElement>('rec-colorbar').checked, triad: $<HTMLInputElement>('rec-triad').checked, status: $<HTMLInputElement>('rec-status').checked };
  const wasRunning = running;
  shooting = true;
  syncRecordButton();
  // The picture is of one state: the run waits for it.
  if (running) {
    running = false;
    syncControls();
  }
  const b = $('shot');
  b.textContent = 'Saving…';
  try {
    const shot = await api.shoot(flat ? { native: region, overlays } : { overlays });
    const blob = await new Promise<Blob | null>((res) => shot.canvas.toBlob(res, 'image/png'));
    if (!blob) throw new Error(`the browser could not encode a PNG of ${shot.width} × ${shot.height} pixels`);
    downloadBlob(`shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}-t${+shot.t.toPrecision(4)}.png`, blob);
    const f = shot.native;
    const size = `${shot.width} × ${shot.height} px, ${blob.size >= 1e6 ? `${(blob.size / 1e6).toFixed(1)} MB` : `${Math.ceil(blob.size / 1e3)} kB`}`;
    if (!f) notice(`saved a screenshot: ${size}`);
    else {
      const capped = f.side < f.n ? `; the map is capped at ${f.side}: ${+(f.n / f.side).toPrecision(3)} cells per pixel` : '';
      notice(`saved ${f.width} × ${f.height} ${f.side < f.n ? 'pixels' : 'cells'} (${size})${capped}`);
    }
  } catch (e) {
    notice(`Screenshot: ${e instanceof Error ? e.message : String(e)}${flat && region === 'box' ? '; As the view takes less' : ''}`);
  } finally {
    shooting = false;
    b.textContent = 'Screenshot';
    syncRecordButton();
    if (wasRunning && !finished && !recordingSize) {
      running = true;
      syncControls();
    }
  }
}

/** The movie settings, and the Record button that starts and stops a recording with them. */
function bindRecord() {
  // Only where the browser can encode video (WebCodecs).
  const can = typeof VideoEncoder !== 'undefined';
  $('record').hidden = !can;
  // Without video encoding the section keeps only the screenshot and what it shows.
  for (const el of $('movie').querySelectorAll<HTMLElement>('.g3')) el.hidden = !can;
  $('shot').addEventListener('click', () => void takeScreenshot());
  const cadence = $<HTMLSelectElement>('rec-cadence');
  const showCadence = () => {
    $('row-rec-every').hidden = cadence.value !== 'steps';
    $('row-rec-dt').hidden = cadence.value !== 'time';
    $('sum-movie').textContent = cadence.value === 'steps' ? (+$<HTMLInputElement>('rec-every').value > 1 ? `every ${$<HTMLInputElement>('rec-every').value} steps` : 'every step') : `every Δt = ${$<HTMLInputElement>('rec-dt').value}`;
  };
  for (const id of ['rec-cadence', 'rec-every', 'rec-dt']) $(id).addEventListener('input', showCadence);
  cadence.addEventListener('change', showCadence);
  showCadence();
  $('record').addEventListener('click', () => {
    if (recordingSize) {
      api.stopRecording();
      return;
    }
    const sizeSel = $<HTMLSelectElement>('rec-size').value;
    // As the view: no size given, so record() takes the view area's own.
    const [width, height] = sizeSel === 'view' ? [undefined, undefined] : sizeSel.split('x').map(Number);
    const every = Math.max(1, Math.round(+$<HTMLInputElement>('rec-every').value || 1));
    const dt = +$<HTMLInputElement>('rec-dt').value;
    if (cadence.value === 'time' && !(dt > 0)) {
      setHud('Movie: the frame interval must be more than 0');
      return;
    }
    // Auto: no bitrate given, so the movie takes its default for the size and the frame rate.
    const mbit = +$<HTMLSelectElement>('rec-bitrate').value;
    const wasRunning = running;
    recordProgress = null;
    api
      .record({
        ...(cadence.value === 'time' ? { dt } : { every }),
        fps: +$<HTMLSelectElement>('rec-fps').value,
        width,
        height,
        bitrate: mbit > 0 ? mbit * 1e6 : undefined,
        overlays: { colorbar: $<HTMLInputElement>('rec-colorbar').checked, triad: $<HTMLInputElement>('rec-triad').checked, status: $<HTMLInputElement>('rec-status').checked },
      })
      .then((r) => {
        notice(`saved ${r.name}: ${r.frames} frames, ${r.seconds.toFixed(1)} s, ${(r.bytes / 1e6).toFixed(1)} MB`);
        if (wasRunning && r.stopped === 'stopped' && !finished) {
          running = true;
          syncControls();
        }
      })
      .catch((e) => reportError(e));
  });
}

/** Reflect cfg and view into the controls. */
function syncControls() {
  const two = cfg.ndim < 3;
  // A 1D run says so over the view.
  const chip = oneDChip(cfg);
  const chipEl = $('oned-chip');
  chipEl.hidden = !chip;
  if (chip) {
    chipEl.innerHTML = `<b>1D</b> · ${chip.text}`;
    chipEl.title = chip.title;
  }
  // A screenshot of the volume is of the view as it is: no region to choose.
  $('shot-region').hidden = !flatView();
  // The preset menus show a choice that waits for Run (Autoload off), else the run's own.
  const g = pending?.pick ?? { id: cfg.problem.id, n: cfg.n, ndim: cfg.ndim, levels: cfg.amrLevels };
  const gProblem = problemOf(g.id);
  showProblem(g.id);
  syncProblemParams();
  $<HTMLInputElement>('blocks').checked = view.showBlocks;
  // Contours: lines in the slice and projection views, isosurfaces in the volume view.
  const marching = !two && view.mode === 'volume';
  $<HTMLInputElement>('contours').checked = view.contours;
  $<HTMLInputElement>('ncont').value = String(view.nContours);
  $<HTMLInputElement>('ncont').disabled = !view.contours;
  $<HTMLInputElement>('fog').checked = view.fog;
  $<HTMLInputElement>('fog').disabled = !marching;
  $<HTMLInputElement>('exact').checked = view.exact;
  $<HTMLInputElement>('exact').disabled = !marching;
  // Fog and Exact are settings of the isosurfaces: shown under Contours while it is on.
  $('fog-check').hidden = $('exact-check').hidden = !view.contours;
  $<HTMLInputElement>('box').checked = view.box;
  $<HTMLInputElement>('box').disabled = !marching;
  // Turning with the frame: a 3D run in a rotating frame (not a shearing box), in the volume view.
  $('row-turnframe').hidden = !frameOmega();
  $<HTMLInputElement>('turnframe').checked = view.turnWithFrame;
  $<HTMLInputElement>('turnframe').disabled = !marching;
  $<HTMLInputElement>('isoa').value = String(logPos(view.surfaceOpacity, SURFACE_RANGE));
  $<HTMLInputElement>('isoa').disabled = !marching || !view.contours;
  // Field lines: only with a magnetic field, and not through a projection.
  // (A view that holds one field loads the face field when they are asked for: settleView.)
  const magnetic = !!cfg.problem.mhd;
  $('flines-check').hidden = $('row-nflines').hidden = !magnetic;
  $<HTMLInputElement>('flines').checked = view.fieldLines;
  $<HTMLInputElement>('flines').disabled = view.mode === 'projection' && !two;
  $<HTMLInputElement>('nflines').value = String(view.fieldLineCount);
  $<HTMLInputElement>('nflines').disabled = !view.fieldLines;
  $<HTMLInputElement>('blocks').disabled = cfg.amrLevels === 0;
  // A 2D run is always shown as its plane, and has no self-gravity.
  $<HTMLSelectElement>('mode').disabled = two;
  $<HTMLSelectElement>('axis').disabled = two;
  // A projection looks through the whole box: there is no plane to place.
  $<HTMLInputElement>('slicepos').disabled = two || view.mode === 'projection';
  const axisLabel = view.mode === 'projection' && !two ? 'Line of sight' : 'Slice normal';
  const axisRow = $('row-axis');
  if (axisRow.querySelector('span')!.textContent !== axisLabel) axisRow.querySelector('span')!.textContent = axisLabel;
  axisRow.title = view.mode === 'projection' && !two ? 'The axis the projection sums along' : 'The axis the slice is perpendicular to';
  const self = isSelfGravity(cfg.gravity);
  $('sum-time').textContent = `${Number.isFinite(cfg.tEnd) ? `to t = ${+cfg.tEnd.toPrecision(6)}` : 'no end'}${ckpt.on ? `, checkpoints every ${+ckpt.every.toPrecision(6)}` : ''}`;
  // What the Setup dialog holds, at a glance: the grid and the gas.
  $('setup-sum').textContent = `${cfg.viewOnly && !pending ? `View only${cfg.restart && viewNote ? ` (${viewNote})` : ''} · ` : ''}${gridLabel(g.n, g.ndim)}${g.levels ? `, ${g.levels} level${g.levels > 1 ? 's' : ''}` : ''} · ${gProblem.mhd ? 'MHD' : 'Gas'} · ${EOS_LABELS[pending?.pick ? gProblem.eos : cfg.eos]}`;
  showCheckpoints();
  // Fields this run cannot show: a z velocity in 2D, heating without a cooling function, and
  // refinement levels on a uniform grid.
  const fieldSel = $<HTMLSelectElement>('field');
  // Nor a potential without gravity, nor a Jeans length without self-gravity.
  const pulled = cfg.gravity !== 'off' || cfg.gext.some((g) => g !== 0) || !!cfg.problem.masses || placed.length > 0;
  // A barotropic gas carries no internal energy of its own, so it has no temperature or entropy
  // to show, and nothing heats or cools without a cooling function.
  const noHeat = (i: number) => ((i === 1 || i === 9) && !eosHasEnergy(cfg.eos)) || (i === 10 && (cfg.cooling === 'off' || !eosHasEnergy(cfg.eos)));
  const noField = (i: number) => (i === 7 && two) || (i === 14 && cfg.amrLevels === 0) || (i === 21 && (cfg.amrLevels === 0 || !(cfg.timeBins > 0)))
    || (i === 15 && !pulled) || (i === 17 && !self) || noHeat(i)
    || (!!cfg.viewOnly && VIEW_LOST.has(i)) || (!!cfg.viewBaked && UNBAKEABLE.has(i));
  // What the run cannot show is left out of the menu altogether, and so is a group with nothing
  // left in it (the magnetic and radiation groups come and go with their physics: syncSpeciesFields).
  for (const o of fieldSel.options) o.disabled = o.hidden = noField(+o.value);
  if (noField(view.field)) view.field = 0;
  syncSpeciesFields(fieldSel);
  for (const g of fieldSel.querySelectorAll('optgroup')) {
    if (g.label !== 'Magnetic field' && g.label !== 'Radiation') g.hidden = [...g.children].every((o) => (o as HTMLOptionElement).hidden);
  }
  $<HTMLInputElement>('log').disabled = LINEAR_FIELDS.has(view.field);
  const unitSel = $<HTMLSelectElement>('units');
  unitSel.value = cfg.problem.units ? unitSys : 'code';
  unitSel.disabled = !cfg.problem.units;
  const unitLabel = units().label(viewMeasure());
  const fieldName = viewFieldName();
  $('sum-view').textContent = unitLabel ? `${fieldName}, ${unitLabel}` : fieldName;
  showRange();
  $<HTMLInputElement>('tend').value = Number.isFinite(cfg.tEnd) ? String(cfg.tEnd) : '';
  $('blurb').textContent = pending
    ? pending.pick
      ? `Not loaded yet: press Run to load ${gProblem.label}.`
      : 'Changed, not loaded yet: press Run to start again with the new parameters.'
    : cfg.restart ? `${cfg.restart.tool ? 'A star’s magnetic field was added' : 'Loaded from a file'}, at t = ${num(cfg.restart.time)}. Reset returns to that state.` : cfg.problem.blurb;
  $<HTMLSelectElement>('mode').value = two ? 'slice' : view.mode;
  $<HTMLSelectElement>('field').value = view.field === MASS_FRACTION_FIELD ? `${MASS_FRACTION_FIELD}:${view.species}` : String(view.field);
  $<HTMLSelectElement>('axis').value = String(two ? 2 : view.sliceAxis);
  // The slider places the slice, or in the volume view sets the least density drawn.
  const cutting = !two && view.mode === 'volume';
  const slider = $<HTMLInputElement>('slicepos');
  slider.max = cutting ? '1' : '0.999';
  slider.value = String(cutting ? view.minRho : view.slicePos);
  $('row-slicepos').title = cutting
    ? 'Volume view: gas thinner than this is not drawn, whatever the field shown. The slider runs from the smallest density in the box to the largest, on a log scale'
    : 'Where the slice cuts the box along its normal';
  showDensityCut();
  $<HTMLInputElement>('log').checked = view.log;
  $<HTMLInputElement>('opacity').value = String(logPos(view.opacity, OPACITY_RANGE));
  // Opacity belongs to the volume view: a slice is drawn solid.
  const solid = two || view.mode !== 'volume';
  $<HTMLInputElement>('opacity').disabled = solid;
  syncOpacity();
  editor.setEditable(!solid);
  $<HTMLSelectElement>('cmap').value = editor.colormap;
  $('run').textContent = running ? 'Pause' : 'Run';
  // A checkpoint open for viewing does not step.
  const looking = !!cfg.viewOnly && !pending;
  for (const id of ['run', 'step', 'reset'] as const) {
    const b = $<HTMLButtonElement>(id);
    b.disabled = looking;
    if (looking) b.title = 'This checkpoint is open for viewing only: Import state loads it to run, if this device has the GPU memory';
  }
  showLoad();
  syncTools();
}

/** The Load button: there with Autoload off, and ready when a change waits. */
function showLoad() {
  const b = $<HTMLButtonElement>('load');
  b.hidden = autoload;
  b.disabled = !pending;
  b.classList.toggle('waiting', !!pending);
}

/** Factor from a preset parameter's code value to the value its field shows (ProblemParam.unit). */
const paramUnit = (q: ProblemParam) => (!physicalParams() ? 1 : q.unitChoices ? 1 / unitChoice(q).value : q.unit ? q.unit.factor(cfg.params) : 1);
/** The preset's fields are shown in physical units: the run's readouts are, and it is the run's preset. */
const physicalParams = () => !waitingProblem() && units().physical && unitSys !== 'code';
/** The unit picked for each field that has a menu of them (ProblemParam.unitChoices), by its index. */
const unitPicks: Record<string, number> = {};
const unitChoice = (q: ProblemParam) => q.unitChoices![unitPicks[q.id] ?? 0] ?? q.unitChoices![0];

/** The preset's own controls (for example mass ratio and impact parameter). */
function syncProblemParams() {
  const box = $('pparams');
  // A preset picked and not loaded yet (Autoload off) shows its own controls, not the run's.
  const shown = waitingProblem() ?? cfg.problem;
  const values = () => {
    const w = waitingProblem();
    return w ? waitingParams(w) : cfg.params;
  };
  const defs = shown.params ?? [];
  if (box.dataset.problem !== shown.id) {
    box.dataset.problem = shown.id;
    const range = (q: ProblemParam) => `min="${q.min}" max="${q.max}"${q.step ? ` step="${q.step}"` : ''}`;
    box.innerHTML = defs
      .map((q) => {
        const tip = q.title ?? '';
        if (q.kind === 'checkbox') return `<label class="check pcheck" id="pp-${q.id}-row" title="${tip}"><input id="pp-${q.id}" type="checkbox" />${q.label}</label>`;
        if (q.kind === 'slider') {
          return `<div class="f pslider" id="pp-${q.id}-row" title="${tip}"><span>${q.label}</span><div><input id="pp-${q.id}-slide" type="range" ${range(q)} aria-label="${q.label}" />${number(`pp-${q.id}`, range(q))}</div></div>`;
        }
        if (q.unitChoices) {
          const menu = `<select id="pp-${q.id}-unit" aria-label="Unit of ${q.label}">${q.unitChoices.map((c, i) => `<option value="${i}">${c.label}</option>`).join('')}</select>`;
          return `<div class="f punit" id="pp-${q.id}-row" title="${tip}"><span>${q.label}</span><div>${number(`pp-${q.id}`, range(q), q.factor)}${menu}</div></div>`;
        }
        return field(q.label, number(`pp-${q.id}`, range(q), q.factor), `pp-${q.id}-row`, tip);
      })
      .join('');
    for (const q of defs) {
      const el = $<HTMLInputElement>(`pp-${q.id}`);
      const slide = q.kind === 'slider' ? $<HTMLInputElement>(`pp-${q.id}-slide`) : null;
      // Dragging the slider shows its value in the field; letting go applies it.
      slide?.addEventListener('input', () => (el.value = slide.value));
      slide?.addEventListener('change', () => {
        el.value = slide.value;
        el.dispatchEvent(new Event('change'));
      });
      // Another unit shows the same value in it: nothing about the run changes.
      const unitSel = q.unitChoices ? $<HTMLSelectElement>(`pp-${q.id}-unit`) : null;
      unitSel?.addEventListener('change', () => {
        unitPicks[q.id] = +unitSel.value;
        syncControls();
      });
      el.addEventListener('change', () => {
        const into = values();
        const v = q.kind === 'checkbox' ? (el.checked ? 1 : 0) : el.value === '' || Number.isNaN(+el.value) ? into[q.id] : +el.value / paramUnit(q);
        into[q.id] = Math.min(q.max, Math.max(q.min, v));
        // A preset that waits for Run takes the value when it loads.
        if (waitingProblem()) return syncControls();
        setParams({ [q.id]: into[q.id] });
        syncControls();
      });
    }
  }
  for (const q of defs) {
    $(`pp-${q.id}-row`).hidden = !!q.dims && q.dims !== (pending?.pick?.ndim ?? cfg.ndim);
    const el = $<HTMLInputElement>(`pp-${q.id}`);
    // A field with a unit of its own shows it, and its value and range, in physical units.
    const k = paramUnit(q);
    if (q.unitChoices) {
      const unitSel = $<HTMLSelectElement>(`pp-${q.id}-unit`);
      unitSel.hidden = !physicalParams();
      unitSel.value = String(q.unitChoices.indexOf(unitChoice(q)));
      el.min = String(+(q.min * k).toPrecision(6));
      el.max = String(+(q.max * k).toPrecision(6));
    }
    if (q.unit) {
      $(`pp-${q.id}-row`).querySelector('span')!.textContent = k !== 1 ? `${q.label} (${q.unit.label})` : q.label;
      el.min = String(+(q.min * k).toPrecision(6));
      el.max = String(+(q.max * k).toPrecision(6));
      el.step = q.step ? String(+(q.step * k).toPrecision(6)) : 'any';
    }
    const now = values();
    if (q.kind === 'checkbox') el.checked = !!now[q.id];
    else if (document.activeElement !== el) el.value = String(+(now[q.id] * k).toPrecision(6));
    const slide = q.kind === 'slider' ? $<HTMLInputElement>(`pp-${q.id}-slide`) : null;
    if (slide && document.activeElement !== slide) slide.value = String(now[q.id]);
  }
  syncFeatured();
}

/**
 * The setup's featured options that are settings of the run (a viscosity, an equation of
 * state), beside its parameters above: each the control its option has in the Setup dialog,
 * from the same table, applied as the dialog applies it.
 */
function syncFeatured() {
  const box = $('pfeatured');
  // A preset that waits for Run shows its parameters; its settings come with it when it loads.
  const spec = waitingProblem() ? undefined : cfg.problem.setup;
  const feats = (spec?.featured ?? []).map(featuredOf).filter((f) => !featuredPaths(f).every(isParamPath));
  const usable = feats.flatMap((f) => {
    const path = featuredPaths(f)[0];
    const o = spec ? optionAt(spec, path) : undefined;
    return o && (o.control.type === 'number' || o.control.type === 'select' || o.control.type === 'check') ? [{ f, path, o }] : [];
  });
  const key = `${cfg.problem.id}|${customVersion()}|${usable.map((u) => u.path).join('|')}`;
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.replaceChildren();
    usable.forEach(({ f, path, o }, i) => {
      const id = `pf-${i}`;
      const label = f.label ?? o.label;
      const tip = f.tip ?? o.tip;
      const c = o.control;
      const wrap = document.createElement(c.type === 'check' ? 'label' : 'div');
      wrap.title = tip;
      let el: HTMLInputElement | HTMLSelectElement;
      if (c.type === 'select') {
        el = document.createElement('select');
        for (const k of c.choices) el.add(new Option(k.label, String(k.value)));
      } else {
        el = document.createElement('input');
        el.type = c.type === 'check' ? 'checkbox' : 'number';
        if (c.type === 'number') {
          el.step = c.step !== undefined ? String(c.step) : 'any';
          if ((f.min ?? c.min) !== undefined) el.min = String(f.min ?? c.min);
          if ((f.max ?? c.max) !== undefined) el.max = String(f.max ?? c.max);
          if (f.factor ?? c.factor) el.dataset.factor = String(f.factor ?? c.factor);
        }
      }
      el.id = id;
      if (c.type === 'check') {
        wrap.className = 'check pcheck';
        wrap.append(el, label);
      } else {
        wrap.className = 'f';
        const span = document.createElement('span');
        span.textContent = label;
        wrap.append(span, el);
      }
      el.addEventListener('change', () => {
        const edited = cloneSetup(runningSpec(cfg));
        let v: unknown;
        if (c.type === 'check') v = (el as HTMLInputElement).checked;
        else if (c.type === 'select') v = c.choices.find((k) => String(k.value) === el.value)?.value;
        else if (c.type !== 'number') return;
        else if (el.value === '' || Number.isNaN(+el.value)) v = c.nullable ? null : undefined;
        else v = Math.min(c.max ?? Infinity, Math.max(c.min ?? -Infinity, +el.value));
        if (v === undefined) return syncControls();
        for (const q of featuredPaths(f)) setPath(edited, q, v);
        void applySpec(edited);
      });
      box.append(wrap);
    });
  }
  if (!spec || !usable.length) return;
  const now = runningSpec(cfg);
  usable.forEach(({ path, o }, i) => {
    const el = $<HTMLInputElement>(`pf-${i}`);
    const v = valueAt(now, path);
    if (o.control.type === 'check') el.checked = !!v;
    else if (document.activeElement !== el) el.value = v === null || v === undefined ? '' : typeof v === 'number' ? String(+v.toPrecision(6)) : String(v);
  });
}

// Rebuilds are asynchronous (shader compilation). They run one at a time, in the order the
// controls were touched, so a change made while another is compiling is never dropped.
let queue: Promise<void> = Promise.resolve();
/** Counts builds, so a survival check (below) belongs to the build that started it. */
let buildGen = 0;
/** How long a new run must last before its allocation counts as safe. */
const SURVIVAL_MS = 4000;
function enqueue(job: () => Promise<void>): Promise<void> {
  queue = queue.then(async () => {
    building = true;
    resetCompileCount();
    try {
      await job();
    } catch (e) {
      // A build that failed with an error left the page alive: not a crash to learn from.
      allocationSurvived();
      forgetCompileSetup();
      fail(e);
    } finally {
      building = false;
      if (cfg) syncControls();
    }
  });
  return queue;
}

/** The grid a preset gets for the one asked for: what it supports and what fits in GPU memory. */
function resolveGrid(problem: Problem, n: number, ndim: 1 | 2 | 3, levels: number) {
  if (ndim === 1 && !problem.supports1d) ndim = 2;
  if (ndim === 2 && !problem.supports2d) ndim = 3;
  // An axisymmetric (R, z) or spherical (r, theta) problem is two-dimensional (or one-dimensional
  // in R or r), whatever was asked.
  if ((problem.geometry === 'cylindrical' || problem.geometry === 'spherical') && ndim === 3) ndim = 2;
  const allowed = resFor(ndim);
  if (!allowed.includes(n)) n = ndim === 1 ? 4096 : ndim === 2 ? 512 : 128;
  // Refinement: only where the preset allows it, and no deeper than the resolution permits.
  levels = supportsAmr(problem) ? Math.min(levels, maxAmrLevels(n)) : 0;
  // Enough levels for the root blocks to fit in GPU memory; a grid no level count fits goes
  // down to the largest uniform one.
  if (levels > 0 && !amrFits(n, ndim, levels)) levels = Math.max(levels, minAmrLevels(n, ndim));
  if (levels > 0 && !amrFits(n, ndim, levels)) levels = 0;
  if (levels === 0) n = Math.min(n, uniformMax(ndim));
  return { n, ndim, levels };
}

/** A new problem or grid from the menus: built now, or (Autoload off) left waiting for Run. */
function load(id: string, n: number, ndim: 1 | 2 | 3, levels: number, keepPhysics = false) {
  if (autoload) return void build(id, n, ndim, levels, keepPhysics);
  // Physics choices carry over only if every change waiting would have kept them.
  // Parameters set for this preset while it waited stay, through a change of its grid.
  // So do settings changed for it in the Setup dialog.
  const prev = pending?.pick?.id === id ? pending.pick : undefined;
  pending = { pick: { id, ...resolveGrid(problemOf(id), n, ndim, levels), params: prev?.params, settings: prev?.settings }, keepPhysics: keepPhysics && (pending?.pick ? pending.keepPhysics : true) };
  running = false;
  if (cfg) syncControls();
}

/** New initial conditions for the present run: started again now, or (Autoload off) on Run. */
function restartOrWait(replay = false) {
  if (autoload) return void restartRun(replay);
  pending ??= { keepPhysics: true };
  running = false;
  showLoad();
}

/**
 * Load what waits for Run. `play`: whether the run then plays; left out, as Autoplay has it.
 */
function loadPending(play?: boolean) {
  const p = pending;
  if (!p) return;
  pending = null;
  const job = p.pick ? build(p.pick.id, p.pick.n, p.pick.ndim, p.pick.levels, p.keepPhysics, undefined, undefined, false, p.pick.params, p.pick.settings) : restartRun(true);
  job.then(() => {
    if (play !== undefined && sim) running = play;
    if (cfg) syncControls();
  });
}

/**
 * Build a simulation for the selected preset and resolution. `hold`: only set it up (settings,
 * view, panel), and leave the simulation itself waiting for Run or Load, as Autoload off has
 * it: the first load of a page with Autoload off.
 */
function build(problemId: string, n: number, ndim: 1 | 2 | 3, levels: number, keepPhysics = false, link?: LinkOverrides, snap?: Snapshot, hold = false, params?: Record<string, number>, settings?: { setting: Setting; value: unknown }[]) {
  pending = null; // whatever waited for Run is superseded
  return enqueue(async () => {
    // A custom setup's mesh files: fetched (URLs) or found among those this browser keeps.
    const custom = problemId === CUSTOM_ID ? customSetup() : null;
    if (custom) {
      const missing = await ensureMeshFiles(custom.spec);
      if (missing.length) throw new Error(`the setup needs mesh files this page does not have: ${missing.join(', ')}. Drop them on the Setup dialog`);
    }
    const problem = problemOf(problemId);
    ({ n, ndim, levels } = resolveGrid(problem, n, ndim, levels));
    // View only: the file's mesh merged to its level, whatever the presets' grids are (at least
    // one level, so the blocks are the adaptive mesh's).
    const look = snap?.view;
    if (look && snap) {
      levels = Math.max(1, look.level);
      n = snap.rootSize[0] << levels;
      ndim = snap.ndim;
    }
    const next = configFor(problem, n, { ndim, amrLevels: levels });
    if (!Number.isFinite(next.tEnd)) next.tEnd = Infinity;
    if (keepPhysics && cfg && cfg.problem.id === problem.id && !cfg.restart && cfg.ndim === ndim) {
      // Same preset at a new resolution: keep the user's physics choices.
      const { n: _n, problem: _p, amrLevels: _l, restart: _r, ...rest } = cfg;
      // The most V-cycles a step runs depends on the mesh: left at the old mesh's default, it
      // takes the new one's.
      if (rest.vcycles === configFor(problem, cfg.n, { ndim, amrLevels: cfg.amrLevels }).vcycles) rest.vcycles = next.vcycles;
      Object.assign(next, rest);
      // A 3D load shows the gas as a volume, whatever view the last run was left in.
      if (ndim === 3 && !link) view.mode = problem.view?.mode ?? 'volume';
    } else {
      const dv = viewDefaults(problem, ndim);
      Object.assign(view, { field: dv.field, log: dv.log, mode: problem.view?.mode ?? 'volume', sliceAxis: dv.sliceAxis, pitch: dv.pitch, zoom: 1, cx: 0.5, cy: 0.5, tx: 0.5, ty: 0.5, tz: 0.5, minRho: 0, fieldLines: dv.fieldLines, fieldLineCount: dv.fieldLineCount });
    }
    if (params && !(keepPhysics && cfg && cfg.problem.id === problem.id)) {
      // Parameters set while the preset waited for Run (Autoload off).
      Object.assign(next.params, params);
      if (typeof problem.units === 'function') unitConstants(next);
    }
    // Settings changed in the Setup dialog while it waited.
    if (settings && !(keepPhysics && cfg && cfg.problem.id === problem.id)) for (const s of settings) setSetting(next, s.setting, s.value);
    if (link) {
      // Settings carried by a shared link, on top of the preset's defaults; its mirrors, as far
      // as the preset allows them, with their planes' faces.
      Object.assign(next, link.cfg);
      if (link.cfg.symmetry) Object.assign(next, withSymmetry(next, link.cfg.symmetry));
      Object.assign(next.params, link.params);
      // Units that follow the preset's parameters: the equation of state's physical constants are the link's parameters', not the defaults'.
      if (typeof problem.units === 'function') unitConstants(next);
      Object.assign(next.eosParams, link.eosParams);
      Object.assign(next.coolParams, link.coolParams);
      next.opacityParams = { ...next.opacityParams, ...link.opacityParams };
      // Radiation switched on or off by the link: the equation of state's radiation term follows.
      Object.assign(next, syncRadiationEos(next));
      next.fluff = { ...next.fluff, ...link.fluff };
      next.startDamping = { ...next.startDamping, ...link.startDamping };
      next.driving = { ...next.driving, ...link.driving };
      if (ndim === 2) {
        // (An axisymmetric or spherical run has the self-gravity of its rings or shells.)
        if (isSelfGravity(next.gravity) && problem.geometry !== 'cylindrical' && problem.geometry !== 'spherical') next.gravity = 'off';
        next.bcLo = [next.bcLo[0], next.bcLo[1], 0];
        next.bcHi = [next.bcHi[0], next.bcHi[1], 0];
      } else if (ndim === 1) {
        // (A 1D run keeps its self-gravity, a prefix sum; y and z are periodic but a spherical
        // run's poles: configFor.)
        const poles = problem.geometry === 'spherical';
        next.bcLo = [next.bcLo[0], poles ? next.bcLo[1] : 0, 0];
        next.bcHi = [next.bcHi[0], poles ? next.bcHi[1] : 0, 0];
      }
      Object.assign(view, link.view);
      if (link.auto === false) $<HTMLInputElement>('auto').checked = false;
      if (link.colormap || link.opacity) {
        editor.set(link.opacity ?? editor.points, link.colormap ?? editor.colormap);
        pushTransfer();
      }
      // Profile lines the link carries: on show once the run is built.
      if (link.profiles) setLinkProfiles(link.profiles, next);
    }
    // The box is as wide as the run's parameters make it (a preset with a size of its own).
    next.problem = sizedProblem(problem, next.params);
    // View only: never stepped, an atlas just large enough, and no gravity to solve.
    if (look) Object.assign(next, { viewOnly: true, viewBaked: !!look.baked, viewFaces: !!look.faces, amrBudget: look.nodes, gravity: 'off' });
    if (!look?.baked) closeViewing();
    viewNote = look && look.level < look.fileLevel ? `to level ${look.level} of ${look.fileLevel}` : '';
    // From a file: the same preset, started from the saved state instead of its own.
    if (snap) Object.assign(next, restartFrom(snap, next.problem, levels > 0, n >> levels));
    // A new problem brings its own checkpoint interval; the clock starts at the snapshot's time, if any.
    if (cfg?.problem.id !== problem.id) {
      ckpt.every = defaultCheckpointDt(problem);
      // A movie's frame interval: a 600th of the problem's dynamical time.
      const recDt = $<HTMLInputElement>('rec-dt');
      recDt.value = String(defaultFrameDt(problem));
      recDt.dispatchEvent(new Event('input'));
    }
    next.tStop = ckpt.on ? nextCheckpoint(next.restart?.time ?? 0, ckpt.every) : undefined;
    if (hold) {
      // Its settings are kept (Load takes them as a change of grid that keeps the physics).
      cfg = next;
      if (ndim < 3) view.sliceAxis = 2;
      pending = { pick: { id: problemId, n, ndim, levels }, keepPhysics: true };
      running = false;
      setLoadStage('');
      return;
    }
    const old = sim;
    sim = undefined;
    stroke = null;
    draft = null;
    pendingBrush = null;
    sources = [];
    if (old) {
      setLoadStage('Freeing the last run');
      old.destroy();
      // Free the old run's memory before the new one takes its own: the two together are what
      // kills the page on a phone. Waiting for its work to finish is not enough: the browser
      // releases destroyed textures a little later (measured on Chrome under D3D12: both runs
      // were held for about a second), so wait for a second round of work and a moment more.
      await gpu.device.queue.onSubmittedWorkDone();
      gpu.device.queue.submit([]);
      await gpu.device.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 250));
    }
    cfg = next;
    if (ndim < 3) view.sliceAxis = 2;
    // The display's pipelines compile alongside the run's own; attach then finds them ready.
    renderer.fieldPipelines(fieldShaderFor(cfg), cfg, cfg.amrLevels > 0 && cfg.ndim === 3).catch(() => {});
    // Shown as soon as it can be drawn; on the adaptive mesh it starts stepping once its update
    // pass has compiled (gpu/device.ts), the slowest part of a first load.
    const s = await createSim(gpu, cfg, { deferUpdate: true }).finally(() => setLoadStage(''));
    setLoadStage('Preparing the display');
    await renderer.attach(s).finally(() => setLoadStage(''));
    sim = s;
    beginSession(!snap, snap ? 'the run started from an imported file' : undefined);
    // The page survived the allocation for a few seconds: it is safe (gpu/memory.ts).
    const gen = ++buildGen;
    setTimeout(() => {
      if (gen === buildGen && sim === s) allocationSurvived();
    }, SURVIVAL_MS);
    finished = false;
    // A freshly loaded problem starts playing, unless it waits for the user (an empty box); a
    // loaded state waits to be looked at.
    running = autoplay && !headless && !snap && !problem.startPaused;
    lastDiag = undefined;
    batch = 1;
    msPerStep = busyMs = busySteps = 0;
    resuming = false; // a hold on the last run's mesh does not carry over
  });
}

/**
 * The .athdf reader and writer, loaded on first use. A failed load is not kept, so the next
 * try starts over; one that fails because this build's files are gone from the server (a new
 * release since the page was opened) says so.
 */
let athdfLoading: Promise<typeof import('../io/athdf')> | null = null;
function athdfModule() {
  athdfLoading ??= import('../io/athdf').then(async (m) => {
    await m.warm();
    return m;
  });
  athdfLoading.catch(() => (athdfLoading = null));
  return athdfLoading.catch((e) => {
    throw new Error(
      `The file code could not be loaded (${(e as Error)?.message ?? e}). If Shaderous was updated since this page was opened, this build's files are gone: only a reload fetches the new ones, and a reload starts the run again.`,
    );
  });
}
// Fetched a little after the page starts, while this build's files are sure to be there.
if (!headless) setTimeout(() => void athdfModule().catch(() => {}), 10_000);

/** The state of the gas as an .athdf file (see io/athdf.ts), and its time. */
async function athdfOf(s: Simulation) {
  const { snap } = await snapshotOf(s);
  const { writeAthdf } = await athdfModule();
  // gzip level 1: nearly all of the saving of higher levels (files are 8% to 86% of the raw
  // size, depending on how smooth the flow is) for the least time.
  const bytes = await writeAthdf(snap, undefined, 1);
  return { time: snap.time, bytes, stem: `shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}` };
}

/** The state of the gas as a snapshot, with the stem of its file's name. */
async function snapshotOf(s: Simulation) {
  const snap = await capture(s, { program: 'shaderous', build: BUILD, query: linkQuery(), setup: cfg.problem.setup });
  return { snap, stem: `shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}` };
}

/** Save the state of the gas to an .athdf file. */
function exportState() {
  const button = $('export');
  return enqueue(async () => {
    if (!sim) return;
    button.textContent = 'Exporting…';
    try {
      const { time, bytes, stem } = await athdfOf(sim);
      download(`${stem}-t${+time.toPrecision(4)}.athdf`, bytes);
    } finally {
      button.textContent = 'Export state';
    }
  });
}

/** A problem's checkpoint interval, before the user changes it. */
const defaultCheckpointDt = (p: SimConfig['problem']) => p.checkpointDt ?? (Number.isFinite(p.tEnd) ? p.tEnd / 10 : 1);

/**
 * Hold the clock at the first checkpoint after time `t` (or release it, with checkpoints off).
 * Writes the uniform block of a running simulation; a new one picks cfg.tStop up as it is built.
 */
function armCheckpoints(t: number) {
  if (!cfg) return;
  cfg.tStop = ckpt.on ? nextCheckpoint(t, ckpt.every) : undefined;
  if (sim) {
    sim.cfg = cfg;
    sim.writeParams();
  }
  showCheckpoints();
}

/** Write the checkpoint the clock of `s` is held at, then let the run go on to the next. */
function saveCheckpoint(s: Simulation) {
  ckpt.busy = true;
  return enqueue(async () => {
    try {
      if (sim !== s || !ckpt.on || !ckpt.sink) return;
      const { time, bytes, stem } = await athdfOf(s);
      // Numbered, so the files sort in time order whatever the interval.
      const k = String(checkpointIndex(time, ckpt.every)).padStart(4, '0');
      await ckpt.sink.write(`${stem}-ckpt${k}-t${+time.toPrecision(4)}.athdf`, bytes);
      ckpt.written++;
      if (sim === s) armCheckpoints(time);
    } catch (e) {
      // A folder that went away, or permission withdrawn: stop rather than hold the run.
      ckpt.on = false;
      ckpt.error = `Checkpoints stopped: ${(e as Error)?.message ?? e}`;
      armCheckpoints(0);
    } finally {
      ckpt.busy = false;
    }
  });
}

/** The checkpoint controls and the line under them. */
function showCheckpoints() {
  $<HTMLInputElement>('ckpt').checked = ckpt.on;
  $<HTMLInputElement>('ckdt').value = String(+ckpt.every.toPrecision(6));
  const note = $('ckpt-note');
  if (ckpt.error) note.textContent = ckpt.error;
  else if (!ckpt.on) note.textContent = '';
  else {
    const next = cfg?.tStop !== undefined && !(cfg.tStop > cfg.tEnd) ? `, next at t = ${+cfg.tStop.toPrecision(6)}` : '';
    note.textContent = `Saving to ${ckpt.sink?.where}: ${ckpt.written} so far${next}.`;
  }
}

/** Switch checkpoints on (asking for a folder, if the browser can write to one) or off. */
async function setCheckpoints(on: boolean) {
  ckpt.error = '';
  if (on) {
    let sink: CheckpointSink | null = downloadSink;
    if (canPickFolder()) {
      try {
        sink = await pickFolder();
      } catch {
        sink = downloadSink; // a picker that refuses (in a frame, say): downloads still work
      }
    }
    if (!sink) {
      showCheckpoints(); // cancelled: stay off
      return;
    }
    ckpt.sink = sink;
    ckpt.written = 0;
  }
  ckpt.on = on;
  armCheckpoints(await clockNow());
}

/** The time on the simulation's clock now (the readouts can be a fifth of a second behind). */
async function clockNow(): Promise<number> {
  const d = sim && !sim.preparing ? await sim.diagnostics() : undefined;
  return d?.t ?? lastDiag?.t ?? cfg?.restart?.time ?? 0;
}

/** Start a run from an .athdf file, chosen in the page. */
async function importState(file: File) {
  const button = $('import');
  button.textContent = 'Importing…';
  try {
    const warnings = await importAthdf(new Uint8Array(await file.arrayBuffer()));
    if (warnings.length) alert(`Loaded ${file.name}, with differences:\n\n${warnings.map((w) => `• ${w}`).join('\n\n')}`);
  } catch (e) {
    // A file that cannot be used is not a fault of the page: say so and carry on.
    alert(`Could not load ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    button.textContent = 'Import state';
  }
}

/**
 * Start a run from the bytes of an .athdf file: one of ours with its settings, or another
 * code's. Returns what an older file of ours could not carry over as written.
 */
async function importAthdf(bytes: Uint8Array, opts: ImportOptions = {}): Promise<string[]> {
  const { readAthdf } = await athdfModule();
  return importSnapshot(await readAthdf(bytes), opts);
}

/**
 * How a snapshot is imported. `switchBurn`: the file is of a run that watched for burning (he4,
 * c12 and o16 alone, burn/watch.ts), and it continues with all the network's species, the
 * others starting at zero.
 */
interface ImportOptions {
  switchBurn?: boolean;
}

/** What a view-only load merged, for the setup summary ('' when nothing was). */
let viewNote = '';

/**
 * Open an .athdf file for viewing only (plans/view-only.md): read in a worker without copying
 * it, its finest levels merged into coarser ones until the blocks fit this device's GPU memory
 * budget, and shown by an adaptive mesh that holds only what the display reads.
 */
/**
 * A checkpoint open for viewing with one field held (SimConfig.viewBaked): its file, kept open
 * in the reader's worker to bake another field from, the level it is merged to, and the field
 * (and species, for a mass fraction) baked now (-1 before the first).
 */
let viewing: { file: ViewFile; name: string; level: number; field: number; species: string | null; faces: boolean } | null = null;
/** The bake under way, if any (settleView). */
let baking: Promise<void> | null = null;
/**
 * What the last bake that failed was for (settleView's `want`), and why: not tried again on its
 * own each frame (which flickered the loading message without end), only when what is shown
 * changes, or when asked (control, view).
 */
let bakeFailed: { want: string; message: string } | null = null;
/** Fields a view that holds one field cannot bake: they need neighbors, or what a checkpoint lacks. */
const UNBAKEABLE = new Set([11, 12, 13, 15, 20, 21, 27, 30, 31, 32, 33, 34, 35, 36]);
/** Fields no view-only load has: the potential (gravity is off), the shock flag, time bins, burning, radiation. */
const VIEW_LOST = new Set([15, 20, 21, 30, 31, 32, 33, 34, 35, 36]);

function closeViewing() {
  viewing?.file.close();
  viewing = null;
}

/** The species whose mass fraction is shown, by name, or null. */
const shownSpecies = (): string | null => (view.field === 19 ? (cfg.problem.species?.[view.species] ?? null) : null);

/**
 * A view that holds one field: bake the field shown if it is not the one held (on the GPU, from
 * the file, a batch of blocks at a time: AmrSim.bake). Resolves once it holds the field shown.
 */
/** What a view that holds one field is asked to show: its field, species and whether field lines are on (bakeFailed). */
const viewWanted = () => `${view.field} ${shownSpecies()} ${!!view.fieldLines && !!cfg?.problem.mhd}`;

function settleView(): Promise<void> {
  if (baking) return baking;
  if (!cfg?.viewBaked || !viewing || !sim) return Promise.resolve();
  const want = viewWanted();
  const fail = (e: unknown) => {
    bakeFailed = { want, message: e instanceof Error ? e.message : String(e) };
    throw e;
  };
  const v = viewing;
  // Field lines asked for where the face field was not loaded: opened again with it, at the
  // finest level where the faces fit too.
  if (view.fieldLines && cfg.problem.mhd && !cfg.viewFaces) {
    // (Opening it again sets up the view anew, as for a file just opened: what is shown, and how,
    // stays as it was.)
    const shown = { ...view };
    const reopen: Promise<void> = (async () => {
      try {
        await showView(v.file, v.name, { oneField: true, faces: true, settle: false });
        Object.assign(view, shown);
        syncControls();
      } catch (e) {
        view.fieldLines = false;
        alert(`Field lines need the field's faces, which do not fit: ${e instanceof Error ? e.message : String(e)}`);
        return;
      } finally {
        baking = null;
      }
      // (Then the bake of the new mesh, faces and all, as its own step: it would wait on this one.)
      return settleView();
    })();
    baking = reopen;
    return reopen;
  }
  if (UNBAKEABLE.has(view.field)) view.field = 0;
  const species = shownSpecies();
  const faces = !!cfg.viewFaces && !v.faces;
  if (v.field === view.field && v.species === species && !faces) return Promise.resolve();
  const s = sim as Simulation & { bake?: (b: ViewBatch, field: number, species: string | null) => Promise<void>; loadFaces?: (b: ViewBatch) => void };
  const field = view.field;
  baking = (async () => {
    const what = `Reading ${viewFieldName().toLowerCase()}${faces ? ' and the magnetic field' : ''} from the file`;
    setLoadStage(what);
    try {
      let chain = Promise.resolve();
      let failed: unknown = null;
      await v.file.stream(
        v.level,
        bakeVariables(field, species),
        (b) => {
          chain = chain
            .then(async () => {
              await s.bake!(b, field, species);
              if (faces) s.loadFaces!(b);
            })
            .catch((e) => void (failed ??= e));
        },
        (f) => setLoadStage(`${what}: ${Math.round(100 * f)}%`),
        faces,
      );
      await chain;
      if (failed) throw failed;
      v.field = field;
      v.species = species;
      if (faces) v.faces = true;
      bakeFailed = null;
      renderer.invalidate();
    } catch (e) {
      fail(e);
    } finally {
      setLoadStage('');
    }
  })().finally(() => (baking = null));
  return baking;
}

/**
 * Open an .athdf file for viewing only (plans/view-only.md): read in a worker without copying
 * it, and shown either with its state (every field) or holding the one field shown (baked from
 * the file again when the field changes), whichever keeps more of its levels in this device's
 * GPU memory budget; its finest levels merged into coarser ones until it fits. `oneField` picks
 * the second (true) or the first (false) instead.
 */
async function viewState(file: File | string, opts: { oneField?: boolean; faces?: boolean } = {}): Promise<ViewOpened> {
  const button = $('view-file');
  button.textContent = 'Opening…';
  closeViewing();
  const { openViewFile } = await import('../io/view_open');
  const vf = await openViewFile(file);
  try {
    return await showView(vf, typeof file === 'string' ? decodeURIComponent(file.split('/').pop() || 'checkpoint.athdf') : file.name, opts);
  } catch (e) {
    if (viewing?.file === vf) viewing = null;
    vf.close();
    throw e;
  } finally {
    button.textContent = 'Open for viewing';
  }
}

type ViewOpened = { level: number; fileLevel: number; blocks: number; fileBlocks: number; oneField: boolean; faces: boolean };

/**
 * Show a checkpoint open in the reader's worker (viewState). With one field held, the file stays
 * open (`viewing`) to bake another field from; `faces` holds the face field too, for field lines
 * (otherwise they are off until asked for, when it is opened again with them: settleView).
 */
async function showView(vf: ViewFile, name: string, opts: { oneField?: boolean; faces?: boolean; settle?: boolean; maxNodes?: number }): Promise<ViewOpened> {
  const h = vf.header;
  const groups = fileSpeciesGroups(vf.variables);
  const budget = memoryBudget();
  // Planned against the atlas the run will make (AmrSim's atlasShape): its cube of slots, not
  // just the bytes, holds fewer blocks than the budget over a block's bytes.
  const d = gpu.device;
  const most = opts.maxNodes ?? Infinity;
  const withState = planView(h, Math.min(most, AmrSim.viewCapacity(d, h.ndim, viewBlockBytes(h.ndim, vf.hasFaces, groups, groups > 0), groups)));
  const faces = !!opts.faces && vf.hasFaces;
  const oneField = planView(h, Math.min(most, AmrSim.viewCapacity(d, h.ndim, viewBakedBlockBytes(h.ndim) + (faces ? viewFacesBlockBytes(h.ndim) : 0), 0)));
  // (Species that set the equation of state need the composition at each cell, which a baked
  // view does not hold: such a file is shown with its state unless asked otherwise.)
  const baked = opts.oneField ?? (!!oneField && (!withState || oneField.level > withState.level) && groups === 0);
  const plan = baked ? oneField : withState;
  if (!plan) throw new Error(`not even the root grid's blocks fit in this device's GPU memory budget (${(budget / GB).toFixed(1)} GB)`);
  let snap: Snapshot;
  if (baked) {
    snap = await vf.tables(plan.level);
    viewing = { file: vf, name, level: plan.level, field: -1, species: null, faces: false };
  } else {
    if (viewing?.file === vf) viewing = null;
    const reading = `Reading ${name}`;
    setLoadStage(reading);
    try {
      snap = await vf.read(plan.level, (f) => setLoadStage(`${reading}: ${Math.round(100 * f)}%`));
    } finally {
      setLoadStage('');
    }
  }
  snap.view = { level: plan.level, nodes: plan.nodes, fileLevel: plan.fileLevel, baked, faces: baked && faces };
  await importSnapshot(snap);
  // The GPU may not have held that atlas (AmrSim.create falls back to a smaller one): then the
  // mesh lacks some of the plan's blocks, so plan again against the atlas made.
  const made = sim instanceof AmrSim ? sim.cap : Infinity;
  // (Holding the state, the file is read again at the level that fits.)
  if (made < plan.nodes) return await showView(vf, name, { ...opts, oneField: baked, maxNodes: made });
  if (baked) {
    // Without the face field there are no field lines to draw until they are asked for.
    if (!faces) view.fieldLines = false;
    syncControls();
    if (opts.settle !== false) await settleView();
  } else vf.close();
  return { level: plan.level, fileLevel: plan.fileLevel, blocks: plan.leaves, fileBlocks: plan.fileLeaves, oneField: baked, faces: baked ? faces : vf.hasFaces };
}

/** A view that holds one field: the file's own cells under these points, as probe() reports them; null otherwise. */
async function viewProbe(points: number[][]) {
  if (!cfg?.viewBaked || !viewing) return null;
  return viewing.file.probe(points);
}

/** Start the run from a snapshot (one read from an .athdf file), as importAthdf does. */
async function importSnapshot(snap: Snapshot, opts: ImportOptions = {}): Promise<string[]> {
  {
    const cubic = snap.rootSize[0] === snap.rootSize[1] && (snap.ndim < 3 || snap.rootSize[2] === snap.rootSize[0]);
    const width = snap.hi.map((h, a) => h - snap.lo[a]);
    if (!cubic || Math.abs(width[0] - width[1]) > 1e-4 * width[0] || (snap.ndim === 3 && Math.abs(width[0] - width[2]) > 1e-4 * width[0])) {
      throw new Error('only square or cubic domains can be loaded');
    }
    // Our own file carries its settings. Anyone else's keeps the current ones and takes the
    // file's dimensionality and the nearest resolution on offer.
    const own = snap.meta !== undefined;
    let warnings: string[] = [];
    let link: LinkOverrides;
    if (own) {
      let meta = snap.meta!;
      // A file that carries its setup restarts as itself, whatever the preset files have become
      // since: as its preset if it still is that preset, else as a custom setup.
      let { spec } = meta.setup ? readSetup(meta.setup, (id) => PRESET_FILES[id]) : { spec: null };
      if (opts.switchBurn) {
        if (!spec) throw new Error('this file carries no setup to switch to all the network\u2019s species');
        spec = switchedSetup(spec);
        const added = expandSpecies(snap.data);
        warnings.push(`switched to all the network's species: ${added.join(', ')} start at zero`);
      }
      if (spec) {
        const q = new URLSearchParams(meta.query);
        q.set('problem', adopt(spec));
        meta = { ...meta, query: q.toString() };
      }
      const switched = warnings;
      ({ link, warnings } = snapshotSettings(meta, BUILD));
      warnings = [...switched, ...warnings];
    }
    else link = decodeLink(new URLSearchParams(linkQuery()));
    if (!own) {
      const problem = problemOf(link.problem);
      if (snap.ndim === 1 && !problem.supports1d) throw new Error(`this file is 1D, and the ${problem.label} preset is not: pick a 1D preset first`);
      if (snap.ndim === 2 && !problem.supports2d) throw new Error(`this file is 2D, and the ${problem.label} preset is 3D only: pick a 2D preset first`);
      const allowed = resFor(snap.ndim);
      const finest = snap.rootSize[0] << maxLevel(snap);
      const fits = allowed.filter((n) => n <= finest);
      link.ndim = snap.ndim;
      link.n = fits.length ? fits[fits.length - 1] : allowed[0];
      link.amr = maxLevel(snap) > 0 && supportsAmr(problem) ? Math.min(maxLevel(snap), maxAmrLevels(link.n)) : 0;
      if (link.amr === 0) link.n = Math.min(link.n, uniformMax(link.ndim));
      // The run continues to wherever the current settings say; a foreign time means nothing here.
      delete link.cfg.tEnd;
    }
    await build(link.problem, link.n, link.ndim, link.amr, false, link, snap);
    for (const w of warnings) console.warn(`import: ${w}`);
    return warnings;
  }
}

/** The current settings as a query string (see ui/link.ts). */
function linkQuery(): string {
  return encodeLink({ cfg, view, auto: $<HTMLInputElement>('auto').checked, colormap: editor.colormap, opacity: editor.points, setup: customLink?.text || undefined, profiles: linkProfiles() }).toString();
}

// The address bar follows the settings, so the page can be shared or reloaded as it stands.
let lastQuery = '';
let linkFrame = 0;
function syncAddress(force = false) {
  if (!cfg || !sim || building) return;
  // A checkpoint open for viewing is its file's merged mesh, not a setup a link could rebuild.
  if (cfg.viewOnly) return;
  if (!force && ++linkFrame % 30) return;
  // (A setting that reached the run by a path that does not log it: logged late, but logged.)
  noteSettings();
  const q = linkQuery();
  if (q === lastQuery) return;
  lastQuery = q;
  history.replaceState(null, '', `${location.pathname}?${q}${headless ? '&headless=1' : ''}`);
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
      setLoadStage(INIT_STAGE);
      await painted();
      sim.cfg = cfg;
      await sim.reset();
      armCheckpoints(cfg.restart?.time ?? 0);
      clearSources();
      beginSession(!cfg.restart, cfg.restart ? 'the run starts from a saved state' : undefined);
      if (replay) running = autoplay;
      finished = false;
      lastDiag = undefined;
    } finally {
      reloading = null;
      endLoadStage(INIT_STAGE); // a star now settling has named its own
    }
  });
}

/** Parameter changes that only touch the uniform block. */
function applyParams() {
  if (!sim) return;
  sim.cfg = cfg;
  sim.writeParams();
  finished = false;
  noteSettings();
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
        await sim.reset();
        armCheckpoints(cfg.restart?.time ?? 0);
        clearSources();
        beginSession(!cfg.restart, cfg.restart ? 'the run starts from a saved state' : undefined);
        lastDiag = undefined;
        if (finished) running = autoplay;
      } else noteSettings();
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
    if (eosHasEnergy(kind)) cfg.eosParams.gamma = Math.max(1.001, cfg.eosParams.gamma);
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
      // The tabulated equation of state needs its table on the device before its shaders are built.
      await prepareEosTable(gpu.device, kind, cfg.eosParams);
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

// --- the setup document -------------------------------------------------------------------
// Every setting of the simulation is a field of one document (setup/spec.ts). The run stands
// for one (setup/running.ts runningSpec); the Setup dialog edits a copy and hands it back,
// and the page does the least that gets the run there.

/** A setup of the library, by name. */
const getLibrarySetup = (name: string): SetupSpec | null => getSetup(name)?.spec ?? null;

/** The custom setup's document packed for a link, once it has been (packing takes a moment). */
let customLink: { version: number; text: string } | null = null;
function packCustom() {
  const c = customSetup();
  const v = customVersion();
  if (!c || customLink?.version === v) return;
  customLink = { version: v, text: '' };
  encodeSetupParam(c.spec)
    .then((text) => {
      if (customVersion() !== v) return;
      // Too long for a link: the address then names no setup, and Copy link in the dialog says so.
      customLink = { version: v, text: text.length <= MAX_LINK ? text : '' };
      lastQuery = '';
    })
    .catch(() => {});
}

/**
 * The id a document runs under: its preset's if it is that preset (on any grid), else the
 * custom setup's, which it becomes.
 */
function adopt(spec: SetupSpec): string {
  const preset = spec.id && spec.id !== CUSTOM_ID ? presetSpec(spec.id) : undefined;
  if (preset && sameSetup(spec, preset)) return spec.id!;
  // A preset since changed says so in its name.
  if (preset && spec.name === preset.name) spec = { ...spec, name: `${spec.name} (modified)` };
  setCustomSetup(spec);
  packCustom();
  return CUSTOM_ID;
}

/** Open a document as a new run, on its own grid (the present one where it names none). */
function openSpec(spec: SetupSpec): Promise<void> {
  const id = adopt(spec);
  const d = spec.domain ?? {};
  const g = cfg ? { n: cfg.n, ndim: cfg.ndim, levels: cfg.amrLevels } : { n: DEFAULT_SETUP.n, ndim: DEFAULT_SETUP.ndim, levels: DEFAULT_SETUP.amr };
  return build(id, d.n ?? g.n, d.ndim ?? g.ndim, d.amr ?? g.levels);
}

/** Resolves once every queued rebuild has finished. */
async function idle() {
  let q: Promise<void>;
  do {
    q = queue;
    await q;
  } while (q !== queue);
}

/**
 * Settings the run holds directly, changed mid-run: written into the configuration, then a
 * uniform write, or new shaders where one is compiled in, as each one needs.
 */
function applySettings(list: { setting: Setting; value: unknown }[]) {
  if (!list.length) return;
  let eos: EosKind | undefined;
  let shaders = false;
  let gravity = false;
  let again = false;
  let rebuilt = false;
  for (const { setting, value } of list) {
    const key = setting.cfg;
    if (key === 'eos') {
      eos = value as EosKind;
      continue;
    }
    // The diffusive fluxes are compiled in only while they are in use: switching one on or
    // off rebuilds the passes, a new strength is only a parameter.
    const diffusion = diffusionKey(cfg);
    const periodic = [0, 1, 2].map((a) => cfg.bcLo[a] === 0);
    setSetting(cfg, setting, value);
    if (key === 'eosParams.gamma') {
      // An energy equation needs gamma above 1. Where the species set the equation of state,
      // gamma is part of each one's heat capacity, which the shaders are built with.
      if (eosHasEnergy(cfg.eos)) cfg.eosParams.gamma = Math.max(1.001, cfg.eosParams.gamma);
      if (speciesNuclides(cfg.problem.species)) shaders = true;
    }
    // The merge threshold stays below the refinement one.
    if (key === 'refineHi' || key === 'refineLo') cfg.refineLo = Math.min(cfg.refineLo, cfg.refineHi / 2);
    if (diffusionKey(cfg) !== diffusion) shaders = true;
    // On an adaptive mesh a newly periodic axis joins blocks that were never neighbors and
    // may differ by several levels: build the mesh again from the start.
    if (key === 'bc' && cfg.amrLevels > 0 && [0, 1, 2].some((a) => cfg.bcLo[a] === 0 && !periodic[a])) again = true;
    if (key === 'gravity' || key === 'G') gravity = true;
    if (setting.option.apply === 'shaders') shaders = true;
    // What the simulator is built with (radiation's passes, the pressure solver, mirrors, the
    // driving's modes): a parameter write would leave the run as it was with the setting
    // showing as changed. (Radiation switched on from the Setup dialog ran on without it.)
    if (setting.option.apply === 'restart') rebuilt = true;
    // The equation of state's own radiation term goes with the switch (as a link's does).
    if (key === 'radiation') Object.assign(cfg, syncRadiationEos(cfg));
  }
  if (rebuilt) {
    if (eos && eos !== cfg.eos) {
      cfg.eos = eos;
      if (eosHasEnergy(eos)) cfg.eosParams.gamma = Math.max(1.001, cfg.eosParams.gamma);
    }
    // The same setup with the settings as they now are, from its start; the view stays.
    const mode = view.mode;
    build(cfg.problem.id, cfg.n, cfg.ndim, cfg.amrLevels, true).then(() => {
      view.mode = mode;
      if (cfg) syncControls();
    });
    return;
  }
  if (eos && eos !== cfg.eos) switchEos(eos);
  else if (shaders) rebuildPipelines();
  else applyParams();
  if (gravity) queue.then(() => sim?.refreshPotential());
  if (again) restartRun();
}

/**
 * A parameter of the run's setup (a featured option of its parts) takes new values: the run
 * carries on where the values only size the point masses or set the physical scale, is built
 * again for a new box, and otherwise starts again from the new initial conditions.
 */
function setParams(changed: Record<string, number>) {
  const defs = (cfg.problem.params ?? []).filter((q) => q.id in changed);
  if (!defs.length) return;
  for (const q of defs) cfg.params[q.id] = Math.min(q.max, Math.max(q.min, changed[q.id]));
  // Units that follow the parameters change the physical constants an equation of state other
  // than a barotropic one takes from them: refresh those (the setup's own values still win),
  // and the run must start again.
  const physical = typeof cfg.problem.units === 'function' && eosHasEnergy(cfg.eos);
  if (physical) unitConstants(cfg);
  if (defs.some((q) => q.id === cfg.problem.boxParam)) {
    // The side of the box: the grid's cells change size, so the run is built again.
    const { n, ndim, amrLevels } = cfg;
    if (autoload) build(cfg.problem.id, n, ndim, amrLevels, true);
    else {
      pending = { pick: { id: cfg.problem.id, n, ndim, levels: amrLevels }, keepPhysics: true };
      running = false;
    }
  } else if (defs.every((q) => q.scale) && !physical) {
    // Only the physical units: the run is the same in code units.
  } else if (defs.every((q) => q.live || (q.scale && !physical)) && sim) {
    // Only the point masses' sizes: the run carries on.
    sim.cfg = cfg;
    sim.retuneMasses();
  } else {
    // Part of the initial conditions (an orbit, the gas's starting state): start again.
    restartOrWait(true);
  }
}

/**
 * If an edited document differs from the run's in its parts only by values of the setup's own
 * parameters, take those as parameters (the run stays the setup it is, and its link stays
 * short) and say so.
 */
function applyAsParams(edited: SetupSpec): boolean {
  const base = cfg.problem.setup;
  if (!base) return false;
  const changed: Record<string, number> = {};
  for (const f of paramFeatures(base)) {
    const v = getPath(edited, featuredPaths(f)[0]);
    const x = typeof v === 'boolean' ? (v ? 1 : 0) : v;
    if (typeof x === 'number' && x !== cfg.params[featuredId(f)]) changed[featuredId(f)] = x;
  }
  if (!Object.keys(changed).length) return false;
  if (changeBetween(runningSpec({ ...cfg, params: { ...cfg.params, ...changed } }), edited).structure) return false;
  setParams(changed);
  return true;
}

/**
 * Take an edited document: nothing, a uniform write, new shaders or a fresh start, whichever is
 * the least that gets the run there (setup/running.ts changeBetween). Resolves once it has.
 */
async function applySpec(edited: SetupSpec): Promise<void> {
  if (!cfg) return;
  let change = changeBetween(runningSpec(cfg), edited);
  if (change.cost === 'none') return;
  if (change.structure && applyAsParams(edited)) {
    // Only the setup's own parameters differed in its parts: the run is still that setup.
    await idle();
    change = changeBetween(runningSpec(cfg), edited);
  }
  if (change.structure) {
    // The parts, the box, the units or what the gas is made of: a new run from the document.
    await openSpec(edited);
  } else {
    if (change.grid) {
      // The same setup on another grid keeps the physics chosen (in the same dimensions).
      const d = edited.domain ?? {};
      await build(cfg.problem.id, d.n ?? cfg.n, d.ndim ?? cfg.ndim, d.amr ?? cfg.amrLevels, true);
    }
    // The grid the run got may not be the one asked for (memory): settings are compared on it.
    const target = { ...edited, domain: { ...edited.domain, ndim: cfg.ndim, n: cfg.n, amr: cfg.amrLevels } };
    let now = changeBetween(runningSpec(cfg), target);
    if (now.described) {
      // Its name, description or featured options: the run goes on under the edited document.
      const p = problemOf(adopt(target));
      const fresh = configFor(p, cfg.n, { ndim: cfg.ndim, amrLevels: cfg.amrLevels });
      cfg = { ...cfg, problem: sizedProblem(p, fresh.params), params: fresh.params };
      if (sim) sim.cfg = cfg;
      // A point mass's softening or sink: taken up where the masses are now.
      if (now.retune) sim?.retuneMasses();
      now = changeBetween(runningSpec(cfg), target);
    }
    applySettings(now.settings);
  }
  await idle();
  noteSettings();
  syncControls();
}

/** A link that opens the page with a document: its preset and what differs, or the document whole. */
async function linkFor(spec: SetupSpec): Promise<{ url: string; tooLong: boolean }> {
  const d = spec.domain ?? {};
  const [n, ndim, amrLevels] = [d.n ?? cfg.n, d.ndim ?? cfg.ndim, d.amr ?? cfg.amrLevels];
  const base = spec.id && spec.id !== CUSTOM_ID ? presetSpec(spec.id) : undefined;
  const fromPreset = base ? changeBetween({ ...base, domain: { ...base.domain, ndim, n, amr: amrLevels } }, spec) : null;
  let c: SimConfig;
  let setup: string | undefined;
  if (fromPreset && !fromPreset.structure) {
    c = configFor(problemOf(spec.id!), n, { ndim, amrLevels });
    for (const { setting, value } of fromPreset.settings) setSetting(c, setting, value);
  } else {
    c = configFor(compiledFrom({ ...spec, id: CUSTOM_ID }).problem, n, { ndim, amrLevels });
    setup = await encodeSetupParam(spec);
  }
  const q = encodeLink({ cfg: c, view: viewDefaults(c.problem, ndim), auto: true, colormap: 'inferno', opacity: defaultOpacity(), setup });
  const url = `${location.origin}${location.pathname}?${q}`;
  return { url, tooLong: url.length > MAX_LINK };
}

/**
 * With Autoload off, a preset picked and not loaded yet (or the run's own preset picked again,
 * fresh): the configuration it will load with, as build() makes it. The Setup dialog shows this,
 * not the run's.
 */
function waitingConfig(): SimConfig | null {
  const pick = pending?.pick;
  if (!pick || !cfg || (pick.id === cfg.problem.id && pending!.keepPhysics)) return null;
  const c = freshConfig(problemOf(pick.id), pick, pick.params);
  for (const s of pick.settings ?? []) setSetting(c, s.setting, s.value);
  return c;
}

/** A preset's configuration on a grid, with parameters set over its defaults (as build() has them). */
function freshConfig(problem: Problem, g: { n: number; ndim: 1 | 2 | 3; levels: number }, params?: Record<string, number>): SimConfig {
  const c = configFor(problem, g.n, { ndim: g.ndim, amrLevels: g.levels });
  if (params) {
    Object.assign(c.params, params);
    if (typeof problem.units === 'function') unitConstants(c);
  }
  return c;
}

/**
 * An edited document for a preset that waits for Run: it keeps waiting, as that preset with new
 * parameters and settings where only those differ, else as the custom setup it has become.
 */
function waitWith(edited: SetupSpec) {
  const pick = pending!.pick!;
  const d = edited.domain ?? {};
  const g = { n: d.n ?? pick.n, ndim: d.ndim ?? pick.ndim, levels: d.amr ?? pick.levels };
  const problem = problemOf(pick.id);
  const params: Record<string, number> = { ...pick.params };
  for (const f of problem.setup ? paramFeatures(problem.setup) : []) {
    const v = getPath(edited, featuredPaths(f)[0]);
    const x = typeof v === 'boolean' ? (v ? 1 : 0) : v;
    if (typeof x === 'number') params[featuredId(f)] = x;
  }
  // Measured from the preset's defaults on the new grid, the settings that differ are all of them.
  let change: ReturnType<typeof changeBetween> | null = null;
  try {
    const base = runningSpec(freshConfig(problem, g, params));
    change = changeBetween(base, { ...edited, domain: { ...edited.domain, ...base.domain } });
  } catch {}
  const same = !!change && !change.structure && !change.described;
  const id = same ? pick.id : adopt(edited);
  pick.params = same ? params : undefined;
  pick.settings = same && change!.settings.length ? change!.settings : undefined;
  load(id, g.n, g.ndim, g.levels);
}

/** What the Setup dialog asks of the page (ui/setup/host.ts). */
const setupHost: SetupHost = {
  running: () => runningSpec(waitingConfig() ?? cfg),
  async apply(spec) {
    // A preset that waits for Run (Autoload off) takes the edit and goes on waiting.
    if (waitingConfig()) waitWith(spec);
    else await applySpec(spec);
  },
  presets: () =>
    MENU.map((g) => ({
      group: g.group,
      entries: g.families.flatMap((f) => f.variants.map((v) => ({ id: v.problem.id, label: f.variants.length > 1 ? `${f.label}: ${v.label}` : f.label }))),
    })),
  presetSpec: (id) => presetSpec(id)!,
  // A downloaded preset is whole: a variant's file with its parent's content in it.
  presetFile: (id) => formatSetup(presetSpec(id)!),
  memory(spec) {
    const d = spec.domain ?? {};
    const [n, ndim, levels] = [d.n ?? cfg.n, d.ndim ?? cfg.ndim, d.amr ?? cfg.amrLevels];
    return { need: gridBytes(n, ndim, levels), budget: memoryBudget() };
  },
  refusal(spec) {
    try {
      const d = spec.domain ?? {};
      const [n, ndim, levels] = [d.n ?? cfg.n, d.ndim ?? cfg.ndim, d.amr ?? cfg.amrLevels];
      const problem = compiledFrom(spec).problem;
      // As build() loads it: a run that cannot refine takes the uniform grid.
      return refusal(configFor(problem, n, { ndim, amrLevels: supportsAmr(problem) ? levels : 0 }));
    } catch {
      return null;
    }
  },
  grids: () => ({
    res1d: RES_1D, res2d: RES_2D, res3d: RES_3D, uniformMax1d: uniformMax(1), uniformMax2d: uniformMax(2), uniformMax3d: uniformMax(3), maxLevels: maxAmrLevels,
    fits: (n: number, ndim: 1 | 2 | 3, levels: number) => (levels > 0 ? amrFits(n, ndim, levels) : n <= uniformMax(ndim)),
  }),
  link: linkFor,
  units(spec) {
    try {
      const u = compiledFrom(spec).problem.units;
      return typeof u === 'function' ? u({}) : u;
    } catch {
      return undefined;
    }
  },
  physical: () => units().physical && unitSys !== 'code',
};

/** A setup file dropped on the page: opened in the dialog, with what reading it left out. */
async function openSetupFile(file: File) {
  const { spec, report } = readSetup(await file.text(), (id) => PRESET_FILES[id]);
  if (!spec) return void alert(`Could not open ${file.name}: ${report.error}`);
  openSetupDialog(setupHost, { file: spec, notes: report.notes });
}
window.addEventListener('dragover', (e) => {
  if ([...(e.dataTransfer?.items ?? [])].some((i) => i.kind === 'file')) e.preventDefault();
});
window.addEventListener('drop', (e) => {
  const file = [...(e.dataTransfer?.files ?? [])].find((f) => /\.json$/i.test(f.name));
  if (!file) return;
  e.preventDefault();
  void openSetupFile(file);
});

function bind() {
  const on = (id: string, fn: (el: HTMLInputElement & HTMLSelectElement) => void, ev = 'change') =>
    $(id).addEventListener(ev, () => fn($(id) as HTMLInputElement & HTMLSelectElement));
  const numOr = (el: HTMLInputElement, fallback: number) => (el.value === '' || Number.isNaN(+el.value) ? fallback : +el.value);

  // The grid is the run's own, or the one that waits for Run: the Setup dialog changes it.
  const picked = () => {
    const g = pending?.pick ?? { id: cfg.problem.id, n: cfg.n, ndim: cfg.ndim, levels: cfg.amrLevels };
    return { id: $<HTMLSelectElement>('variant').value || g.id, n: g.n, ndim: g.ndim, levels: g.levels };
  };
  // A preset with a grid of its own (Problem.grid) opens on it; another keeps the grid chosen.
  const pickedPreset = () => {
    const p = picked();
    const g = problemOf(p.id).grid;
    return g ? { ...p, ndim: g.ndim ?? p.ndim, n: g.n ?? p.n, levels: g.amr ?? p.levels } : p;
  };
  on('problem', (el) => {
    // One of the user's own setups, from the library: opened as the custom setup.
    if (el.value.startsWith('lib:')) {
      const spec = getLibrarySetup(el.value.slice(4));
      if (spec) void openSpec(spec);
      else syncControls();
      return;
    }
    // A new problem opens on its first variant.
    showProblem(el.value === CUSTOM_ID ? CUSTOM_ID : familyById(el.value).variants[0].problem.id);
    const { id, n, ndim, levels } = pickedPreset();
    load(id, n, ndim, levels);
  });
  on('variant', () => {
    const { id, n, ndim, levels } = pickedPreset();
    load(id, n, ndim, levels);
  });
  $('setup').addEventListener('click', () => openSetupDialog(setupHost));
  on('blocks', (el) => (view.showBlocks = el.checked));
  on('contours', (el) => {
    view.contours = el.checked;
    syncControls();
  });
  on('fog', (el) => (view.fog = el.checked));
  on('exact', (el) => (view.exact = el.checked));
  on('box', (el) => (view.box = el.checked));
  on('turnframe', (el) => (view.turnWithFrame = el.checked));
  on('isoa', (el) => (view.surfaceOpacity = logAt(+el.value, SURFACE_RANGE)), 'input');
  // The lines follow the count as it is typed or stepped; leaving the field tidies what it shows.
  const contourCount = (el: HTMLInputElement) => Math.min(MAX_CONTOURS, Math.max(1, Math.round(numOr(el, view.nContours))));
  on('ncont', (el) => (view.nContours = contourCount(el)), 'input');
  on('ncont', (el) => (el.value = String((view.nContours = contourCount(el)))));
  on('flines', (el) => {
    view.fieldLines = el.checked;
    syncControls();
  });
  const lineCount = (el: HTMLInputElement) => Math.min(MAX_FIELD_LINES, Math.max(1, Math.round(numOr(el, view.fieldLineCount))));
  on('nflines', (el) => (view.fieldLineCount = lineCount(el)), 'input');
  on('nflines', (el) => (el.value = String((view.fieldLineCount = lineCount(el)))));
  $<HTMLInputElement>('autoplay').checked = autoplay;
  on('autoplay', (el) => {
    autoplay = el.checked;
    try {
      localStorage.setItem('shaderous.autoplay', autoplay ? 'on' : 'off');
    } catch {}
  });
  $<HTMLInputElement>('autoload').checked = autoload;
  on('autoload', (el) => {
    autoload = el.checked;
    try {
      localStorage.setItem('shaderous.autoload', autoload ? 'on' : 'off');
    } catch {}
    // Switched back on with a change waiting: it loads now.
    if (autoload) loadPending();
    showLoad();
  });
  const timingBox = $<HTMLInputElement>('gputiming');
  timingBox.disabled = !timingAvailable();
  if (timingBox.disabled) {
    timingBox.parentElement!.title = 'This GPU or browser does not offer timestamp queries';
    $('timing').textContent = 'Timestamps are not available here.';
  }
  on('gputiming', (el) => {
    setTiming(el.checked);
    lastTiming = 0;
    $('timing').textContent = el.checked ? 'Measuring…' : '';
  });
  // GPU memory: the automatic budget unless the user sets one (gpu/memory.ts). A new budget
  // starts the run again at its size.
  showMemory();
  const rebuildForMemory = () => {
    showMemory();
    const { id, n, ndim, levels } = picked();
    load(id, n, ndim, levels, true);
  };
  on('gpumem', (el) => {
    const gb = +el.value;
    const auto = autoBudget().bytes;
    // An empty field, or the automatic value typed back in, means automatic.
    setUserBudget(el.value === '' || !(gb > 0) || Math.abs(gb * GB - auto) < 0.005 * GB ? null : Math.max(0.25, gb) * GB);
    rebuildForMemory();
  });
  $('gpumem-auto').addEventListener('click', () => {
    if (userBudget() === null && !autoBudget().lowered) return;
    setUserBudget(null);
    forgetCrashes();
    rebuildForMemory();
  });
  // The warning over the simulation (showMemory) opens this panel.
  $('mem-warn').addEventListener('click', () => {
    const perf = $<HTMLDetailsElement>('perf');
    perf.open = true;
    perf.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    $('gpumem').focus({ preventScroll: true });
  });
  on('ckpt', (el) => void setCheckpoints(el.checked));
  on('ckdt', (el) => {
    ckpt.every = +el.value > 0 ? +el.value : defaultCheckpointDt(cfg.problem);
    if (ckpt.on) clockNow().then(armCheckpoints);
    else showCheckpoints();
  });
  on('tend', (el) => {
    cfg.tEnd = el.value === '' ? Infinity : +el.value;
    applyParams();
  });
  $('run').addEventListener('click', () => {
    // A change waiting for Run (Autoload off): load it, then play.
    if (pending) return loadPending(true);
    if (cfg?.viewOnly) return;
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
  // Load (Autoload off): what waits, paused.
  $('load').addEventListener('click', () => loadPending(false));
  bindRecord();
  $('step').addEventListener('click', () => {
    // A change waiting for Run: Step and Reset load it too, and leave it paused.
    if (pending) return loadPending(false);
    running = false;
    stepOnce = true;
    syncControls();
  });
  $('reset').addEventListener('click', () => {
    if (!sim) return;
    if (pending) return loadPending(false);
    restartRun(finished); // replay a run that had reached its end time
    syncControls();
  });

  $('save-log').addEventListener('click', async () => {
    const log = await session.snapshot();
    if (!log || !cfg) return;
    const t = log.trace.length ? log.trace[log.trace.length - 1].t : 0;
    downloadBlob(`shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}-session-t${+t.toPrecision(4)}.json`, new Blob([JSON.stringify(log)], { type: 'application/json' }));
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
  const viewPicker = $<HTMLInputElement>('view-file-input');
  $('view-file').addEventListener('click', () => viewPicker.click());
  viewPicker.addEventListener('change', () => {
    const file = viewPicker.files?.[0];
    viewPicker.value = '';
    if (file) viewState(file).catch((e) => alert(`Could not open ${file.name}: ${e instanceof Error ? e.message : String(e)}`));
  });

  $('share').addEventListener('click', async () => {
    syncAddress(true);
    const url = location.href;
    let copied = true;
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      copied = false; // the dialog shows the link to copy by hand
    }
    showShare(url, copied);
  });

  on('mode', (el) => {
    view.mode = el.value as ViewState['mode'];
    syncControls();
  });
  on('units', (el) => {
    unitSys = el.value as DisplaySystem;
    try {
      localStorage.setItem('shaderous.units', unitSys);
    } catch {
      // Not remembered; the choice still holds for this page.
    }
    syncControls();
    if (diagOf) showDiagUnits();
  });
  on('field', (el) => {
    // A species' mass fraction is the field and the species, as "19:2".
    const [f, sp] = el.value.split(':');
    view.field = +f;
    if (sp !== undefined) view.species = +sp;
    // A density gradient spans decades (shocks against smooth flow), so it reads best on a log
    // scale; refinement levels are a handful of integers.
    if (view.field === 13 || view.field === 18) view.log = true;
    if (view.field === 14 || view.field === 21) view.log = false;
    syncControls();
  });
  on('axis', (el) => (view.sliceAxis = +el.value));
  on('slicepos', (el) => {
    if (view.mode === 'volume' && cfg.ndim === 3) {
      view.minRho = +el.value;
      showDensityCut();
    } else view.slicePos = +el.value;
  }, 'input');
  on('log', (el) => (view.log = el.checked));
  on('opacity', (el) => {
    view.opacity = logAt(+el.value, OPACITY_RANGE);
    syncOpacity();
  }, 'input');
  on('cmap', (el) => {
    editor.colormap = el.value;
    editor.draw();
    pushTransfer();
  });
  const manual = () => {
    $<HTMLInputElement>('auto').checked = false;
    const k = units().scale(viewMeasure());
    view.lo = numOr($<HTMLInputElement>('lo'), view.lo * k) / k;
    view.hi = numOr($<HTMLInputElement>('hi'), view.hi * k) / k;
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
  // A slice or a projection (always, in a 2D run) is panned and zoomed in its plane instead of orbited.
  const flat = () => view.mode !== 'volume' || (cfg?.ndim ?? 3) < 3;
  const clampPan = () => {
    const h = 0.5 / view.zoom;
    view.cx = Math.min(1 - h, Math.max(h, view.cx));
    view.cy = Math.min(1 - h, Math.max(h, view.cy));
  };
  const zoomAt = (clientX: number, clientY: number, factor: number) => {
    const r = canvas.getBoundingClientRect();
    const sx = (clientX - r.left - r.width / 2 + sliceShift()) / sliceSpan();
    const sy = -(clientY - r.top - r.height / 2) / sliceSpan();
    const z = Math.min(maxZoom(), Math.max(1, view.zoom * factor));
    view.cx += sx * (1 / view.zoom - 1 / z);
    view.cy += sy * (1 / view.zoom - 1 / z);
    view.zoom = z;
    clampPan();
  };
  /**
   * Volume view: move the point the camera orbits so that what lies at its depth follows a drag
   * of (dx, dy) pixels. The point stays inside the box, so the box cannot be panned out of view.
   */
  const panBy = (dx: number, dy: number) => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const f = cameraFrame(view, w / h);
    const k = (2 * f.tan * view.dist) / h; // box lengths per pixel at the target
    const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
    view.tx = clamp01(view.tx + (-dx * f.right[0] + dy * f.up[0]) * k);
    view.ty = clamp01(view.ty + (-dx * f.right[1] + dy * f.up[1]) * k);
    view.tz = clamp01(view.tz + (-dx * f.right[2] + dy * f.up[2]) * k);
  };
  canvas.addEventListener('dblclick', () => {
    if (tool) {
      // With a tool in hand a double click is two edits, or the end of a polygon.
      if (draft?.mode === 'polygon' && draft.pts.length >= 3) commitDraft();
      return;
    }
    if (flat()) [view.zoom, view.cx, view.cy] = [1, 0.5, 0.5];
    else [view.tx, view.ty, view.tz] = [0.5, 0.5, 0.5];
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
    // The profile tool places a line; an end of the profile on show can be dragged (profile tool below).
    if (e.button === 0 && pointers.size === 0 && !stroke && !draft && profilePress(e)) return;
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
    if (tool && tool !== 'obstacle' && !readsOnly(tool) && sim && !building && e.button === 0 && pointers.size === 0 && !stroke) {
      beginStroke(e);
      return;
    }
    if (stroke && !stroke.released) {
      // A second finger: this is a pinch after all. Whatever has been painted stays.
      pointers.set(stroke.id, { x: stroke.x + canvas.getBoundingClientRect().left, y: stroke.y + canvas.getBoundingClientRect().top });
      endStroke(false);
    }
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    // A press of the query tool that is let go where it began is a click: it pins the readout.
    queryPress = tool === 'query' && e.button === 0 && pointers.size === 1 ? { x: e.clientX, y: e.clientY } : null;
  });
  const release = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    profileRelease(e);
    if (queryPress && tool === 'query' && e.type === 'pointerup' && Math.hypot(e.clientX - queryPress.x, e.clientY - queryPress.y) < 5) void pinAt(e.clientX, e.clientY);
    queryPress = null;
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
  // In the volume view it pans, tool or not.
  canvas.addEventListener('contextmenu', (e) => {
    if (tool || !flat()) e.preventDefault();
  });
  canvas.addEventListener('pointerleave', () => {
    profileLineHover(null);
    hideCellPop();
    clearContourHover();
    hover = null;
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
    {
      const r = canvas.getBoundingClientRect();
      hover = { x: e.clientX - r.left, y: e.clientY - r.top };
    }
    if (draft && (draft.id === e.pointerId || (draft.id === -1 && pointers.size === 0))) {
      seeMods(e);
      draft.cur = at(e);
      drawBrush();
      return;
    }
    if (profileMove(e, pointers.size)) return;
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
    if (pointers.size === 0) contourHover(e.clientX, e.clientY);
    else clearContourHover();
    const last = pointers.get(e.pointerId);
    if (!last) return;
    const before = pointers.size === 2 ? spread() : 0;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (flat()) {
      // Slice: the point under the pointer (or under the middle of a pinch) stays under it.
      const share = pointers.size === 2 ? 0.5 : 1;
      const k = share / (sliceSpan() * view.zoom);
      view.cx -= (e.clientX - last.x) * k;
      view.cy += (e.clientY - last.y) * k;
      if (pointers.size === 2 && before > 0) {
        const [a, b] = [...pointers.values()];
        zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, spread() / before);
      } else clampPan();
    } else if (pointers.size === 1) {
      // Pan with Shift, the middle button, or the right button (which orbits with a tool in hand).
      if (e.shiftKey || e.buttons & 4 || (!tool && e.buttons & 2)) panBy(e.clientX - last.x, e.clientY - last.y);
      else {
        view.yaw -= (e.clientX - last.x) * 0.006;
        view.pitch = Math.min(1.5, Math.max(-1.5, view.pitch + (e.clientY - last.y) * 0.006));
      }
    } else if (pointers.size === 2 && before > 0) {
      // Two fingers: the middle of the pinch carries the view with it.
      panBy((e.clientX - last.x) / 2, (e.clientY - last.y) / 2);
      view.dist = Math.min(MAX_DIST, Math.max(MIN_DIST, (view.dist * before) / Math.max(spread(), 1)));
    }
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      if (flat()) zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015));
      else view.dist = Math.min(MAX_DIST, Math.max(MIN_DIST, view.dist * Math.exp(e.deltaY * 0.001)));
    },
    { passive: false },
  );
}

// --- editing strokes ----------------------------------------------------------------------
type Vec3 = [number, number, number];
/** A slice or a projection (always, in a 2D run) rather than the volume view. */
const flatView = () => view.mode !== 'volume' || (cfg?.ndim ?? 3) < 3;
const phone = window.matchMedia('(max-width: 700px)');
/**
 * Where the slice sits at zoom 1: the screen pixels across it, and how far its middle sits left
 * of the canvas's middle. It is the largest centered square that fits with a small margin,
 * unless that reaches under the rail on the right (color bar and tools): then it moves flush
 * against the canvas's left edge, and shrinks if it still does not fit beside the rail. Phones
 * keep it centered: there the rail lies over the simulation anyway and can be put away.
 */
function sliceLayout(): { span: number; shift: number } {
  const w = canvas.clientWidth;
  const span = Math.max(1, Math.min(w, canvas.clientHeight) * 0.94);
  if (phone.matches) return { span, shift: 0 };
  // The rail's left edge, less the same gap as between the rail and the right edge.
  const room = $('rail').getBoundingClientRect().left - canvas.getBoundingClientRect().left - 12;
  if ((w + span) / 2 <= room) return { span, shift: 0 };
  const fit = Math.max(1, Math.min(span, room));
  return { span: fit, shift: (w - fit) / 2 };
}
/** Screen pixels across the whole slice at zoom 1. */
const sliceSpan = () => sliceLayout().span;
/** Screen pixels the slice's middle sits left of the canvas's middle. */
const sliceShift = () => sliceLayout().shift;

/**
 * The texels of the slice's map (`side` a side) a native screenshot takes: all of them, or those
 * of the part of the box in view (left of the rail), rounded out to whole texels. Rows count up
 * from the box's bottom edge, as the map's do.
 */
function nativeRegion(region: 'box' | 'view', side: number): { i0: number; i1: number; j0: number; j1: number } {
  const all = { i0: 0, i1: side, j0: 0, j1: side };
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (region === 'box' || !(w > 0 && h > 0)) return all;
  const { span, shift } = sliceLayout();
  const k = span * view.zoom;
  // The rail lies over the canvas's right edge (but on a phone, where it can be put away).
  const room = phone.matches ? w : $('rail').getBoundingClientRect().left - canvas.getBoundingClientRect().left;
  const right = room > 0 ? Math.min(w, room) : w;
  // As render/slice.wgsl maps a pixel to the slice: x from the left, y from the bottom.
  const x0 = (-w / 2 + shift) / k + view.cx;
  const x1 = (right - w / 2 + shift) / k + view.cx;
  const y0 = -h / (2 * k) + view.cy;
  const y1 = h / (2 * k) + view.cy;
  // Outward to whole texels; an edge within a millionth of a texel of one stays on it.
  const lo = (v: number) => Math.max(0, Math.min(side, Math.floor(v * side + 1e-6)));
  const hi = (v: number) => Math.max(0, Math.min(side, Math.ceil(v * side - 1e-6)));
  return { i0: lo(x0), i1: hi(x1), j0: lo(y0), j1: hi(y1) };
}

/**
 * A 1D run's native screenshot (plans/one-dimensional.md): its row of texels repeated along y
 * holds nothing a square picture's height would add, and at 8192 texels a square would be 8192
 * pixels tall. A strip an eighth as tall as it is wide (32 pixels at least), about the middle
 * of the region asked for.
 */
function nativeStrip(r: { i0: number; i1: number; j0: number; j1: number }): { i0: number; i1: number; j0: number; j1: number } {
  const tall = Math.max(32, Math.ceil((r.i1 - r.i0) / 8));
  if (r.j1 - r.j0 <= tall) return r;
  const j0 = Math.round((r.j0 + r.j1 - tall) / 2);
  return { ...r, j0, j1: j0 + tall };
}
/** Pixels a side, at most, of a curved 1D run's native screenshot: its half-disk of rings. */
const NATIVE_RINGS = 4096;

/** A native screenshot has drawn into the canvas and not yet given it back (draw(null)). */
let nativeShot = false;

/** The canvas shows a frame of another size than its own (a movie's, a native screenshot's): fitted, the pointer camera waiting. */
function setRecordingStyle(size: [number, number] | null) {
  canvas.style.objectFit = size ? 'contain' : '';
  canvas.style.pointerEvents = size ? 'none' : '';
}
/** Drag length, in pixels, at which the velocity tool reaches its top speed. */
const fullDrag = () => 0.25 * Math.min(canvas.clientWidth, canvas.clientHeight);

/**
 * The point of the box under a point of the canvas, in fractions of the box, with the
 * directions that screen-right and screen-up correspond to there and the pixels per box length.
 * In the volume view the point lies on the plane facing the camera at the given distance.
 */
function locate(px: number, py: number, depth: number): { pos: Vec3; right: Vec3; up: Vec3; scale: number } {
  const l = locateInView(px, py, depth);
  // A spherical run's slice is the (R, z) half-plane: the point is the cell of its (r, theta),
  // outside the box off the grid; screen-right and screen-up as the cell fractions move there.
  if (flatView() && curvedView(cfg)) {
    const at = (x: number, y: number) => {
      const v = locateInView(x, y, depth).pos;
      return sphFromView(cfg, v[0], v[1]) ?? [-1, -1];
    };
    const p = at(px, py);
    const [dr, du] = [at(px + 1, py), at(px, py - 1)];
    const d = (q: number[]): Vec3 => (q[0] < 0 || p[0] < 0 ? [0, 0, 0] : [(q[0] - p[0]) * l.scale, (q[1] - p[1]) * l.scale, 0]);
    return { pos: [p[0], p[1], l.pos[2]], right: d(dr), up: d(du), scale: l.scale };
  }
  // With mirror images shown the view's box is the whole configuration: the point is folded
  // back into the computed part, whose fractions everything else takes, and lengths double.
  const mir = mirrorShown(cfg, view);
  if (!mir.some(Boolean)) return l;
  const k = (v: Vec3) => v.map((c, a) => (mir[a] ? 2 * c : c)) as Vec3;
  return { pos: l.pos.map((d, a) => mirrorComputed(d, mir[a])) as Vec3, right: k(l.right), up: k(l.up), scale: l.scale / 2 };
}

/** locate() in the view's own box (the whole configuration, when mirror images are shown). */
function locateInView(px: number, py: number, depth: number): { pos: Vec3; right: Vec3; up: Vec3; scale: number } {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (flatView()) {
    const a = view.cx + (px - w / 2 + sliceShift()) / (sliceSpan() * view.zoom);
    const b = view.cy - (py - h / 2) / (sliceSpan() * view.zoom);
    const axis = cfg.ndim < 3 ? 2 : view.sliceAxis;
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

// --- profile tool -------------------------------------------------------------------------
// (plans/profile-tool.md) Lines through the box and a plot of a field along them. With the
// profile tool in hand, a press places a line's start and a drag (or a second click) its end; in
// the volume view both lie on the plane facing the camera through the point orbited, or with Alt
// on the gas seen under the pointer (the end then on the plane through the start, Alt again
// putting it on the gas too). Each new line is added to those on show (up to MAX_PROFILES), in a
// colour of its own; they stay on show, whatever tool is in hand, with their plots overlaid in
// one panel until the panel is closed. The panel plots one field or several along every line:
// the first solid against the left axis, the others each in a pattern of its own (profileDash),
// the second against an axis on the right. The GPU samples each field along all the lines in one
// pass (Renderer.encodeProfile), frame by frame while the run goes.
/** A line on show: its ends in fractions of the computed box, its colour, and its samples as last read. */
interface PlacedProfile {
  a: Vec3;
  b: Vec3;
  /** the run's dimensions when it was placed: another kind of run drops it */
  dims: 1 | 2 | 3;
  /** index into PROFILE_COLORS */
  color: number;
  /** the samples last read along it, one a field (samplesOf picks the one of a field) */
  data: ProfileSamples[];
}
let profiles: PlacedProfile[] = [];
/** A field plotted along the lines, and its log axis (null: as the field's default). */
interface ProfileFieldChoice {
  field: number;
  species: number;
  log: boolean | null;
}
/** The fields plotted along every line, in the panel's order: the first against the left axis. Never empty. */
let profileFields: ProfileFieldChoice[] = [{ field: 0, species: 0, log: null }];
/** Lines set by a link before their run was built: the panel opens on the first frame of the run. */
let profileOpenPending = false;
/** The samples read along a line. */
interface ProfileSamples {
  a: Vec3;
  b: Vec3;
  field: number;
  species: number;
  projection: number | null;
  /** distance from the start, code length */
  s: Float64Array;
  length: number;
  values: Float32Array;
  levels: Float32Array;
  solid: Uint8Array;
}
/** The lines on show are to be sampled on the next frame (a line or the field changed). */
let profileWanted = false;
/** Profiles sampled, and the last one taken up: a read that comes back late is dropped. */
let profileSeq = 0;
let profileShown = 0;
/**
 * The points along the lines on show, one after another, as last laid out (profileLine()), what
 * they were laid out for, and where each line's lie among them.
 */
let profileLineCache: { key: string; points: Float32Array<ArrayBuffer>; lines: { a: Vec3; b: Vec3; s: Float64Array; at: number; n: number }[] } | null = null;
/** When the lines were last sampled (performance.now()). */
let profileAt = 0;
/** The distance hovered (code length), in the plot or on a line (`line`, -1 from the plot); null: none. */
let profileHover: { s: number; line: number } | null = null;
/** Each field's y range, eased while the run goes (`at` when last eased); started again when its field or scale changes (`key`). */
let profileRanges: ({ key: string; range: [number, number]; at: number } | null)[] = [];
/**
 * A line being placed: the press (CSS pixels), its start in the box once known (in the volume
 * view the gas under the press is read back first), how deep the start lies along the view, the
 * pointer, and the end on the gas under it (Alt, volume view). `id` is the pointer pressing, -1
 * once the press was a click (the end is placed by a second one), -2 once the end is placed but
 * the start is still being read. `t0`: when the press came (performance.now()).
 */
let profileDraft: { id: number; t0: number; x0: number; y0: number; a: Vec3 | null; depth: number; x: number; y: number; alt: Vec3 | null } | null = null;
/**
 * A press let go within this long (ms) is a click, however far the pointer slipped meanwhile:
 * the end is then placed by a second click, not where the press was let go (which drew a tiny
 * line when a click wobbled).
 */
const PROFILE_CLICK_MS = 400;
/** An end of a line being dragged, and how deep it lies along the view. */
let profileGrab: { id: number; line: number; end: 0 | 1; depth: number } | null = null;
/** One read of the gas under the pointer at a time (Alt in the volume view). */
let profilePicking = false;

/** How deep along the view direction the gas under a canvas point lies (volume view), or null. */
async function gasDepth(px: number, py: number): Promise<number | null> {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const [ux, uy] = [(2 * px) / w - 1, 1 - (2 * py) / h];
  const t = await renderer.pick(ux, uy);
  if (t === null) return null;
  const f = cameraFrame(view, w / h);
  return t / Math.hypot(1, ux * (w / h) * f.tan, uy * f.tan);
}

/** The point of the box under a canvas point at a depth (ignored in a plane), z 0 in 2D. */
function profilePoint(px: number, py: number, depth: number): Vec3 {
  const p = locate(px, py, depth).pos;
  if (cfg.ndim < 3) p[2] = 0;
  return p;
}

/**
 * A 1D run's profile lines go the one way that is not redundant (plans/one-dimensional.md): in a
 * Cartesian run, horizontal (the end moved, `moving`, takes the other's y); in a spherical or
 * axisymmetric one, radial, from the center out to the outer end at its polar angle. Both are
 * a line of constant y in the grid's fractions (y is theta / pi, or the angle around the disk).
 * Any other run's line is as it was drawn.
 */
function lockedLine(a: Vec3, b: Vec3, moving: 0 | 1 = 1, c: SimConfig = cfg): [Vec3, Vec3] {
  return lockLine(a, b, moving, c.ndim, radialProfiles(c));
}
/** A 1D spherical or axisymmetric run, whose profile lines are radial (lockedLine). */
const radialProfiles = (c: SimConfig | undefined) => !!c && c.ndim === 1 && (c.problem.geometry === 'spherical' || c.problem.geometry === 'cylindrical');

/** Shift: the pointer moved around the press to the nearest 15° step. */
function snapAngle(x0: number, y0: number, x: number, y: number): [number, number] {
  const r = Math.hypot(x - x0, y - y0);
  const step = Math.PI / 12;
  const a = Math.round(Math.atan2(y - y0, x - x0) / step) * step;
  return [x0 + r * Math.cos(a), y0 + r * Math.sin(a)];
}

/** Which end of which line on show lies within reach of a canvas point, if one does (the line drawn last first). */
function profileEndNear(px: number, py: number): { line: number; end: 0 | 1 } | null {
  if (!profiles.length || !cfg) return null;
  const at = projector();
  let best: { line: number; end: 0 | 1 } | null = null;
  let dist = 9;
  for (let i = profiles.length - 1; i >= 0; i--) {
    for (const end of [0, 1] as const) {
      const c = at(end ? profiles[i].b : profiles[i].a);
      const d = Math.hypot(c.x - px, c.y - py);
      if (c.w > 0 && d < dist) [best, dist] = [{ line: i, end }, d];
    }
  }
  return best;
}

/** A press on the canvas: grab an end of a line, or place a line with the profile tool. Whether it was taken. */
function profilePress(e: PointerEvent): boolean {
  if (!sim || building || !cfg) return false;
  const r = canvas.getBoundingClientRect();
  const [px, py] = [e.clientX - r.left, e.clientY - r.top];
  if (profileDraft && profileDraft.id === -1) {
    // The second click of a line placed by clicks.
    [profileDraft.x, profileDraft.y] = e.shiftKey && cfg.ndim > 1 ? snapAngle(profileDraft.x0, profileDraft.y0, px, py) : [px, py];
    finishDraft();
    return true;
  }
  if (tool !== 'profile' && tool !== null) return false;
  const near = profileEndNear(px, py);
  if (near) {
    const l = profiles[near.line];
    const p = near.end ? l.b : l.a;
    profileGrab = { id: e.pointerId, ...near, depth: flatView() ? view.dist : projector()(p).w };
    return true;
  }
  if (tool !== 'profile') return false;
  // In the volume view the line lies on the plane facing the camera through the point orbited;
  // with Alt the start is on the gas seen under the press (with none there, on that plane).
  const onGas = !flatView() && e.altKey;
  const draftNow = { id: e.pointerId, t0: performance.now(), x0: px, y0: py, a: onGas ? null : profilePoint(px, py, view.dist), depth: view.dist, x: px, y: py, alt: null };
  profileDraft = draftNow;
  if (onGas) {
    void gasDepth(px, py).then((d) => {
      if (profileDraft !== draftNow) return;
      draftNow.depth = d ?? view.dist;
      draftNow.a = profilePoint(px, py, draftNow.depth);
      if (draftNow.id === -2) finishDraft();
      else drawProfileLine();
    });
  }
  drawProfileLine();
  return true;
}

/** The pointer moved: a line being placed, or an end being dragged, follows it. Whether it was taken. */
function profileMove(e: PointerEvent, pressed: number): boolean {
  const r = canvas.getBoundingClientRect();
  const [px, py] = [e.clientX - r.left, e.clientY - r.top];
  const d = profileDraft;
  if (d && d.id !== -2 && (d.id === e.pointerId || (d.id === -1 && pressed === 0))) {
    [d.x, d.y] = e.shiftKey && cfg.ndim > 1 ? snapAngle(d.x0, d.y0, px, py) : [px, py];
    // Alt in the volume view: the end on the gas under the pointer, read one at a time.
    d.alt = null;
    if (e.altKey && !flatView() && !profilePicking) {
      profilePicking = true;
      const [x, y] = [d.x, d.y];
      void gasDepth(x, y)
        .then((depth) => {
          if (profileDraft === d && depth !== null && d.x === x && d.y === y) d.alt = profilePoint(x, y, depth);
          drawProfileLine();
        })
        .finally(() => (profilePicking = false));
    }
    drawProfileLine();
    return true;
  }
  const g = profileGrab;
  const l = g ? profiles[g.line] : undefined;
  if (g && l && g.id === e.pointerId) {
    const [x, y] = e.shiftKey && cfg.ndim > 1 ? snapFrom(l, g.end, px, py) : [px, py];
    const p = profilePoint(x, y, g.depth);
    const c = clipSegment(...lockedLine(g.end ? l.a : p, g.end ? p : l.b, g.end), cfg.ndim);
    if (c) {
      l.a = c[0];
      l.b = c[1];
      requestProfile();
    }
    drawProfileLine();
    return true;
  }
  if (pressed === 0) profileLineHover(e);
  return false;
}

/** Shift while an end is dragged: the line kept at 15° steps on screen, about its other end. */
function snapFrom(l: PlacedProfile, end: 0 | 1, px: number, py: number): [number, number] {
  const o = projector()(end ? l.a : l.b);
  return snapAngle(o.x, o.y, px, py);
}

function profileRelease(e: PointerEvent) {
  const d = profileDraft;
  if (d && d.id === e.pointerId) {
    if (e.type !== 'pointerup') profileDraft = null;
    // A press let go where it began, or soon after it came, is a click: the end is placed with
    // the next one. (A radial line needs one point, its outer end: a click places it.)
    else if (!radialProfiles(cfg) && (Math.hypot(d.x - d.x0, d.y - d.y0) < 5 || performance.now() - d.t0 < PROFILE_CLICK_MS)) d.id = -1;
    else finishDraft();
    drawProfileLine();
  }
  if (profileGrab && profileGrab.id === e.pointerId) {
    profileGrab = null;
    requestProfile();
  }
}

/** Add the line being drawn to those on show, once its start is known. */
function finishDraft() {
  const d = profileDraft;
  if (!d) return;
  if (!d.a) {
    // The gas under the press is still being read: placed when it is (profilePress).
    d.id = -2;
    return;
  }
  profileDraft = null;
  const b = d.alt ?? profilePoint(d.x, d.y, d.depth);
  const c = clipSegment(...lockedLine(d.a, b), cfg.ndim);
  if (!c) {
    notice('That line does not cross the box');
    drawProfileLine();
    return;
  }
  if (profiles.length >= MAX_PROFILES) {
    notice(`At most ${MAX_PROFILES} profile lines at once: remove one in the profile panel first`);
    drawProfileLine();
    return;
  }
  // The first line plots the field the view shows; later ones plot the panel's.
  if (!profiles.length) profileFields = [{ field: view.field, species: view.species, log: null }];
  addProfile(c[0], c[1]);
  openProfilePop(true);
}

/** A line added to those on show, in the first colour none of them has. */
function addProfile(a: Vec3, b: Vec3, data: ProfileSamples | null = null): PlacedProfile {
  const used = new Set(profiles.map((p) => p.color));
  const color = Math.max(0, PROFILE_COLORS.findIndex((_, i) => !used.has(i)));
  const p: PlacedProfile = { a, b, dims: cfg.ndim, color, data: data ? [data] : [] };
  profiles.push(p);
  profileRanges = [];
  profileHover = null;
  requestProfile();
  syncProfileLines();
  drawProfileLine();
  return p;
}

/** Remove one line; the last one closes the panel. */
function removeProfile(i: number) {
  if (profiles.length <= 1) return closeProfile();
  profiles.splice(i, 1);
  profileGrab = null;
  profileHover = null;
  profileRanges = [];
  requestProfile();
  syncProfileLines();
  drawProfilePop();
  drawProfileLine();
}

/** Close the profile: every line and the panel. */
function closeProfile() {
  profiles = [];
  profileDraft = null;
  profileGrab = null;
  profileHover = null;
  profileOpenPending = false;
  profileFields = [profileFields[0]];
  profilePop.hidden = true;
  drawProfileLine();
}

/**
 * The lines a link carries (link.ts), set before their run is built: drawn and sampled once it
 * is, the panel opened on its first frame. The field defaults to the one the view shows.
 */
function setLinkProfiles(lp: LinkProfiles, c: SimConfig) {
  // (A 1D run's lines made horizontal or radial, as it draws them: lockedLine.)
  profiles = lp.lines.slice(0, MAX_PROFILES).map(([a, b], i) => {
    const [la, lb] = lockedLine([...a] as Vec3, [...b] as Vec3, 1, c);
    return { a: la, b: lb, dims: c.ndim, color: i, data: [] };
  });
  profileFields = lp.fields?.length ? lp.fields.map((f) => ({ field: f.field, species: f.species ?? 0, log: f.log ?? null })) : [{ field: view.field, species: view.species, log: null }];
  profileRanges = [];
  profileHover = null;
  profileDraft = null;
  profileGrab = null;
  profilePop.hidden = true;
  profileOpenPending = profiles.length > 0;
  requestProfile();
}

/** The lines on show as a link carries them, or undefined with none. */
function linkProfiles(): LinkProfiles | undefined {
  if (!profiles.length) return undefined;
  return { lines: profiles.map((p) => [p.a, p.b] as [Vec3, Vec3]), fields: profileFields.map((f) => ({ ...f })) };
}

/** Whether the view's menu offers a field in this run (not disabled, nor hidden with its group). */
function fieldOffered(field: number): boolean {
  const opt = [...$<HTMLSelectElement>('field').options].find((q) => q.value === String(field) || q.value.startsWith(`${field}:`));
  return !opt || !(opt.disabled || opt.hidden || (opt.parentElement as HTMLOptGroupElement | null)?.hidden);
}

/** The pointer over the canvas, nothing pressed: on a line it marks the distance there in the plot. */
let profileHoverFromLine = false;
function profileLineHover(e: PointerEvent | null) {
  let hover: { s: number; line: number } | null = null;
  let px = 0;
  let py = 0;
  if (e) {
    const r = canvas.getBoundingClientRect();
    [px, py] = [e.clientX - r.left, e.clientY - r.top];
  }
  if (e && profiles.length && !profileDraft && !profileGrab) {
    const at = projector();
    let best = 6;
    profiles.forEach((p, i) => {
      const d = samplesOf(p, 0);
      if (!d) return;
      const [A, B] = [at(p.a), at(p.b)];
      const [dx, dy] = [B.x - A.x, B.y - A.y];
      const len2 = dx * dx + dy * dy;
      if (!(A.w > 0 && B.w > 0 && len2 > 1)) return;
      const t = Math.min(1, Math.max(0, ((px - A.x) * dx + (py - A.y) * dy) / len2));
      const off = Math.hypot(A.x + t * dx - px, A.y + t * dy - py);
      if (off < best) {
        best = off;
        hover = { s: d.s[Math.round(t * (d.s.length - 1))], line: i };
      }
    });
  }
  // An end that can be dragged shows a grab cursor.
  canvas.classList.toggle('on-profile-end', !!e && (tool === 'profile' || tool === null) && profileEndNear(px, py) !== null);
  if (!hover && !profileHoverFromLine) return;
  profileHoverFromLine = !!hover;
  const h = hover as { s: number; line: number } | null;
  if (h?.s !== profileHover?.s || h?.line !== profileHover?.line) {
    profileHover = h;
    drawProfilePop();
    drawProfileLine();
  }
}

/** The lines on the canvas: the one being placed, and those on show with their ends and tenths. */
let profileLineDrawn = '';
function drawProfileLine() {
  let html = '';
  if (cfg && (profileDraft || profiles.length)) {
    const at = projector();
    const seg = (A: { x: number; y: number }, B: { x: number; y: number }) => `<line class="trace" x1="${A.x}" y1="${A.y}" x2="${B.x}" y2="${B.y}" />`;
    const several = profiles.length > 1;
    for (const [i, p] of profiles.entries()) {
      const [A, B] = [at(p.a), at(p.b)];
      if (!(A.w > 0 && B.w > 0)) continue;
      const [dx, dy] = [B.x - A.x, B.y - A.y];
      const len = Math.hypot(dx, dy) || 1;
      const [nx, ny] = [-dy / len, dx / len];
      let ticks = '';
      for (let k = 1; k < 10; k++) {
        // Each tenth of the line, as the box holds it: foreshortened as the view sees it.
        const c = at(p.a.map((v, j) => v + (k / 10) * (p.b[j] - v)));
        const h = k === 5 ? 5 : 3;
        ticks += `M${c.x - h * nx} ${c.y - h * ny}L${c.x + h * nx} ${c.y + h * ny}`;
      }
      let g = seg(A, B) + `<path class="ticks" d="${ticks}" />`;
      const d = samplesOf(p, 0);
      if (profileHover && d && d.length > 0 && profileHover.s <= d.length) {
        const t = profileHover.s / d.length;
        const c = at(p.a.map((v, j) => v + t * (p.b[j] - v)));
        g += `<circle class="mark" cx="${c.x}" cy="${c.y}" r="4" />`;
      }
      g += `<circle class="end start" cx="${A.x}" cy="${A.y}" r="5" /><circle class="end" cx="${B.x}" cy="${B.y}" r="5" />`;
      // Several lines: each numbered at its start, as the panel numbers them.
      if (several) g += `<text class="tag" x="${A.x - 9 * dx / len}" y="${A.y - 9 * dy / len}">${i + 1}</text>`;
      html += `<g style="--c:${PROFILE_COLORS[p.color]}">${g}</g>`;
    }
    const d = profileDraft;
    if (d) {
      let A = d.a ? at(d.a) : { x: d.x0, y: d.y0, w: 1 };
      let B = d.alt ? at(d.alt) : { x: d.x, y: d.y, w: 1 };
      // A 1D run's line as it will be placed: horizontal, or radial (lockedLine).
      if (d.a && cfg.ndim === 1) {
        const [la, lb] = lockedLine(d.a, d.alt ?? profilePoint(d.x, d.y, d.depth));
        [A, B] = [at(la), at(lb)];
      }
      const used = new Set(profiles.map((p) => p.color));
      const c = PROFILE_COLORS[Math.max(0, PROFILE_COLORS.findIndex((_, i) => !used.has(i)))];
      if (A.w > 0) html += `<g style="--c:${c}">${seg(A, B)}<circle class="end start" cx="${A.x}" cy="${A.y}" r="4" /><circle class="end" cx="${B.x}" cy="${B.y}" r="4" /></g>`;
    }
  }
  if (html !== profileLineDrawn) $('profile-line').innerHTML = profileLineDrawn = html;
}

/** Ask for the lines on show to be sampled on the next frame. */
function requestProfile() {
  profileWanted = true;
}

/** The points along a line, and their distances from its start in code length: `n`, or two a finest cell. */
function profileLine(a: Vec3, b: Vec3, n?: number): { points: Float32Array<ArrayBuffer>; s: Float64Array } {
  const dims = gridDims(cfg);
  const cells = Math.hypot(...[0, 1, 2].map((k) => (b[k] - a[k]) * dims[k]));
  const count = sampleCount(cells, n);
  const pts = samplePoints(a, b, count);
  const points = new Float32Array(4 * count);
  pts.forEach((q, i) => points.set([q[0], q[1], q[2], 0], 4 * i));
  const axes = cfg.ndim === 3 ? [0, 1, 2] : [0, 1];
  const [pa, pb] = [a, b].map((f) => axes.map((k) => fracPos(cfg, f[k], k)));
  return { points, s: distances(pa, pb, count) };
}

/**
 * Each frame: sample the lines on show in the frame's own commands, after its steps, all in one
 * pass (their points one after another), so the plot follows the run frame by frame (as fast as
 * frames are drawn, up to the display's rate). While the run stands still they are sampled when a
 * line or the field changes, and twice a second in case the state changed some other way.
 * Returns what reads them back once `enc` is submitted.
 */
function encodeFrameProfile(now: number, enc: GPUCommandEncoder, s: Simulation): (() => void) | null {
  if (!profiles.length || building || s !== sim) return null;
  if (profilePop.hidden && !profileOpenPending) return null;
  if (profiles.some((p) => p.dims !== cfg.ndim) || (spherical(cfg) && cfg.ndim > 1)) {
    closeProfile();
    return null;
  }
  if (profileOpenPending) {
    // Lines from a link: a field this run does not offer gives way to the one shown.
    profileOpenPending = false;
    const kept = profileFields.filter((f) => fieldOffered(f.field) && !(cfg.viewBaked && f.field !== view.field));
    profileFields = kept.length ? (cfg.viewBaked ? kept.slice(0, 1) : kept) : [{ field: view.field, species: view.species, log: null }];
    syncProfileLines();
    openProfilePop(true);
    drawProfileLine();
  }
  if (!running && !profileWanted && now - profileAt < 500) return null;
  const key = `${profiles.map((p) => `${p.a.join()}|${p.b.join()}`).join(';')}|${cfg.ndim}|${cfg.n}|${gridDims(cfg).join()}`;
  if (profileLineCache?.key !== key) {
    const laid = profiles.map((p) => ({ a: [...p.a] as Vec3, b: [...p.b] as Vec3, ...profileLine(p.a, p.b) }));
    const points = new Float32Array(laid.reduce((t, l) => t + l.points.length, 0));
    let at = 0;
    const lines = laid.map((l) => {
      points.set(l.points, 4 * at);
      const n = l.s.length;
      const out = { a: l.a, b: l.b, s: l.s, at, n };
      at += n;
      return out;
    });
    profileLineCache = { key, points, lines };
  }
  const cache = profileLineCache;
  // One pass a field, each with a slot of its own for its settings (they all reach the queue
  // before any pass runs). The first must go; a later one with no readback free is left for the
  // next frame.
  const projection = projecting() ? view.sliceAxis : null;
  const opts = profileFields.map((f) => ({ field: f.field, species: f.species, projection }));
  const reads: ((() => Promise<ProfileValues>) | null)[] = [];
  for (const [k, o] of opts.entries()) {
    let read: (() => Promise<ProfileValues>) | null;
    try {
      read = renderer.encodeProfile(enc, s, cache.points, key, o, false, k);
    } catch {
      if (k === 0) return null; // the view is not attached to this run yet
      read = null;
    }
    if (!read && k === 0) return null; // compiling, or every readback busy: the next frame tries again
    reads.push(read);
  }
  profileWanted = reads.some((r) => !r);
  profileAt = now;
  const seq = ++profileSeq;
  const owners = profiles.slice();
  return () => {
    Promise.all(reads.map((r) => (r ? r() : null)))
      .then((got) => {
        if (seq < profileShown || s !== sim) return;
        profileShown = seq;
        owners.forEach((p, i) => {
          // (A line removed meanwhile is not drawn again.)
          if (!profiles.includes(p)) return;
          const l = cache.lines[i];
          const cut = <T extends Float32Array | Uint8Array>(v: T) => v.subarray(l.at, l.at + l.n) as T;
          // A field not read this time keeps what it had.
          const fresh: ProfileSamples[] = [];
          got.forEach((q, k) => {
            if (q) fresh.push({ a: l.a, b: l.b, ...opts[k], s: l.s, length: l.s[l.n - 1], values: cut(q.values), levels: cut(q.levels), solid: cut(q.solid) });
          });
          p.data = [...fresh, ...p.data.filter((d) => !fresh.some((f) => f.field === d.field && f.species === d.species))];
        });
        // (The view's menu may have changed with the run: a new one loaded with the panel open.)
        syncProfileMenu();
        drawProfilePop();
        drawProfileLine();
      })
      .catch(() => {});
  };
}

/** Sample a field along the segment from a to b (fractions of the computed box): `n` samples, or two a finest cell. */
async function profileOf(s: Simulation, a: Vec3, b: Vec3, o: { field: number; species: number; projection: number | null }, n?: number): Promise<ProfileSamples> {
  const line = profileLine(a, b, n);
  const count = line.s.length;
  const pts = Array.from({ length: count }, (_, i) => line.points.subarray(4 * i, 4 * i + 3));
  const r = await renderer.profile(s, pts, o);
  return { a: [...a] as Vec3, b: [...b] as Vec3, ...o, s: line.s, length: line.s[count - 1], ...r };
}

/** The name of a field (a mass fraction by its species), as projected when it is. */
function profileFieldName(field: number, species: number, projected: boolean): string {
  if (field === MASS_FRACTION_FIELD) return `X(${speciesLabel(cfg.problem.species?.[species] ?? '?')})`;
  return projected && field === 0 ? 'Column density' : FIELDS[field];
}
/** What a profiled field measures: density projected is a column density. */
const profileMeasure = (field: number, projected: boolean): Measure => (projected && field === 0 ? [1, -2, 0] : fieldMeasure(field, cfg.eosParams.gamma));
/** A field's value axis is logarithmic: as chosen, or as the view shows the field (fields of either sign never). */
function profileLog(p: { field: number; log: boolean | null } = profileFields[0]): boolean {
  if (LINEAR_FIELDS.has(p.field)) return false;
  if (p.log !== null) return p.log;
  return p.field === view.field ? view.log : true;
}
/** A line's samples of the k-th field plotted, as last read, if there are any. */
function samplesOf(p: PlacedProfile, k: number): ProfileSamples | null {
  const f = profileFields[k];
  return (f && p.data.find((d) => d.field === f.field && d.species === f.species)) || null;
}
/** Each line's samples of the first field. */
const profileSamples = (p: PlacedProfile) => samplesOf(p, 0);
/** A field's value in the panel's menus: its index, a mass fraction's with its species. */
const fieldValue = (f: { field: number; species: number }) => (f.field === MASS_FRACTION_FIELD ? `${MASS_FRACTION_FIELD}:${f.species}` : String(f.field));

// The panel: the field, the scale, the plot, and the lines' ends.
const profilePop = $('profile-pop');
const profileCanvas = profilePop.querySelector('canvas')!;
const profileRows = $('profile-fields');
/** The panel has been moved by hand: it stays where it was put. */
let profilePopMoved = false;

/** Open the panel: beside the newest line's end when it is new (or the panel was not moved). */
function openProfilePop(place: boolean) {
  const was = profilePop.hidden;
  profilePop.hidden = false;
  syncProfileMenu();
  syncProfileLines();
  const last = profiles[profiles.length - 1];
  if ((was || place) && !profilePopMoved && !phone.matches && last) {
    const at = projector();
    const r = canvas.getBoundingClientRect();
    const [A, B] = [at(last.a), at(last.b)];
    const pw = profilePop.offsetWidth;
    const ph = profilePop.offsetHeight;
    // In the view area (left of the rail), beside the line: right of it, left of it, below or
    // above it, the first that fits; else in the view area's lower left corner.
    const [x0, x1] = [r.left + 8, $('rail').getBoundingClientRect().left - 8];
    const [y0, y1] = [r.top + 8, Math.min(r.bottom, window.innerHeight) - 8];
    const [lx, hx] = [Math.min(A.x, B.x) + r.left, Math.max(A.x, B.x) + r.left];
    const [ly, hy] = [Math.min(A.y, B.y) + r.top, Math.max(A.y, B.y) + r.top];
    const midY = Math.min(y1 - ph, Math.max(y0, B.y + r.top - ph / 2));
    const midX = Math.min(x1 - pw, Math.max(x0, B.x + r.left - pw / 2));
    const spots: [number, number][] = [[hx + 24, midY], [lx - 24 - pw, midY], [midX, hy + 24], [midX, ly - 24 - ph]];
    const [x, y] = spots.find(([sx, sy]) => sx >= x0 && sx + pw <= x1 && sy >= y0 && sy + ph <= y1) ?? [x0, y1 - ph];
    profilePop.style.left = `${Math.round(Math.max(4, Math.min(window.innerWidth - pw - 4, x)))}px`;
    profilePop.style.top = `${Math.round(Math.max(4, Math.min(window.innerHeight - ph - 4, y)))}px`;
  }
  drawProfilePop();
}

/**
 * The panel's fields, a row each: its pattern, its menu (the view's, with what this run cannot
 * show left out, and the fields of the other rows), its log switch and a button to remove it.
 * The rows are made again when their number or the view's menu changes, and set every time.
 */
let profileRowsFor = '';
function syncProfileMenu() {
  if (!profiles.length) return;
  const main = $<HTMLSelectElement>('field');
  const n = profileFields.length;
  const rowsKey = `${n}|${main.innerHTML}`;
  if (profileRowsFor !== rowsKey) {
    profileRowsFor = rowsKey;
    profileRows.innerHTML = profileFields
      .map((_, k) => {
        const dash = profileDash(k).join(' ');
        return `<div class="frow" data-k="${k}"><svg class="dash" viewBox="0 0 22 8" aria-hidden="true"><line x1="1" y1="4" x2="21" y2="4"${dash ? ` stroke-dasharray="${dash}"` : ''} /></svg><select data-k="${k}" aria-label="Field ${k + 1}">${main.innerHTML}</select><label title="A logarithmic axis for this field"><input type="checkbox" data-log="${k}" /> log</label><button type="button" data-rm="${k}" title="Stop plotting this field" aria-label="Remove field ${k + 1}">&times;</button></div>`;
      })
      .join('');
    // A row more makes the panel taller: kept on screen.
    const r = profilePop.getBoundingClientRect();
    if (!profilePop.hidden && !phone.matches && r.bottom > window.innerHeight - 4) profilePop.style.top = `${Math.round(Math.max(4, window.innerHeight - 4 - r.height))}px`;
  }
  const used = profileFields.map(fieldValue);
  profileFields.forEach((f, k) => {
    const row = profileRows.querySelector<HTMLElement>(`.frow[data-k="${k}"]`);
    if (!row) return;
    const sel = row.querySelector('select')!;
    // (The options' states as the view's menu holds them now; a field another row plots, off.)
    for (const [i, o] of [...main.options].entries()) {
      const q = sel.options[i];
      if (!q) continue;
      q.disabled = o.disabled || (used.includes(q.value) && q.value !== used[k]);
      q.hidden = o.hidden;
    }
    sel.value = used[k];
    // A view that holds one field has only that one on the GPU.
    sel.disabled = !!cfg.viewBaked;
    sel.title = cfg.viewBaked
      ? 'A view that holds one field can profile only that field: change the field shown to profile another'
      : k === 0 ? 'The field plotted along the lines, against the left axis' : k === 1 ? 'A field plotted along the same lines, against the axis on the right' : 'A field plotted along the same lines, on its own scale (hover the plot to read it)';
    const log = row.querySelector<HTMLInputElement>('input[data-log]')!;
    log.checked = profileLog(f);
    log.disabled = LINEAR_FIELDS.has(f.field);
    row.querySelector<HTMLElement>('button[data-rm]')!.style.visibility = n > 1 ? '' : 'hidden';
  });
  // A field more: while the run offers one no row plots.
  const add = $<HTMLButtonElement>('profile-add');
  add.disabled = !!cfg.viewBaked || !nextProfileField();
}

/** The first field the run offers that no row plots yet, or null. */
function nextProfileField(): ProfileFieldChoice | null {
  const used = profileFields.map(fieldValue);
  const o = [...$<HTMLSelectElement>('field').options].find((q) => !q.disabled && !q.hidden && !(q.parentElement as HTMLOptGroupElement | null)?.hidden && !used.includes(q.value));
  if (!o) return null;
  const [field, species] = o.value.split(':').map(Number);
  return { field, species: species || 0, log: null };
}

/** The panel's list of lines (with two or more): each one's colour and number, to swap its ends or remove it. */
let profileLinesDrawn = '';
function syncProfileLines() {
  const several = profiles.length > 1;
  const html = several
    ? profiles.map((p, i) => `<span class="chip" style="--c:${PROFILE_COLORS[p.color]}"><i></i>${i + 1}<button type="button" data-swap="${i}" title="Swap line ${i + 1}'s ends" aria-label="Swap line ${i + 1}'s ends">&#8646;</button><button type="button" data-del="${i}" title="Remove line ${i + 1}" aria-label="Remove line ${i + 1}">&times;</button></span>`).join('')
    : '';
  const list = $('profile-lines');
  if (html !== profileLinesDrawn) list.innerHTML = profileLinesDrawn = html;
  list.hidden = !several;
  // One Swap for one line; with several each has its own.
  $('profile-swap').hidden = several;
}

/** Swap a line's ends. */
function swapProfile(i: number) {
  const p = profiles[i];
  if (!p) return;
  [p.a, p.b] = [p.b, p.a];
  profileHover = null;
  requestProfile();
  drawProfileLine();
}

profileRows.addEventListener('change', (e) => {
  const t = e.target as HTMLElement;
  if (!profiles.length) return;
  if (t instanceof HTMLSelectElement && t.dataset.k !== undefined) {
    const k = Number(t.dataset.k);
    const [field, species] = t.value.split(':').map(Number);
    profileFields[k] = { field, species: species || 0, log: null };
  } else if (t instanceof HTMLInputElement && t.dataset.log !== undefined) {
    const f = profileFields[Number(t.dataset.log)];
    if (f) f.log = t.checked;
  } else return;
  profileRanges = [];
  syncProfileMenu();
  drawProfilePop();
  requestProfile();
});
profileRows.addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-rm]');
  if (!b || profileFields.length < 2) return;
  profileFields.splice(Number(b.dataset.rm), 1);
  profileRanges = [];
  syncProfileMenu();
  drawProfilePop();
  requestProfile();
});
$('profile-add').addEventListener('click', () => {
  const f = nextProfileField();
  if (!f || !profiles.length) return;
  profileFields.push(f);
  profileRanges = [];
  syncProfileMenu();
  drawProfilePop();
  requestProfile();
});
$('profile-swap').addEventListener('click', () => swapProfile(0));
$('profile-lines').addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest('button');
  if (!b) return;
  if (b.dataset.swap !== undefined) swapProfile(Number(b.dataset.swap));
  else if (b.dataset.del !== undefined) removeProfile(Number(b.dataset.del));
});
$('profile-close').addEventListener('click', () => closeProfile());
$('profile-csv').addEventListener('click', () => {
  const lines = profiles.filter((p) => profileSamples(p));
  if (!lines.length) return;
  const u = units();
  const projected = profileSamples(lines[0])!.projection !== null;
  const unit = (l: string) => (l ? ` (${l})` : '');
  const ms = profileFields.map((f) => profileMeasure(f.field, projected));
  const csv = profileCsv(
    lines.map((p) => {
      const d = profileSamples(p)!;
      return { s: Array.from(d.s, (v) => v * u.scale('length')), values: profileFields.map((_, k) => {
        const q = samplesOf(p, k);
        return q ? Array.from(q.values, (v) => v * u.scale(ms[k])) : null;
      }), levels: d.levels };
    }),
    { distance: `distance${unit(u.label('length'))}`, values: profileFields.map((f, k) => `${profileFieldName(f.field, f.species, projected)}${unit(u.label(ms[k]))}`) },
  );
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}-profile.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
});
// Hovering the plot marks the distance there on every line.
profileCanvas.addEventListener('pointermove', (e) => {
  const list = profiles.map(profileSamples).filter((d): d is ProfileSamples => !!d);
  if (!list.length) return;
  const r = profileCanvas.getBoundingClientRect();
  const at = distanceAt(e.clientX - r.left, r.width, plotLength(list), profileFields.length > 1);
  profileHoverFromLine = false;
  if (at !== profileHover?.s) {
    profileHover = at === null ? null : { s: at, line: -1 };
    drawProfilePop();
    drawProfileLine();
  }
});
profileCanvas.addEventListener('pointerleave', () => {
  profileHover = null;
  drawProfilePop();
  drawProfileLine();
});
// The panel is moved by its title bar.
{
  const head = profilePop.querySelector<HTMLElement>('.head')!;
  let grab: { dx: number; dy: number } | null = null;
  head.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest('button') || phone.matches) return;
    head.setPointerCapture(e.pointerId);
    const r = profilePop.getBoundingClientRect();
    grab = { dx: e.clientX - r.left, dy: e.clientY - r.top };
  });
  head.addEventListener('pointermove', (e) => {
    if (!grab) return;
    profilePopMoved = true;
    const x = Math.max(4 - profilePop.offsetWidth / 2, Math.min(window.innerWidth - profilePop.offsetWidth / 2, e.clientX - grab.dx));
    const y = Math.max(0, Math.min(window.innerHeight - 24, e.clientY - grab.dy));
    profilePop.style.left = `${Math.round(x)}px`;
    profilePop.style.top = `${Math.round(y)}px`;
  });
  const drop = () => (grab = null);
  head.addEventListener('pointerup', drop);
  head.addEventListener('pointercancel', drop);
}

const escapeHtml = (t: string) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Draw the panel's plot and words from the samples last read. */
function drawProfilePop() {
  if (profilePop.hidden || !profiles.length || !cfg) return;
  const f = profileFields[0];
  const lines = profiles.map((p) => ({ p, d: profileSamples(p) }));
  const first = lines.find((l) => l.d)?.d;
  const u = units();
  const projected = first ? first.projection !== null : projecting();
  const ms = profileFields.map((q) => profileMeasure(q.field, projected));
  const m = ms[0];
  const named = (fld: { field: number; species: number }, un: string) => `${profileFieldName(fld.field, fld.species, projected)}${un ? `, ${un}` : ''}`;
  const extra = profileFields.length - 1;
  const title = profilePop.querySelector('.title')!;
  const text = `${named(f, u.label(m))}${extra === 1 ? ` and ${named(profileFields[1], u.label(ms[1]))}` : extra > 1 ? ` and ${extra} more` : ''}${projected ? ` (projected along ${'xyz'[first?.projection ?? view.sliceAxis]})` : ''}`;
  if (title.textContent !== text) title.textContent = text;
  const w = profileCanvas.clientWidth;
  const h = profileCanvas.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  if (profileCanvas.width !== Math.round(w * dpr) || profileCanvas.height !== Math.round(h * dpr)) {
    profileCanvas.width = Math.round(w * dpr);
    profileCanvas.height = Math.round(h * dpr);
  }
  const ctx = profileCanvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cs = getComputedStyle(document.documentElement);
  const color = (v: string, fb: string) => cs.getPropertyValue(v).trim() || fb;
  const colors = { dim: color('--dim', '#8792a8'), frame: color('--line', '#262c3a'), mark: color('--text', '#d6dbe6') };
  const ls = u.scale('length');
  // Each field: every line's samples in the units shown, and its axis, eased toward the values'
  // range while the run goes (widened at once, narrowed by time).
  const now = performance.now();
  const plotted = profileFields.map((fld, k) => {
    const log = profileLog(fld);
    const vs = u.scale(ms[k]);
    const series = profiles.flatMap((p) => {
      const d = samplesOf(p, k);
      return d ? [{ s: Array.from(d.s, (v) => v * ls), values: Array.from(d.values, (v) => v * vs), solid: d.solid, levels: k === 0 && cfg.amrLevels > 0 ? d.levels : undefined, color: PROFILE_COLORS[p.color] }] : [];
    });
    const target = series.length ? seriesRange(series, log) : null;
    const key = `${fld.field}:${fld.species}:${log}:${projected}:${unitSys}:${profiles.length}`;
    if (target) {
      const r = profileRanges[k];
      if (!r || r.key !== key) profileRanges[k] = { key, range: target, at: now };
      else {
        // It widens at once; it narrows by time (a quarter second to close most of the way),
        // however often the samples come.
        const e = 1 - Math.exp(-(now - r.at) / 250);
        const ease = (cur: number, to: number, grow: boolean) => (grow ? to : cur + e * (to - cur));
        r.range = [ease(r.range[0], target[0], target[0] < r.range[0]), ease(r.range[1], target[1], target[1] > r.range[1])];
        r.at = now;
      }
    }
    return { series, log, range: target ? profileRanges[k]!.range : null };
  });
  const [main, ...more] = plotted;
  const series = main.series;
  drawProfilePlot(ctx, w, h, { ...main, mark: profileHover ? profileHover.s * ls : null, colors, more });
  // Under the plot: the values at the distance hovered, or the line's ends and length.
  const foot = profilePop.querySelector('.foot')!;
  const lu = u.label('length');
  const len = (v: number) => `${fmt3(v * ls)}${lu ? ` ${lu}` : ''}`;
  const pos = (q: Vec3) => `(${[0, 1, 2].slice(0, cfg.ndim).map((k) => fmt3(fracPos(cfg, q[k], k) * ls)).join(', ')})`;
  let html: string;
  const several = profiles.length > 1;
  if (profileHover && series.length) {
    const hs = profileHover.s;
    const parts = lines.flatMap(({ p, d }, i) => {
      if (!d) return [];
      const k = sampleNear(d.s, hs);
      if (k < 0) return [];
      // Each field's value at the same sample, after slashes.
      const v = profileFields
        .map((_, j) => {
          const q = samplesOf(p, j);
          return !q || k >= q.values.length ? '–' : q.solid[k] ? 'solid' : u.fmt(q.values[k], ms[j]);
        })
        .join(' / ');
      const lev = cfg.amrLevels > 0 ? `, level ${d.levels[k]}` : '';
      return [several ? `<span style="color:${PROFILE_COLORS[p.color]}">${i + 1}</span> ${escapeHtml(v)}${lev}` : `${escapeHtml(v)}${lev}`];
    });
    html = `at ${escapeHtml(len(hs))}: ${parts.join(several ? '; ' : '')}`;
  } else if (!several) {
    const d = lines[0].d;
    html = escapeHtml(`${pos(profiles[0].a)} to ${pos(profiles[0].b)}${d ? `, ${len(d.length)} long` : ''}`);
  } else {
    html = escapeHtml(`${profiles.length} lines; hover the plot to read them`);
  }
  if (foot.innerHTML !== html) {
    foot.innerHTML = html;
    // (Several fields' values can run past the panel's edge: all of them in the tooltip.)
    foot.setAttribute('title', foot.textContent ?? '');
  }
}


/**
 * The API's profile (api.ts profile()): along a line given in fractions of the box, or one of
 * those on show (`line`, from 0). `show`: true puts the line on show alone, 'add' adds it to
 * those on show.
 */
async function apiProfile(o: { from?: number[]; to?: number[]; field?: string | number; n?: number; projection?: number | null; show?: boolean | 'add'; line?: number }) {
  const s = sim;
  if (!s) throw new Error('no simulation is loaded');
  // (A 1D spherical run's profiles are radial, sampled along r alone: plans/one-dimensional.md.)
  if (spherical(cfg) && cfg.ndim > 1) throw new Error('profiles are not available on a spherical grid yet');
  let a: Vec3;
  let b: Vec3;
  const onShow = o.from === undefined && o.to === undefined;
  if (onShow) {
    if (!profiles.length) throw new Error('no profile is on show in the page: give `from` and `to`');
    const i = o.line ?? 0;
    const p = Number.isInteger(i) ? profiles[i] : undefined;
    if (!p) throw new Error(`profile: no line ${i} on show: there ${profiles.length === 1 ? 'is 1' : `are ${profiles.length}`}, numbered from 0`);
    [a, b] = [p.a, p.b];
  } else {
    if (o.line !== undefined) throw new Error('profile: give `line` (a line on show) or `from` and `to`, not both');
    const point = (v: unknown, name: string): number[] => {
      if (!Array.isArray(v) || v.length < cfg.ndim || v.length > 3 || !v.every((x) => typeof x === 'number' && Number.isFinite(x))) {
        throw new Error(`profile: "${name}" takes [x, y${cfg.ndim === 3 ? ', z' : ''}] in fractions of the box`);
      }
      return v;
    };
    // A 1D run's line made horizontal, or radial from the center (lockedLine).
    const c = clipSegment(...lockedLine(point(o.from, 'from') as Vec3, point(o.to, 'to') as Vec3), cfg.ndim);
    if (!c) throw new Error('profile: that line does not cross the box');
    [a, b] = c;
  }
  if (o.show !== undefined && o.show !== true && o.show !== false && o.show !== 'add') throw new Error('profile: "show" is true (the line alone), "add" (with those on show) or false');
  const fallback = profiles.length && (onShow || o.show === 'add') ? { field: profileFields[0].field, species: profileFields[0].species } : { field: view.field, species: view.species };
  const f = resolveField(o.field, cfg.problem.species ?? [], fallback);
  if (!fieldOffered(f.field)) throw new Error(`profile: ${FIELDS[f.field]} is not available in this run`);
  if (cfg.viewBaked && (f.field !== view.field || (f.field === MASS_FRACTION_FIELD && f.species !== view.species))) {
    throw new Error(`profile: a view that holds one field can profile only that field (${viewFieldName()}); change the field shown first`);
  }
  let projection: number | null = null;
  if (o.projection !== undefined && o.projection !== null) {
    if (cfg.ndim !== 3) throw new Error('profile: a 2D run has nothing to project');
    if (![0, 1, 2].includes(o.projection)) throw new Error('profile: "projection" is an axis: 0, 1 or 2');
    projection = o.projection;
  }
  if (o.show === 'add' && profiles.length >= MAX_PROFILES) throw new Error(`profile: at most ${MAX_PROFILES} lines on show at once`);
  const r = await profileOf(s, a, b, { ...f, projection }, o.n);
  if (o.show) {
    // On show alone: its field alone. Added: its field first, the panel's others kept, and the
    // lines already there sampled again.
    if (o.show === true) {
      profiles = [];
      profileFields = [{ ...f, log: null }];
    } else {
      const rest = profileFields.slice(1).filter((q) => q.field !== f.field || q.species !== f.species);
      profileFields = [{ ...f, log: profileFields[0].field === f.field && profileFields[0].species === f.species ? profileFields[0].log : null }, ...rest];
    }
    addProfile(a, b, r);
    openProfilePop(true);
  }
  return {
    from: a.slice(0, cfg.ndim),
    to: b.slice(0, cfg.ndim),
    field: profileFieldName(f.field, f.species, projection !== null),
    index: f.field,
    ...(f.field === MASS_FRACTION_FIELD ? { species: cfg.problem.species![f.species] } : {}),
    projection,
    length: r.length,
    s: r.s,
    values: r.values,
    levels: r.levels,
    solid: r.solid,
  };
}

// --- query tool ---------------------------------------------------------------------------
// While it is in hand, the cell under the pointer is read back and shown beside it, with the
// source, sink or point mass under the pointer if there is one. One read is in flight at a
// time: the latest position is read next, and a running simulation is read again every 150 ms
// while the pointer rests.
//
// A click pins the readout. On a source, a sink or a point mass it then follows that one;
// anywhere else a massless tracer is let go at the spot (a point mass with a token mass, no
// sink: it goes where gravity takes it) and the readout follows the tracer. A pinned readout
// stays, whatever tool is in hand, until it is unpinned or what it follows is removed.
const cellPop = $('cell-pop');
let queryPoint: { x: number; y: number } | null = null;
let queryBusy = false;

/** What a query can be about besides the gas: one of the run's point masses, or a source that stays put. */
type Thing = { mass: number } | { source: SourceSpec };
/** A pinned query: what it follows, and whether that is a tracer the pin itself let go. */
let pin: (Thing & { own?: boolean }) | null = null;
/** The gas where the pinned thing is, as last read, when that was asked for, and whether a read is in flight. */
let pinCell: CellProbe | null = null;
/** The pinned tracer's Fixed as last set here, until the masses read back show it (undefined: as read). */
let pinFixed: boolean | undefined;
let pinAsked = 0;
let pinBusy = false;

function queryAt(clientX: number, clientY: number) {
  if (pin) return; // a pinned readout stays where it is
  queryPoint = { x: clientX, y: clientY };
  if (!queryBusy) void runQuery();
}

function hideCellPop() {
  queryPoint = null;
  if (!pin) cellPop.hidden = true;
}

/** Let the pin go, without removing what it followed. */
function dropPin() {
  pin = null;
  pinCell = null;
  pinFixed = undefined;
  recorded = null;
  queryView.reset();
  cellPop.classList.remove('pinned');
  cellPop.hidden = true;
}

/** Unpin the readout: a tracer the pin let go is taken out of the run with it. */
function unpin() {
  const was = pin;
  dropPin();
  if (was && 'mass' in was && was.own) removePlaced((_, index) => index === was.mass);
}

/**
 * Whether the pinned thing can be removed: a source, or a mass the tools placed (not the preset's
 * own). A tracer the readout let go itself is not offered: unpinning already takes it out.
 */
const pinRemovable = () => !!pin && !pin.own && ('source' in pin || pin.mass >= (sim?.massCount ?? 0) - placed.length);

/** Remove what the readout is pinned to: the sink, the source (with the particle it rides) or the tracer. */
function removePinned() {
  const was = pin;
  if (!was || !sim || !pinRemovable()) return;
  dropPin();
  if ('mass' in was) removePlaced((_, index) => index === was.mass);
  else {
    sources = sources.filter((q) => q !== was.source);
    pushSources(sources);
  }
  syncTools();
}

/** Where a thing is now, in fractions of the box; null for a mass not read back yet. */
function thingPos(t: Thing): number[] | null {
  if ('source' in t) return t.source.pos;
  const m = massesNow()[t.mass];
  return m ? m.pos.map((v, a) => boxFrac(cfg, v, a)) : null;
}

/** The source, sink or point mass drawn within a few pixels of a point of the canvas, nearest first. */
function thingAt(px: number, py: number): Thing | null {
  if (!cfg) return null;
  const at = projector();
  const things: Thing[] = [...massesNow().map((_, i) => ({ mass: i })), ...sources.filter((q) => q.carrier === undefined).map((q) => ({ source: q }))];
  let best: Thing | null = null;
  let least = 10;
  for (const t of things) {
    const f = thingPos(t);
    if (!f) continue;
    const c = at(f);
    const d = Math.hypot(c.x - px, c.y - py);
    if (c.w > 0 && d < least) [best, least] = [t, d];
  }
  return best;
}

/** The point of the box under a point of the canvas (in the volume view, the gas seen there), or null. */
async function boxPointAt(px: number, py: number): Promise<Vec3 | null> {
  let depth = view.dist;
  if (!flatView()) {
    // Volume view: the gas that is seen under the pointer.
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const [ux, uy] = [(2 * px) / w - 1, 1 - (2 * py) / h];
    const t = await renderer.pick(ux, uy);
    if (t === null) return null;
    const f = cameraFrame(view, w / h);
    depth = t / Math.hypot(1, ux * (w / h) * f.tan, uy * f.tan);
  }
  const f = locate(px, py, depth).pos;
  if (cfg.ndim < 3) f[2] = 0;
  return f.some((v) => v < 0 || v >= 1) ? null : f;
}

/** The cell of the gas at a point of the box (in fractions of it), or null outside the box. */
async function probeAt(s: Simulation, f: number[], stencil = false): Promise<CellProbe | null> {
  const g = [f[0], f[1], cfg.ndim < 3 ? 0 : f[2]] as [number, number, number];
  return g.some((v) => !(v >= 0 && v < 1)) ? null : s.probe(g, { stencil });
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
    // A source, sink or point mass under the pointer is shown with the gas where it is.
    const thing = thingAt(px, py);
    const f = thing ? thingPos(thing) : await boxPointAt(px, py);
    // The network's graph needs the cell's species, which come with its stencil.
    const c = f ? await probeAt(s, f, netGraphShown()) : null;
    if (s !== sim || !queryPoint || tool !== 'query' || pin) return;
    if (!c && !thing) cellPop.hidden = true;
    else showQuery(thing, c, queryPoint.x, queryPoint.y);
  } catch {
    if (!pin) cellPop.hidden = true;
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

/** A click with the query tool: pin the readout to what is there, or to a tracer let go there. */
async function pinAt(clientX: number, clientY: number) {
  const s = sim;
  if (!s || building) return;
  const r = canvas.getBoundingClientRect();
  const [px, py] = [clientX - r.left, clientY - r.top];
  unpin();
  let to: (Thing & { own?: boolean }) | null = thingAt(px, py);
  if (!to) {
    const f = await boxPointAt(px, py);
    if (!f || s !== sim) return;
    // A tracer, held where it is (the readout's Fixed lets it go): softened over two cells, so
    // that a mass it passes does not fling it.
    const ok = addPlaced(s, { ...carrierSpec(f, [0, 0, 0], 2 / cfg.n), fixed: true }, 'query');
    notice(ok ? '' : `A run holds at most ${MAX_MASSES} sinks, moving sources, tracers and point masses`);
    if (!ok) return;
    to = { mass: s.massCount - 1, own: true };
    lastDiag = undefined;
    syncControls();
  }
  queryPoint = null;
  pin = to;
  pinCell = null;
  pinFixed = undefined;
  recorded = null;
  queryView.reset();
  pinAsked = 0;
  cellPop.classList.add('pinned');
}

/** Keep a pinned readout with what it follows: beside it on the screen, and read again as the run goes on. Called once per frame. */
function updatePin(now: number) {
  const s = sim;
  if (!pin || !s || !cfg) return;
  // A source that was pinned may have been cleared away by its tool.
  if ('source' in pin && !sources.includes(pin.source)) return dropPin();
  const f = thingPos(pin);
  if (!f) {
    // A mass not read back yet (just placed); one that has left the run for good is let go.
    if ('mass' in pin && lastDiag && pin.mass >= s.massCount) dropPin();
    return;
  }
  if (!pinBusy && now - pinAsked >= 150) {
    pinBusy = true;
    pinAsked = now;
    probeAt(s, f, true).then(
      (c) => {
        pinBusy = false;
        if (s === sim) pinCell = c;
      },
      () => (pinBusy = false),
    );
  }
  const c = projector()(f);
  const r = canvas.getBoundingClientRect();
  showQuery(pin, pinCell, r.left + c.x, r.top + c.y);
}

/** The readout: what is known of a source, sink or point mass, then the cell of the gas there. */
function showQuery(thing: Thing | null, c: CellProbe | null, x: number, y: number) {
  const dims = cfg.ndim;
  const u = units();
  const vec = (v: number[], m: Measure) => {
    const l = u.label(m);
    return `(${v.slice(0, dims).map((a) => fmt3(a * u.scale(m))).join(', ')})${l ? ` ${l}` : ''}`;
  };
  const rows: Row[] = [];
  // A row of a number (or a vector's components) in measure m, kept as a series while pinned.
  // (The pinned thing's own rows are named after it in the series: its velocity, the gas's.)
  let owner = '';
  const num = (key: string, label: string, v: number | number[], m: Measure | null, text?: string) => {
    const raw = Array.isArray(v) ? v.slice(0, dims) : [v];
    const shown = text ?? (Array.isArray(v) ? (m ? vec(v, m) : `(${raw.map((a) => fmt3(a)).join(', ')})`) : m ? u.fmt(v, m) : fmt3(v));
    rows.push({ key, label, text: shown, name: owner ? `${owner} ${label.toLowerCase()}` : undefined, series: { raw, scale: m ? u.scale(m) : 1, unit: m ? u.label(m) : '' } });
  };
  const txt = (key: string, label: string, text: string, cls?: string) => rows.push({ key, label, text, cls });
  let title = c ? `Cell (${c.index.join(', ')})` : 'Outside the box';
  if (thing) {
    const box = cfg.problem.box;
    const m = 'mass' in thing ? massesNow()[thing.mass] : undefined;
    const source = 'source' in thing ? thing.source : sources.find((q) => q.carrier === thing.mass);
    const kind = source ? (source.decay > 0 ? 'Sink' : 'Source') : !m ? 'Point mass' : !m.massless ? 'Point mass' : m.sinkRate > 0 && m.sinkRadius > 0 ? 'Sink' : 'Tracer';
    title = `${kind}${'mass' in thing ? ` ${thing.mass + 1}` : ''}${m?.fixed || (source && !m) ? ', fixed' : ''}${c ? ` in cell (${c.index.join(', ')})` : ''}`;
    owner = kind;
    if (m) {
      if (m.massless) txt('mass', 'Mass', 'none');
      else if (cfg.G > 0) num('mass', 'Mass', m.gm / cfg.G, 'mass');
      else num('mass', 'Mass', m.gm, null);
      num('m-pos', 'Position', m.pos, 'length');
      num('m-vel', 'Velocity', m.vel, 'velocity');
      num('m-speed', 'Speed', Math.hypot(...m.vel.slice(0, dims)), 'velocity');
      num('m-soft', 'Softening', m.soft, 'length');
      if (m.sinkRadius > 0 && m.sinkRate > 0) {
        num('m-sink-r', 'Sink radius', m.sinkRadius, 'length');
        num('m-taken', 'Gas taken in', m.accreted, 'mass');
      }
    } else if (source) {
      num('s-pos', 'Position', source.pos.map((v, a) => fracPos(cfg, v, a)), 'length');
    }
    if (source) {
      num('s-radius', 'Radius', source.radius * box, 'length');
      if (source.decay > 0) num('s-decay', 'Removal rate', source.decay, 'rate');
      else num('s-rate', 'Density fed in', source.rate, [1, -3, -1] as unknown as Measure);
      if (!m && Math.hypot(...source.vel) > 0) num('s-vel', 'Gas leaves at', source.vel, 'velocity');
    }
    if (c) txt('head', 'Gas there', '', 'head');
    owner = '';
  }
  if (c) {
    const eos = cellEos(c).eos;
    const rho = c.u1[0];
    const vel = [c.u1[1] / rho, c.u1[2] / rho, c.u1[3] / rho];
    const eint = c.u2[1];
    const p = eos.pressure(rho, eint);
    const cs = eos.soundSpeed(rho, p);
    const speed = Math.hypot(...vel.slice(0, dims));
    if (!thing) num('pos', 'Position', c.pos, 'length');
    num('dx', 'Cell size', c.dx, 'length');
    if (c.block) {
      const b = c.block;
      num('level', 'Level', b.level, null, `${b.level} of ${cfg.amrLevels}`);
      txt('block', 'Block', `(${b.coords.join(', ')}), slot ${b.slot}`);
      if (b.indicator === null) txt('indicator', 'Indicator', 'not yet read');
      else num('indicator', 'Indicator', b.indicator, null, `${fmt3(b.indicator)} (split ${cfg.refineHi}, merge ${cfg.refineLo})`);
    }
    if (c.u2[3] > 0.5) txt('solid', 'Obstacle', 'solid cell; gas held for later', 'solid');
    num('rho', 'Density', rho, 'density');
    num('vel', 'Velocity', vel, 'velocity');
    num('speed', 'Speed', speed, 'velocity');
    num('mach', 'Mach number', cs > 0 ? speed / cs : 0, null);
    num('p', 'Pressure', p, 'pressure');
    num('T', 'Temperature', eos.temperature(rho, eint), 'temperature');
    num('cs', 'Sound speed', cs, 'velocity');
    if (u.label('specificEnergy')) num('e', 'Internal energy', eint / rho, 'specificEnergy');
    else num('e', 'Internal energy', eint / rho, null, `${fmt3(eint / rho)} per mass`);
    if (u.label('pressure')) num('E', 'Total energy', c.u2[0], 'pressure');
    else num('E', 'Total energy', c.u2[0], null, `${fmt3(c.u2[0])} per volume`);
    num('scalar', 'Scalar', c.u2[2] / rho, null);
  }
  cellPop.querySelector('.title')!.textContent = title;
  const pinned = !!pin;
  queryView.setRows(rows, pinned);
  if (pinned && c?.stencil) setEquations(c);
  else queryView.setEquations(null);
  showNetGraph(c);
  // A new reading of the pinned cell: add it to the series.
  if (pinned && c && c !== recorded) {
    recorded = c;
    queryView.record((lastDiag?.t ?? 0) * u.scale('time'), lastDiag?.steps ?? 0, u.label('time'));
  }
  $('cell-remove').hidden = !pinRemovable();
  ($('cell-export') as HTMLElement).hidden = !pinned;
  // A pinned tracer can be held where it is or let go (the GPU's flag, or the one just set).
  const tracer = pin && 'mass' in pin ? massesNow()[pin.mass] : undefined;
  const canFix = !!tracer && tracer.massless && !(tracer.sinkRate > 0 && tracer.sinkRadius > 0) && !sources.some((q) => pin && 'mass' in pin && q.carrier === pin.mass);
  ($('cell-fixed-row') as HTMLElement).hidden = !canFix;
  if (canFix && document.activeElement !== cellFixed) cellFixed.checked = pinFixed ?? tracer.fixed;
  cellPop.hidden = false;
  // Beside the pointer (or what it is pinned to), kept on screen.
  const pw = cellPop.offsetWidth;
  const ph = cellPop.offsetHeight;
  const left = x + 18 + pw > window.innerWidth - 4 ? x - 18 - pw : x + 18;
  const top = Math.min(window.innerHeight - ph - 4, Math.max(4, y + 18));
  cellPop.style.left = `${Math.round(Math.max(4, left))}px`;
  cellPop.style.top = `${Math.round(top)}px`;
}
/** The readout's rows, equations and their series (ui/readout_view.ts). */
const queryPop = $('diag-pop');
const queryView = new ReadoutView(cellPop.querySelector('dl')!, cellPop.querySelector('.eqs')!, { el: queryPop, title: queryPop.querySelector('.title')!, canvas: queryPop.querySelector('canvas')! }, cellPop);
/** The reading of the pinned cell last added to the series. */
let recorded: CellProbe | null = null;

// --- the burning network's graph -----------------------------------------------------------
// A run that burns with the network (or watches for it) shows its reactions at the cell read, as
// a graph under the readout's equations (ui/burn_graph.ts). It starts collapsed to its header;
// whether it is open is kept per viewer.
// On a phone it starts collapsed every time, as the readout's other sections do there.
let netGraphOpen = false;
try {
  netGraphOpen = !phone.matches && localStorage.getItem('shaderous.netgraph') === 'open';
} catch {
  /* storage unavailable: collapsed */
}
const netGraph = new BurnGraphView(cellPop.querySelector('.burn-graph')!, netGraphOpen, (open) => {
  netGraphOpen = open;
  if (phone.matches) return;
  try {
    localStorage.setItem('shaderous.netgraph', open ? 'open' : 'closed');
  } catch {
    /* kept for this page only */
  }
});
queryView.recordAlso(() => netGraph.series());
// On a phone the readout is a sheet along the bottom, its sections one above another, each
// folded to its header until it is tapped (style.css): the values and the equations here, the
// network in its own view. Elsewhere the headers do nothing.
cellPop.addEventListener('click', (e) => {
  const head = (e.target as Element).closest<HTMLElement>('.fold-head');
  if (head && phone.matches) cellPop.classList.toggle(`open-${head.dataset.fold}`);
});
queryView.popsOver(cellPop.querySelector('.burn-graph')!);

/** The run burns with the network and carries its species (all thirteen, or the watched three). */
function netGraphRun(): boolean {
  const spec = cfg ? burnSpec(cfg) : null;
  return !!spec && 'network' in spec && !!cellReactions(spec, { ...cfg.problem, units: problemUnits(cfg.problem, cfg.params) }, 1, 1, []);
}
/** A hover's read takes the cell's species (its stencil) only for an open graph; a pinned read always has them. */
const netGraphShown = () => netGraphOpen && netGraphRun();

/** The graph of the network's reactions at the cell read (none: hidden). */
function showNetGraph(c: CellProbe | null) {
  const rx = c?.stencil?.species?.[13];
  const rho = c?.u1[0] ?? 0;
  if (!netGraphRun() || !c || !rx || !(rho > 0)) return netGraph.update(null);
  const X = rx.map((v) => v / rho);
  const T = cellEos(c).eos.temperature(rho, c.u2[1]);
  const cell = cellReactions(burnSpec(cfg), { ...cfg.problem, units: problemUnits(cfg.problem, cfg.params) }, rho, T, X);
  if (!cell) return netGraph.update(null);
  netGraph.update({ cell, network: runBurnRates()?.(rho, T, X) ?? null });
}
$('cell-export').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([queryView.log.csv()], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}-query.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
});
$('cell-remove').addEventListener('click', removePinned);
const cellFixed = $('cell-fixed') as HTMLInputElement;
cellFixed.addEventListener('change', () => {
  if (!sim || !pin || !('mass' in pin)) return;
  pinFixed = cellFixed.checked;
  sim.setTracerFixed(pin.mass, pinFixed);
  logAction({ kind: 'fix', index: pin.mass, fixed: pinFixed });
  cellFixed.blur();
});

// --- the equations at a pinned cell ---------------------------------------------------------
// The pinned readout shows the equations the run solves, each term colored by its size in that
// cell against the largest term of its equation (ui/terms.ts), on a log ramp over six decades.

/** Start-up damping's rate at time t (hydro/damping.wgsl start_damping_rate). */
function calmRate(t: number): number {
  const sd = cfg.startDamping;
  if (!sd || !(sd.rate > 0) || !(sd.time > 0) || !(t < sd.time)) return 0;
  return 0.5 * sd.rate * (1 + Math.cos((Math.PI * Math.max(t, 0)) / sd.time));
}

/**
 * The equation of state of the pinned cell, and of a cell of any mass fractions: where the species
 * set it, each cell's own (species/mixture.ts), else the run's.
 */
function cellEos(c: CellProbe | null): { eos: Eos; at: ((X: number[]) => Eos) | null } {
  const coeffs = problemCoefficients(cfg.problem, cfg.eosParams.gamma);
  const at = coeffs ? (X: number[]) => makeEos(cfg.eos, cfg.eosParams, mixMoments(coeffs, X)) : null;
  const rx = c?.stencil?.species?.[13];
  const rho = c?.u1[0] ?? 0;
  return { eos: at && rx && rho > 0 ? at(rx.map((r) => r / rho)) : makeEos(cfg.eos, cfg.eosParams), at };
}

/** The pinned run's burn rates, made once per run (ui/burn_rates.ts). */
let rates: { of: SimConfig; fn: ReturnType<typeof burnRates> } | null = null;
function runBurnRates() {
  if (rates?.of !== cfg) rates = { of: cfg, fn: burnRates(burnSpec(cfg), { ...cfg.problem, units: problemUnits(cfg.problem, cfg.params) }, () => lastDiag?.dt ?? 0) };
  return rates.fn;
}

/** The weight of a source or sink at the cell at `pos` (relative to the box center): hydro/sources.wgsl source_weight. */
function sourceWeight(q: SourceSpec, pos: number[]): { w: number; vel: number[] } {
  let at = q.pos as number[];
  let vel = q.vel as number[];
  if (q.carrier !== undefined) {
    const m = massesNow()[q.carrier];
    if (!m) return { w: 0, vel };
    at = m.pos.map((v, a) => boxFrac(cfg, v, a));
    vel = m.vel;
  }
  const d = [0, 1, 2].map((a) => {
    if (a === 2 && cfg.ndim < 3) return 0;
    let v = boxFrac(cfg, pos[a], a) - at[a];
    if (cfg.bcLo[a] === 0) v -= Math.round(v);
    return v;
  });
  const r = Math.hypot(...d) / q.radius;
  if (!(r < 1)) return { w: 0, vel };
  const e0 = Math.min(1 - q.softness, 0.999);
  const t = Math.min(1, Math.max(0, (r - e0) / (1 - e0)));
  return { w: 1 - t * t * (3 - 2 * t), vel };
}

/** The share of the gas the point masses' sinks take per unit time at `pos` (gravity/bh.wgsl sink_fraction). */
function massSinkRate(pos: number[]): number {
  let k = 0;
  for (const m of massesNow()) {
    if (!(m.sinkRadius > 0 && m.sinkRate > 0)) continue;
    const d = [0, 1, 2].map((a) => (a === 2 && cfg.ndim < 3 ? 0 : pos[a] - m.pos[a]));
    const q = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / m.sinkRadius ** 2;
    if (q >= 1) continue;
    // The rate is in units of sqrt(G M / r_sink^3), per unit time for a massless particle.
    const rate = m.massless ? m.sinkRate : m.sinkRate * Math.sqrt(m.gm / m.sinkRadius ** 3);
    k += (1 - q) ** 2 * rate;
  }
  return k;
}

/** The pinned cell's equations, to the queryView. */
function setEquations(c: CellProbe) {
  if (cylindrical(cfg) || spherical(cfg)) return queryView.setEquations(null, 'Equations are shown on Cartesian grids only');
  const ce = cellEos(c);
  const rot = rotationOf(cfg);
  const fd = lastDiag?.frame;
  const omega = [rot[0], rot[1], fd ? fd.omega : rot[2]];
  const shear = cfg.problem.shear;
  const cosmo = cosmologyOf(cfg);
  const list = cellEquations({
    ndim: cfg.ndim, dx: c.dx, stencil: c.stencil!, pos: c.pos, eos: ce.eos, eosAt: ce.at, gext: cfg.gext,
    masses: massesNow().filter((m) => !m.massless).map((m) => ({ gm: m.gm, pos: m.pos, soft: m.soft })),
    cooling: cfg.cooling, coolParams: cfg.coolParams, calm: calmRate(lastDiag?.t ?? 0),
    resistivity: cfg.resistivity,
    radiation: radiationOn(cfg) ? { c: cfg.radC, chat: cfg.radChat, a: cfg.radA, localC: cfg.radLocalC, opacity: (rho, T) => meanOpacity(cfg.opacity, cfg.opacityParams, rho, T) } : null,
    species: cfg.problem.species?.length ? { names: cfg.problem.species, burnRates: runBurnRates() } : null,
    frame: {
      omega, centrifugal: rot[3] !== 0, omegaDot: fd?.omegaDot ?? 0, accel: c.stencil?.frameAccel ?? [0, 0, 0],
      shear: shear ? { omega: shear.omega, q: shear.q, stratified: !!shear.stratified } : null,
    },
    sources: sources.map((q) => ({ ...sourceWeight(q, c.pos), rate: q.rate, decay: q.decay })),
    sinkRate: massSinkRate(c.pos),
    hubble: cosmo ? hubble(cosmo, lastDiag?.scale ?? 1) : 0,
  });
  queryView.setEquations(list);
}
$('cell-unpin').addEventListener('click', unpin);

// --- contour labels -----------------------------------------------------------------------
// While the run is paused, the contour under the pointer (slice and projection views) is drawn
// highlighted and its value shown beside the pointer. The map under the pointer is read back,
// one read at a time; the latest position is read next.
const contourPop = $('contour-pop');
let contourPoint: { x: number; y: number } | null = null;
let contourBusy = false;
/** What the label was last found for: the pointer, the view, and whether the run was paused. */
let contourKey = '';

const contoursHoverable = () => !running && view.contours && !tool && !!sim && !building && flatView();

function contourHover(clientX: number, clientY: number) {
  contourPoint = { x: clientX, y: clientY };
  if (!contoursHoverable()) return hideContourLabel();
  if (!contourBusy) void runContourHover();
}

function hideContourLabel() {
  renderer.hoverLevel = 0;
  contourPop.hidden = true;
  contourKey = '';
}

function clearContourHover() {
  contourPoint = null;
  hideContourLabel();
}

/** The label follows the view while the pointer rests (zoom, a new range, another field). */
function refreshContourHover() {
  if (!contourPoint) return;
  if (!contoursHoverable()) {
    if (renderer.hoverLevel || !contourPop.hidden) hideContourLabel();
    return;
  }
  const key = contourViewKey();
  if (key !== contourKey && !contourBusy) void runContourHover();
}

const contourViewKey = () =>
  [contourPoint?.x, contourPoint?.y, view.field, view.species, view.mode, view.sliceAxis, view.slicePos, view.lo, view.hi, view.log, view.nContours, view.zoom, view.cx, view.cy, unitSys].join();

async function runContourHover() {
  const q = contourPoint;
  if (!q || !contoursHoverable()) return;
  contourBusy = true;
  const key = contourViewKey();
  try {
    const r = canvas.getBoundingClientRect();
    const span = sliceSpan() * view.zoom;
    let a = view.cx + (q.x - r.left - canvas.clientWidth / 2 + sliceShift()) / span;
    let b = view.cy - (q.y - r.top - canvas.clientHeight / 2) / span;
    // A spherical run's slice: the point of the (r, theta) map under the (R, z) one.
    if (curvedView(cfg)) [a, b] = sphFromView(cfg, a, b) ?? [-1, -1];
    const hit = a >= 0 && a < 1 && b >= 0 && b < 1 ? await renderer.contourAt(a, b, view, span) : null;
    if (contourPoint !== q || !contoursHoverable()) return;
    contourKey = key;
    if (!hit) {
      renderer.hoverLevel = 0;
      contourPop.hidden = true;
      return;
    }
    renderer.hoverLevel = hit.level;
    contourPop.textContent = `${viewFieldName()} = ${units().fmt(hit.value, viewMeasure())}`;
    contourPop.hidden = false;
    // Beside the pointer, kept on screen.
    const pw = contourPop.offsetWidth;
    const ph = contourPop.offsetHeight;
    const left = q.x + 14 + pw > window.innerWidth - 4 ? q.x - 14 - pw : q.x + 14;
    contourPop.style.left = `${Math.round(Math.max(4, left))}px`;
    contourPop.style.top = `${Math.round(Math.min(window.innerHeight - ph - 4, Math.max(4, q.y - ph - 10)))}px`;
  } catch {
    hideContourLabel();
  } finally {
    contourBusy = false;
    if (contourPoint && contourPoint !== q) void runContourHover();
  }
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
    const axis = cfg.ndim < 3 ? 2 : view.sliceAxis;
    const [ia, ib] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
    const k = sliceSpan() * view.zoom;
    rx = [0, 0, 0, w / 2 - sliceShift() - view.cx * k];
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
  // With mirror images shown, a fraction f of the computed box is at 0.5 + f / 2 of the view's.
  const mir = mirrorShown(cfg, view);
  for (let a = 0; a < 3; a++) {
    if (!mir[a]) continue;
    for (const row of [rx, ry, rw]) {
      row[3] += 0.5 * row[a];
      row[a] *= 0.5;
    }
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
  if (sources.length && sim) pushSources([]);
  sources = [];
  placed = [];
  dropPin();
  toolNotice = '';
}

/**
 * Where a point of the box (in fractions of it) is on the canvas in the current view, in CSS
 * pixels; w is not positive for a point behind the camera.
 */
function projector(): (p: number[]) => { x: number; y: number; w: number } {
  const m = outlineMatrix();
  // A spherical run's slice draws (r, theta) at (R, z) (sphToView).
  const sph = flatView() && curvedView(cfg);
  return (q) => {
    const p = sph ? [...sphToView(cfg, q[0], q[1]), q[2]] : q;
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    return { x: (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w, y: (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w, w };
  };
}

/** The point masses in the run, as last read back (none between a reset and the first readback). */
function massesNow(): Diagnostics['masses'] {
  const live = liveMasses && liveMasses.of === sim ? liveMasses.list : null;
  return (lastDiag && live ? live : lastDiag?.masses) ?? [];
}

/** Greatest length in pixels of a point mass's velocity arrow. */
const VEL_ARROW = 48;

/** Draw the markers of the sources and sinks where they are in the current view. */
let marksDrawn = '';
function drawSources() {
  let html = '';
  // Between resets and the first readback there is nothing to draw (lastDiag is cleared then).
  const masses = massesNow();
  if ((sources.length || masses.length) && cfg) {
    const at = projector();
    const box = cfg.problem.box;
    const vFull = 5 * toolRefs().vRef;
    // An arrow from the marker at `c` (of the point `pos` of the box) along the velocity `vel`:
    // half as long as the drag that would set a new mass moving that fast, no longer than
    // VEL_ARROW pixels, and none where it would be shorter than its own head (a mass slower
    // than a few tenths of the tools' reference speed).
    const velArrow = (c: { x: number; y: number; w: number }, pos: number[], vel: number[]) => {
      const speed = Math.hypot(...vel);
      const long = vFull > 0 ? Math.min(VEL_ARROW, (0.5 * fullDrag() * speed) / vFull) : 0;
      if (long < 4) return '';
      // A point that far along the velocity, in the box: foreshortened as the view sees it.
      const reach = long / locate(c.x, c.y, flatView() ? 1 : c.w).scale;
      const tip = at(pos.map((v, i) => v + (reach * vel[i]) / speed));
      const [ux, uy] = [tip.x - c.x, tip.y - c.y];
      const len = Math.hypot(ux, uy);
      if (!(tip.w > 0) || len < 4) return '';
      const [nx, ny] = [ux / len, uy / len];
      return `<path class="vel" d="M${c.x} ${c.y}L${tip.x} ${tip.y}M${tip.x - 4 * nx - 2.5 * ny} ${tip.y - 4 * ny + 2.5 * nx}L${tip.x} ${tip.y}L${tip.x - 4 * nx + 2.5 * ny} ${tip.y - 4 * ny - 2.5 * nx}" />`;
    };
    // The masses that sources ride are drawn as those sources, where the masses are now.
    const ridden = new Set(sources.map((q) => q.carrier).filter((i) => i !== undefined));
    for (const q of sources) {
      const rides = q.carrier !== undefined ? masses[q.carrier] : undefined;
      // (Between a placement and the next readback the mass is not yet known: where it was put.)
      const where = rides ? rides.pos.map((v, a) => boxFrac(cfg, v, a)) : q.pos;
      const c = at(where);
      if (!(c.w > 0)) continue; // behind the camera
      // Size on screen: from a point one radius away along an axis in the plane of the view.
      const e = at(where.map((v, i) => v + q.radius * locate(c.x, c.y, flatView() ? 1 : c.w).right[i]));
      const r = Math.hypot(e.x - c.x, e.y - c.y);
      if (q.carrier !== undefined) {
        html += `<g class="source moving"><circle cx="${c.x}" cy="${c.y}" r="${r}" />${rides ? velArrow(c, where, rides.vel) : ''}</g>`;
        continue;
      }
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
    // The point masses, where the last readback found them: the sink's edge and a dot, and
    // an arrow along its velocity.
    for (const [index, q] of masses.entries()) {
      if (ridden.has(index)) continue;
      const pos = q.pos.map((v, a) => boxFrac(cfg, v, a));
      const c = at(pos);
      if (!(c.w > 0)) continue;
      const arrow = velArrow(c, pos, q.vel);
      const e = at(pos.map((v, i) => v + (q.sinkRadius / box) * locate(c.x, c.y, flatView() ? 1 : c.w).right[i]));
      // The sink's edge, at its true radius however small, and a dot at the mass. A mass
      // that takes no gas has no edge to show.
      const r = Math.hypot(e.x - c.x, e.y - c.y);
      const edge = q.sinkRate > 0 ? `<circle cx="${c.x}" cy="${c.y}" r="${r}" />` : '';
      const ring = pin && 'mass' in pin && pin.mass === index ? `<circle class="pinned" cx="${c.x}" cy="${c.y}" r="7" />` : '';
      html += `<g class="sink point">${ring}${edge}${arrow}<circle class="dot" cx="${c.x}" cy="${c.y}" r="${Math.min(2.5, Math.max(1, r / 2))}" /></g>`;
    }
  }
  if (html !== marksDrawn) $('marks').innerHTML = marksDrawn = html;
}

/**
 * Where the pointer is over the canvas (CSS pixels from its corner), whatever it is doing:
 * null once it has left.
 */
let hover: { x: number; y: number } | null = null;

/** Center of the brush last drawn (in fractions of the box), for the guides through it. */
let guideAt: Vec3 | null = null;

/**
 * Volume view: how far beyond the plane through the point the camera orbits the brush's ball
 * lies, along the view direction, in box lengths (the arrow keys, a hundredth of the box a
 * press). Under the pointer still; where it lies in depth is what changes. While it is not 0 a
 * stroke paints at that depth rather than on the gas seen under the pointer. Back to 0 with Home
 * or another tool.
 */
let ballOffset = 0;
let ballOffsetTool: ToolKind | null = null;

/** Distance from the eye, along the view direction, of the brush's ball while it hovers. */
function brushDepth(): number {
  if (tool !== ballOffsetTool) {
    ballOffset = 0;
    ballOffsetTool = tool;
  }
  return Math.max(MIN_DIST * 0.25, view.dist + ballOffset);
}

// ↑ and ↓ move the ball of the tool in hand away and back along the view, a hundredth of the box
// a press (a tenth with Shift); Home puts it back on the plane through the point orbited.
window.addEventListener('keydown', (e) => {
  if (!['ArrowUp', 'ArrowDown', 'Home'].includes(e.key) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (!tool || readsOnly(tool) || tool === 'obstacle' || !cfg || cfg.ndim !== 3 || flatView()) return;
  // (Not where the keys mean something already: a field, a slider, a menu. A button, the tool's
  // own just pressed, has no use for them.)
  const t = e.target as HTMLElement | null;
  if (t && (t.closest('input, select, textarea, [contenteditable]') || t.isContentEditable)) return;
  e.preventDefault();
  brushDepth();
  if (e.key === 'Home') ballOffset = 0;
  else {
    const step = (e.shiftKey ? 0.1 : 0.01) * (e.key === 'ArrowUp' ? 1 : -1);
    ballOffset = Math.max(MIN_DIST * 0.25 - view.dist, ballOffset + step);
  }
  if (hover && !stroke) drawBrush(hover.x, hover.y);
});

/**
 * Volume view: guides along x, y and z through the center of the brush on show, so it can be
 * placed in depth, and the contour levels on its ball. Called once per frame.
 */
function updateGuides() {
  // The brush stays under the pointer while the view moves beneath it (dragged with another
  // button, or zoomed): drawn again where the pointer is now, so the guides go with it.
  if (hover && tool && !readsOnly(tool) && tool !== 'obstacle' && !stroke) drawBrush(hover.x, hover.y);
  const on = !!tool && tool !== 'obstacle' && !readsOnly(tool) && !!cfg && cfg.ndim === 3 && !flatView() && $('brush').style.display !== 'none';
  renderer.guides = on ? guideAt : null;
  // Its ball, with the contour levels drawn on its surface (render/raymarch.wgsl).
  const kind = stroke?.kind ?? tool;
  renderer.ball = on && guideAt && kind ? { pos: [guideAt[0], guideAt[1], guideAt[2]], radius: toolParams[kind].size } : null;
}

/** Outline of the brush under the pointer, and the arrow of a velocity stroke. */
function drawBrush(px?: number, py?: number) {
  const svg = $('brush');
  if (!tool || !cfg || readsOnly(tool)) {
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
  const drag = stroke?.kind === 'velocity' || stroke?.kind === 'source' || stroke?.kind === 'star' || (stroke?.kind === 'sink' && !sinkFixed);
  const x = drag ? stroke!.x0 : (px ?? stroke?.x ?? 0);
  const y = drag ? stroke!.y0 : (py ?? stroke?.y ?? 0);
  const { scale, pos } = locate(x, y, stroke?.ready ? stroke.depth : brushDepth());
  guideAt = pos;
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

/** The scales the tools measure against now: the densest gas and a reference speed. */
function toolRefs(): { rhoMax: number; vRef: number } {
  const d = lastDiag;
  // Fastest gas now. In a flow that is (nearly) at rest that would leave nothing to set, so the
  // reference never falls below a fifth of the sound speed of the hottest gas.
  const cs = d ? Math.sqrt(Math.max(0, cfg.eosParams.gamma * (cfg.eos === 'gas_radiation' ? cfg.eosParams.R : 1) * d.ranges[1][1])) : 0;
  return {
    rhoMax: Math.max(d?.ranges[0][1] ?? 1, cfg.problem.toolScale?.rho ?? 0),
    vRef: Math.max(d?.ranges[4][1] ?? 0, 0.2 * cs, cfg.problem.toolScale?.v ?? 0),
  };
}

function beginStroke(e: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  const [x, y] = [e.clientX - r.left, e.clientY - r.top];
  const s: Stroke = {
    id: e.pointerId, kind: tool!, x0: x, y0: y, x, y, shift: e.shiftKey, depth: brushDepth(), ready: flatView() || ballOffset !== 0,
    released: false, dabbed: false, t0: performance.now(), flowed: 0,
    ...toolRefs(),
  };
  stroke = s;
  // The orbit preview wants the gas as it is at the press.
  if (s.kind === 'sink' || s.kind === 'source') refreshLumps(s.t0, true);
  if (!s.ready) {
    // Volume view: paint on the gas that is seen under the pointer, or, where the view is
    // empty, on the plane through the middle of the box (a ball moved in depth with the arrow
    // keys paints where it is instead).
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
  if (!apply && (stroke.kind === 'velocity' || stroke.kind === 'source' || stroke.kind === 'sink' || stroke.kind === 'star')) stroke = null;
  drawBrush();
}

/**
 * The point mass the sink tool places at `pos` (in fractions of the box), moving at `vel`: what
 * placeMass() adds to the run and what the preview of its orbit is drawn for.
 */
function massSpec(pos: Vec3, vel: Vec3, q: { size: number; amount: number; mass: number; soften: number }, fixed = sinkFixed): PointMass {
  const box = cfg.problem.box;
  // The gas in the box sets the scale; with no readback yet, a box of unit density does.
  const gas = lastDiag?.mass ?? box ** cfg.ndim;
  const two = cfg.ndim < 3;
  // A preset without a gravitational constant gets G = 1 when a point mass is placed.
  const G = cfg.G > 0 ? cfg.G : 1;
  const where = {
    pos: pos.map((v, i) => (two && i === 2 ? 0 : fracPos(cfg, v, i))),
    vel: fixed ? [0, 0, 0] : two ? [vel[0], vel[1], 0] : vel,
    sinkRadius: q.size * box,
    fixed,
  };
  if (!(q.mass > 0)) {
    // Without a mass: a token one, a billionth of the gas, so that it has an orbit to follow
    // and pulls on nothing. Its softening is its size, and its flux is measured as a source's
    // is: against the time the fastest signal takes to cross it.
    return { ...where, gm: 1e-9 * G * gas, soft: q.size * box, sinkRate: (q.amount * (toolRefs().vRef || 1)) / (q.size * box), massless: true };
  }
  return { ...where, gm: G * q.mass * gas, soft: q.soften * box, sinkRate: 4 * q.amount };
}

/** The massless particle a moving source rides: placed at `pos` (in fractions of the box), moving at `vel`. */
/** The star tool's star as a point mass, for the preview of its path: its mass, softened over its radius. */
function starSpec(pos: Vec3, vel: Vec3): PointMass {
  const box = cfg.problem.box;
  return {
    gm: (cfg.G > 0 ? cfg.G : 1) * 10 ** starLogRho * starMassPerRho(),
    pos: pos.map((v, a) => fracPos(cfg, v, a)),
    vel,
    soft: toolParams.star.size * box,
    sinkRadius: 0,
    sinkRate: 0,
  };
}

function carrierSpec(pos: Vec3, vel: Vec3, size: number): PointMass {
  const box = cfg.problem.box;
  const gas = lastDiag?.mass ?? box ** cfg.ndim;
  const two = cfg.ndim < 3;
  return {
    gm: 1e-9 * (cfg.G > 0 ? cfg.G : 1) * gas,
    pos: pos.map((v, i) => (two && i === 2 ? 0 : fracPos(cfg, v, i))),
    vel: two ? [vel[0], vel[1], 0] : vel,
    soft: size * box,
    sinkRadius: 0,
    sinkRate: 0,
    massless: true,
  };
}

/**
 * Where a stroke that is dragged out began (in the box), and the velocity its drag asks for: a
 * longer drag is faster, and a click (or a drag of a few pixels) leaves it at rest, as `still`
 * always does.
 */
function strokeAim(s: Stroke, still = false): { at: ReturnType<typeof locate>; vel: Vec3 } {
  const at = locate(s.x0, s.y0, s.depth);
  const [dx, dy] = [s.x - s.x0, s.y0 - s.y];
  const len = Math.hypot(dx, dy);
  // (A press made before the run's first readout has no reference speed yet: the one there is by now.)
  const speed = still || len < 6 ? 0 : Math.min(1, len / fullDrag()) * 5 * (s.vRef || toolRefs().vRef);
  const vel = [0, 1, 2].map((i) => (speed ? (speed * (dx * at.right[i] + dy * at.up[i])) / len : 0)) as Vec3;
  return { at, vel };
}

/**
 * Place a sink with the sink tool: at `pos` (in fractions of the box), moving at `vel`. With a
 * mass, that is the tool's multiple of the gas now in the box, its sink the tool's radius, its
 * softening the tool's, and its sink's rate the tool's flux (25% is the orbital rate at the
 * edge). Without one it is a massless particle (massSpec).
 */
function placeMass(pos: Vec3, vel: Vec3, q: { size: number; amount: number; mass: number; soften: number }, fixed = sinkFixed): boolean {
  if (!sim) return false;
  const spec = massSpec(pos, vel, q, fixed);
  // Gravity needs a constant: a preset without one gets G = 1, shown in the Gravity section.
  // (A massless particle leaves it without: the gas then does not pull on it.)
  if (!(cfg.G > 0) && !spec.massless) cfg.G = 1;
  sim.cfg = cfg;
  noteSettings();
  const ok = addPlaced(sim, spec, 'sink');
  if (ok) {
    lastDiag = undefined;
    syncControls();
  }
  notice(ok ? '' : `A run holds at most ${MAX_MASSES} sinks and point masses`);
  return ok;
}

/**
 * Place a source with the source tool: at `pos` (in fractions of the box), its gas leaving at
 * `vel`, or, not fixed, riding a massless particle set moving at `vel`. Its flux is the tool's,
 * measured against the densest gas and the time a signal takes to cross it (`refs`).
 */
function placeSource(pos: Vec3, vel: Vec3, q: { size: number; softness: number; amount: number }, fixed: boolean, refs: { rhoMax: number; vRef: number }): boolean {
  if (!sim) return false;
  const perTime = (q.amount * (refs.vRef || 1)) / (q.size * cfg.problem.box);
  const source: SourceSpec = { pos, radius: q.size, softness: q.softness, vel, rate: perTime * refs.rhoMax, decay: 0 };
  if (!fixed) {
    // A source that moves rides a massless particle, which the drag set going: the gas it
    // adds leaves with the particle's velocity, not along the drag.
    const ok = addPlaced(sim, carrierSpec(pos, vel, q.size), 'source');
    notice(ok ? '' : `A run holds at most ${MAX_MASSES} sinks, moving sources and point masses`);
    if (!ok) return false;
    source.vel = [0, 0, 0];
    source.carrier = sim.massCount - 1;
    lastDiag = undefined;
    syncControls();
  }
  sources.push(source);
  // The oldest makes room, with the particle it rode.
  if (sources.length > MAX_SOURCES) {
    const oldest = sources.shift()!;
    if (oldest.carrier !== undefined) removePlaced((_, index) => index === oldest.carrier);
  }
  pushSources(sources);
  syncTools();
  return true;
}

/**
 * Place a polytropic star at `pos` (in fractions of the box), moving at `vel`: of the star
 * tool's index, central density, radius and magnetic field, or of those given (the API's tool()).
 * Switches self-gravity on if it is off: isolated, or periodic in a fully periodic box. The star
 * is then settled into the grid's own balance, as a preset's is (problems/settle.ts), and given
 * its magnetic field, if it has one. Resolves once all of that is done.
 */
function placeStar(pos: Vec3, vel: Vec3, given: Partial<StarAction> = {}): Promise<void> {
  if (!sim || cfg.ndim !== 3) return Promise.resolve();
  starDefaults();
  const n = given.n ?? starN;
  const rhoc = given.rhoc ?? 10 ** starLogRho;
  const size = given.radius ?? toolParams.star.size;
  const field = given.field ?? starField;
  const beta = given.beta ?? 10 ** starLogBeta;
  logAction({ kind: 'star', star: { pos, vel, n, rhoc, radius: size, field, beta } });
  const notes: string[] = [];
  if (!isSelfGravity(cfg.gravity)) {
    // A uniform field's strength is not a gravitational constant.
    if (!(cfg.G > 0) || cfg.gravity === 'constant') cfg.G = 1;
    cfg.gravity = [...cfg.bcLo, ...cfg.bcHi].every((b) => b === PERIODIC) ? 'periodic' : 'isolated';
    notes.push(`self-gravity switched on (${cfg.gravity})`);
  }
  const cells = size * cfg.n;
  if (cells < 8) notes.push(`the star is only ${cells.toFixed(1)} cells in radius`);
  if (cfg.eos === 'ideal' && cfg.eosParams.gamma < 1 + 1 / n) notes.push('\u03b3 < 1 + 1/n: the star is convectively unstable');
  const op = starBrush(n, rhoc, size, cfg.problem.box, cfg.G, pos, vel);
  // Thin gas around a star should neither set the timestep nor ask for refinement: unless the
  // run has its own settings, take them from this star.
  if (!(cfg.fluff.rho > 0)) {
    const radius = size * cfg.problem.box;
    const g = cfg.eosParams.gamma;
    // Twice the central specific internal energy, four times the surface escape speed.
    const uc = (op.star!.K * rhoc ** (1 / n)) / Math.max(g - 1, 0.05);
    const vEsc = Math.sqrt((2 * cfg.G * polytropeMass(laneEmdenCached(n), rhoc, radius)) / radius);
    cfg.fluff = { rho: 1e-5 * rhoc, u: 2 * uc, v: 4 * vEsc };
    notes.push('tenuous-gas caps set from the star (Hydro)');
  }
  if (cfg.amrLevels > 0 && !(cfg.refineRho > 0)) {
    cfg.refineRho = 1e-5 * rhoc;
    notes.push('ambient gas no longer refines (Grid)');
  }
  notice(notes.join('; '));
  const target = sim;
  const box = cfg.problem.box;
  const star = { n, K: op.star!.K, rhoc, center: pos.map((x, a) => fracPos(cfg, x, a)), vel, R: size * box };
  syncControls();
  return enqueue(async () => {
    if (target !== sim) return;
    applyParams();
    target.refreshPotential();
    await target.placeStar(op);
    if (target !== sim) return;
    // Self-gravity has to hear about the new mass.
    const d = await target.diagnostics();
    if (target !== sim) return;
    target.massChanged(d.mass);
    lastDiag = undefined;
    const why = await target.settleStar(star);
    if (target !== sim) return;
    if (why) notice([...notes, `the star is as placed, not settled: ${why}`].join('; '));
    if (field !== 'none') await magnetizeStar(star, field, beta);
  });
}

/**
 * Thread the star just placed with a field, as the polytrope preset's variants are: a dipole
 * along z or a random field, confined to where its pressure is above a hundredth of the
 * center's, with plasma beta `beta` there (mhd/star_fields.ts). A field is part of how a run is
 * built, so the run is built again as a magnetized one, from the state it is in: that state is
 * what Reset then returns to. A field the run already has stays, with the star's added to it.
 * Called from within a queued job.
 */
async function magnetizeStar(star: { n: number; K: number; rhoc: number; center: number[]; R: number }, kind: 'dipole' | 'random', beta: number) {
  const old = sim;
  if (!old) return;
  const le = laneEmdenCached(star.n);
  const g = 1 + 1 / star.n;
  const pc = star.K * star.rhoc ** g;
  const profile: StarProfile = {
    pressure: (r) => (r >= star.R ? 0 : star.K * Math.pow(star.rhoc * Math.pow(le.theta(r / star.R), star.n), g)),
    pc,
    pcut: 0.01 * pc,
    radius: star.R,
  };
  const A = kind === 'dipole' ? dipoleField(profile, beta) : randomField(profile, beta, 1);
  const [cx, cy, cz] = star.center;
  const added: MhdSpec = { potential: () => (x, y, z) => A(x - cx, y - cy, z - cz) };
  setLoadStage('Adding the magnetic field');
  try {
    const snap = await capture(old, { program: 'shaderous', build: BUILD, query: linkQuery() });
    if (old !== sim) return;
    const root = cfg.n >> cfg.amrLevels;
    const from = restartFrom(snap, cfg.problem, cfg.amrLevels > 0, root);
    // The star's field alone, for the grid's faces; what the run had is added face by face.
    const alone: SimConfig = { ...cfg, problem: { ...cfg.problem, mhd: added } };
    const sum = (a: Float32Array, b: Float32Array) => a.map((v, i) => v + b[i]);
    const had = from.problem.mhd;
    const box = cfg.problem.box;
    const levels = cfg.amrLevels;
    const mhd: MhdSpec = had?.faces
      ? { faces: (n, nz) => sum(had.faces!(n, nz), initialFaces(alone, n, nz, box / n)) }
      : had?.block
        ? { block: (level, c) => sum(had.block!(level, c), blockFaces(alone, { level, c } as Block, box / root, levels)) }
        : added;
    const next: SimConfig = { ...cfg, problem: { ...from.problem, mhd }, restart: { ...from.restart!, tool: true } };
    next.tStop = ckpt.on ? nextCheckpoint(next.restart!.time, ckpt.every) : undefined;
    // Sources and the tools' own masses are part of the saved state now, or gone.
    sim = undefined;
    stroke = null;
    draft = null;
    pendingBrush = null;
    sources = [];
    placed = [];
    dropPin();
    old.destroy();
    await gpu.device.queue.onSubmittedWorkDone();
    gpu.device.queue.submit([]);
    await gpu.device.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 250));
    cfg = next;
    renderer.fieldPipelines(fieldShaderFor(cfg), cfg, cfg.amrLevels > 0 && cfg.ndim === 3).catch(() => {});
    const s = await createSim(gpu, cfg, { deferUpdate: true });
    await renderer.attach(s);
    sim = s;
    watchRegrids(s);
    finished = false;
    lastDiag = undefined;
  } finally {
    setLoadStage('');
    if (cfg) syncControls();
  }
}

// --- the editing tools from the API (ui/api.ts tool()), and a replay's actions -----------------

/** The tools tool() takes, by name. */
const TOOL_NAMES = ['mass', 'velocity', 'temperature', 'obstacle', 'field', 'pull', 'source', 'sink', 'star', 'clear'];

/**
 * Apply an editing tool as a person would with the pointer, from settings given in numbers:
 * positions in fractions of the box, velocities in code units, the rest as the tool's own
 * sliders take them (ui/api.ts tool(), docs/api.md). Applied now, and logged.
 */
async function toolApi(name: string, o: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
  const s = sim;
  if (!s || !cfg) throw new Error('no simulation is loaded');
  if (s.preparing) throw new Error(`${name}: the run is still being prepared`);
  const known = (keys: string[]) => {
    for (const k of Object.keys(o)) if (!keys.includes(k)) throw new Error(`${name}: no option "${k}" (it takes ${keys.join(', ')})`);
  };
  const vec = (k: string, zero: number, fallback?: Vec3): Vec3 => {
    const v = o[k];
    if (v === undefined && fallback) return fallback;
    if (!Array.isArray(v) || v.length < 2 || v.length > 3 || !v.every((x) => typeof x === 'number' && Number.isFinite(x))) {
      throw new Error(`${name}: "${k}" takes [x, y] or [x, y, z]`);
    }
    return [v[0], v[1], v.length === 3 ? v[2] : zero];
  };
  const point = (k: string) => vec(k, 0.5);
  const scalar = (k: string, fallback: number, lo = -Infinity, hi = Infinity) => {
    const v = o[k] ?? fallback;
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`${name}: "${k}" takes a number`);
    if (v < lo || v > hi) throw new Error(`${name}: "${k}" must be from ${lo} to ${hi}`);
    return v;
  };
  const flag = (k: string, fallback: boolean) => (o[k] === undefined ? fallback : !!o[k]);
  const magnetic = () => {
    if (!cfg.problem.mhd) throw new Error(`${name}: the run has no magnetic field`);
  };
  toolNotice = '';
  const out: Record<string, unknown> = { tool: name };
  switch (name) {
    case 'mass': {
      known(['pos', 'radius', 'softness', 'amount', 'atRest']);
      // Density added at the center, as a share of the densest gas (negative removes gas).
      const amount = scalar('amount', 0.25, -1, 1) * toolRefs().rhoMax;
      brushNow({ kind: 'mass', pos: point('pos'), radius: scalar('radius', 0.06, 0.001, 1), softness: scalar('softness', 0.5, 0, 1), amount, vel: [0, 0, 0], atRest: flag('atRest', true) });
      if (isSelfGravity(cfg.gravity) && cfg.ndim === 3) await tellMass(s);
      break;
    }
    case 'velocity':
      known(['pos', 'radius', 'softness', 'vel']);
      brushNow({ kind: 'velocity', pos: point('pos'), radius: scalar('radius', 0.08, 0.001, 1), softness: scalar('softness', 0.5, 0, 1), amount: 0, vel: vec('vel', 0) });
      break;
    case 'temperature':
      known(['pos', 'radius', 'softness', 'factor']);
      if (!eosHasEnergy(cfg.eos)) throw new Error('temperature: this equation of state ties the temperature to the density');
      brushNow({ kind: 'temperature', pos: point('pos'), radius: scalar('radius', 0.06, 0.001, 1), softness: scalar('softness', 0.5, 0, 1), amount: Math.log(scalar('factor', 1.5, 1e-6, 1e6)), vel: [0, 0, 0] });
      break;
    case 'obstacle': {
      known(['shape', 'center', 'half', 'points', 'axis', 'erase']);
      if (cfg.problem.mhd) throw new Error('obstacle: obstacles are not available with a magnetic field yet');
      const axis = cfg.ndim < 3 ? 2 : scalar('axis', 2, 0, 2);
      if (!Number.isInteger(axis)) throw new Error('obstacle: "axis" is 0, 1 or 2');
      const pair = (k: string): [number, number] => {
        const v = o[k];
        if (!Array.isArray(v) || v.length !== 2 || !v.every((x) => typeof x === 'number' && Number.isFinite(x))) throw new Error(`obstacle: "${k}" takes [a, b]`);
        return [v[0], v[1]];
      };
      let shape: PlaneShape;
      if (o.shape === 'rect' || o.shape === 'ellipse') shape = { shape: o.shape, center: pair('center'), half: pair('half') };
      else if (o.shape === 'polygon') {
        const pts = o.points;
        if (!Array.isArray(pts) || pts.length < 3 || pts.length > MAX_OUTLINE_VERTICES) throw new Error(`obstacle: "points" takes 3 to ${MAX_OUTLINE_VERTICES} corners [a, b]`);
        const corner = (q: unknown): [number, number] => {
          if (!Array.isArray(q) || q.length !== 2 || !q.every((x) => typeof x === 'number' && Number.isFinite(x))) throw new Error('obstacle: each of "points" is [a, b]');
          return [q[0], q[1]];
        };
        shape = { shape: 'polygon', points: pts.map(corner) };
      } else throw new Error('obstacle: "shape" is rect, ellipse or polygon');
      brushNow({ kind: 'obstacle', pos: [0, 0, 0], radius: 1, softness: 0, vel: [0, 0, 0], amount: flag('erase', false) ? -1 : 1, outline: planeOutline(shape, axis as 0 | 1 | 2) });
      break;
    }
    case 'field':
      known(['pos', 'radius', 'softness', 'factor']);
      magnetic();
      brushNow({ kind: 'field', pos: point('pos'), radius: scalar('radius', 0.08, 0.001, 1), softness: scalar('softness', 0.5, 0, 1), amount: Math.log(scalar('factor', 1.5, 1e-3, 1e3)), vel: [0, 0, 0] });
      break;
    case 'pull': {
      known(['from', 'to', 'radius', 'softness', 'grip']);
      magnetic();
      const ops = pullOps(point('from'), point('to'), { radius: scalar('radius', 0.1, 0.001, 1), softness: scalar('softness', 0.7, 0, 1), grip: scalar('grip', 0.5, 0.01, 0.9) }, cfg.n, cfg.ndim < 3);
      for (const op of ops) brushNow(op);
      out.frames = ops.length;
      break;
    }
    case 'sink': {
      known(['pos', 'vel', 'radius', 'rate', 'mass', 'soften', 'fixed']);
      const ok = placeMass(point('pos'), vec('vel', 0, [0, 0, 0]), { size: scalar('radius', 0.04, 0.001, 1), amount: scalar('rate', 0.25, 0, 1), mass: scalar('mass', 0, 0), soften: scalar('soften', 0.04, 0.001, 1) }, flag('fixed', false));
      if (!ok) throw new Error(`sink: ${toolNotice || 'it could not be placed'}`);
      out.index = s.massCount - 1;
      break;
    }
    case 'source': {
      known(['pos', 'vel', 'radius', 'softness', 'rate', 'fixed']);
      const ok = placeSource(point('pos'), vec('vel', 0, [0, 0, 0]), { size: scalar('radius', 0.04, 0.001, 1), softness: scalar('softness', 0.5, 0, 1), amount: scalar('rate', 0.25, 0, 1) }, flag('fixed', false), toolRefs());
      if (!ok) throw new Error(`source: ${toolNotice || 'it could not be placed'}`);
      break;
    }
    case 'star': {
      known(['pos', 'vel', 'n', 'rhoc', 'radius', 'field', 'beta']);
      if (cfg.ndim !== 3) throw new Error('star: a star needs self-gravity, which a 2D run does not have');
      if (!eosHasEnergy(cfg.eos)) throw new Error('star: this equation of state fixes the pressure of the gas, so a star\u2019s cannot be chosen');
      const field = o.field ?? 'none';
      if (field !== 'none' && field !== 'dipole' && field !== 'random') throw new Error('star: "field" is none, dipole or random');
      await placeStar(point('pos'), vec('vel', 0, [0, 0, 0]), { n: scalar('n', starN, 0.5, 4.5), rhoc: scalar('rhoc', 10 ** starLogRho, 1e-30), radius: scalar('radius', toolParams.star.size, 0.005, 0.5), field, beta: scalar('beta', 10 ** starLogBeta, 1) });
      await idle();
      break;
    }
    case 'clear': {
      known(['what']);
      if (o.what !== 'sources' && o.what !== 'sinks') throw new Error('clear: "what" is sources or sinks');
      clearPlaced(o.what === 'sinks');
      break;
    }
    default:
      throw new Error(`no tool "${name}": the tools are ${TOOL_NAMES.join(', ')}`);
  }
  syncTools();
  if (toolNotice) out.notice = toolNotice;
  return out;
}

/** Apply an action of a session log, as a replay does (ui/api.ts replay()). */
async function replayAction(a: Action) {
  const s = sim;
  if (!s) throw new Error('no simulation is loaded');
  switch (a.kind) {
    case 'brush':
      brushNow(a.op);
      break;
    case 'mass':
      if (!addPlaced(s, a.spec, a.role)) throw new Error('the run holds no more point masses than it did');
      lastDiag = undefined;
      syncControls();
      break;
    case 'remove':
      removePlaced((_, index) => a.indices.includes(index));
      break;
    case 'fix':
      s.setTracerFixed(a.index, a.fixed);
      logAction(a);
      break;
    case 'sources':
      sources = a.list.map((q) => ({ ...q }));
      pushSources(sources);
      break;
    case 'massChanged':
      s.massChanged(a.mass);
      logAction(a);
      break;
    case 'star':
      await placeStar(a.star.pos, a.star.vel, a.star);
      await idle();
      break;
    case 'set':
      await applySpec(a.spec);
      break;
    case 'regrid':
      if (a.steps) s.replayRegrid?.(a.steps);
      else if (a.indicators) s.forceRegrid?.(base64ToF32(a.indicators));
      break;
    case 'bins':
      s.replayBins?.(a.pause);
      logAction(a);
      break;
    case 'clock':
      // The end time is a setting (logged as one): what differs from it is a stop of the clock's own.
      cfg.tStop = a.stop !== null && !(a.stop >= cfg.tEnd) ? a.stop : undefined;
      cfg.tFrame = a.frame ?? undefined;
      s.cfg = cfg;
      s.writeParams();
      break;
  }
}

// --- orbit preview --------------------------------------------------------------------------
// While the sink tool or the source tool is in hand (and not set to Fixed), or the star tool,
// the path a sink, source or star placed now would take is drawn (a star's is that of a point
// of its mass, softened over its radius): a line
// with an arrowhead over a 2D run or a slice, a tube with one in the volume view. Under the
// pointer it is the path of a mass let go at rest there; pressed and dragged, of one set moving
// as the drag asks. It is the orbit in the gas as it is now (gravity/preview.ts).

/** The gas as lumps, read back a few times a second while the sink tool holds a mass. */
let gasLumps: { of: Simulation; list: Lump[]; version: number } | null = null;
let lumpsAsked = 0;
let lumpsVersion = 0;
/** The path on show, and what it was found for. */
let orbit: { key: string; path: Orbit } | null = null;
let orbitVersion = 0;
/** What the SVG path was last drawn for. */
let orbitDrawn = '';

/** Ask for the gas's lumps again if the last ones are a quarter of a second old (or `force`). */
function refreshLumps(now: number, force = false) {
  if (!sim || (!force && now - lumpsAsked < 250)) return;
  lumpsAsked = now;
  const of = sim;
  const { box } = cfg.problem;
  const ndim = cfg.ndim;
  of.readGasLumps().then(
    (list) => {
      if (list && of === sim) gasLumps = { of, list: fitLumps(list, box, ndim), version: ++lumpsVersion };
    },
    () => {},
  );
}

/** Find the path of the mass being placed, and hand it to whichever view draws it. Called once per frame. */
function updateOrbit(now: number) {
  const s = stroke;
  // A star moves as a whole as a point of its mass would (3D: it needs self-gravity).
  const star = tool === 'star' && !!cfg && cfg.ndim === 3;
  const moves = toolMoves() || star;
  if (sim && moves) refreshLumps(now);
  // Pressed: the mass the drag would place. Not pressed: one let go at rest under the pointer,
  // where the brush is drawn (in the volume view, on the plane through the middle of the box).
  const pressed = !!s && s.kind === tool && s.ready && !s.released;
  const resting = !s && !!hover && $('brush').style.display !== 'none';
  // (A fixed sink or source goes nowhere: there is no path to show.)
  if (!sim || !moves || !(pressed || resting) || (!star && massesNow().length >= MAX_MASSES)) {
    orbit = null;
  } else {
    const { at, vel } = pressed ? strokeAim(s!) : { at: locate(hover!.x, hover!.y, brushDepth()), vel: [0, 0, 0] as Vec3 };
    const mine = star ? starSpec(at.pos, vel) : tool === 'source' ? carrierSpec(at.pos, vel, toolParams.source.size) : massSpec(at.pos, vel, toolParams.sink);
    const others = massesNow();
    const lumps = gasLumps && gasLumps.of === sim ? gasLumps : null;
    const left = cfg.tEnd - (lastDiag?.t ?? 0);
    const key = JSON.stringify([mine.gm, mine.pos, mine.vel, mine.soft, lumps?.version, others.map((m) => [m.gm, m.pos, m.vel]), left]);
    if (orbit?.key !== key) {
      const frame = cfg.problem.frame === 'gas' ? 1 : cfg.problem.frame === 'masses' ? 2 : 0;
      const path = previewOrbit(
        // (Placing a point mass gives a run without a gravitational constant G = 1; a massless
        // particle leaves it without, and the gas then does not pull on it.)
        { bodies: [...others, { ...mine, soft: mine.soft! }], lumps: lumps?.list ?? [], G: cfg.G > 0 ? cfg.G : mine.massless ? 0 : 1, frame, box: cfg.problem.box },
        { time: left > 0 ? left : Infinity },
      );
      orbit = { key, path };
      orbitVersion++;
    }
  }
  const volume = !!cfg && cfg.ndim === 3 && !flatView();
  renderer.orbit = orbit && volume ? { points: orbit.path.points, others: orbit.path.others, box: cfg.problem.box, version: orbitVersion } : null;
  drawOrbit(volume ? null : (orbit?.path ?? null));
}

/** The path as a line with an arrowhead, over a 2D run or a slice; the stretches off a slice's plane are dashed. */
function drawOrbit(path: Orbit | null) {
  const svg = $('brush');
  const n = path ? path.points.length / 3 : 0;
  // The drag's own arrow steps back while the path is on show (in the volume view too).
  svg.classList.toggle('aimed', !!orbit && orbit.path.points.length > 3);
  const d = ['', '', '', ''];
  if (path && n > 1) {
    const at = projector();
    const box = cfg.problem.box;
    const f = (v: number) => v.toFixed(1);
    const project = (pts: Float32Array) => Array.from({ length: pts.length / 3 }, (_, i) => at([0, 1, 2].map((c) => boxFrac(cfg, pts[3 * i + c], c))));
    const px = project(path.points);
    // Where the run's other masses go meanwhile, faintly: what the path is bending around.
    for (const other of path.others) {
      const q = project(other);
      if (q.length > 1) d[3] += `M${q.map((c) => `${f(c.x)} ${f(c.y)}`).join('L')}`;
    }
    // A slice of a 3D run shows one layer of cells: the path is drawn where it is over that
    // plane, dashed where it has left the layer (by more than the half cell a mass placed in
    // the middle of the layer swings across the plane of a star centered between two layers).
    const slice = cfg.ndim === 3 && view.mode === 'slice';
    const plane = slice ? locate(0, 0, 0).pos[view.sliceAxis] : 0;
    const off = (i: number) => slice && Math.abs(boxFrac(cfg, path.points[3 * i + view.sliceAxis], view.sliceAxis) - plane) > 1.5 / cfg.n;
    for (let i = 1; i < n; i++) d[off(i) ? 1 : 0] += `M${f(px[i - 1].x)} ${f(px[i - 1].y)}L${f(px[i].x)} ${f(px[i].y)}`;
    // The arrowhead: at the end, along the last stretch long enough to give a direction.
    const tip = px[n - 1];
    let k = n - 2;
    while (k > 0 && Math.hypot(tip.x - px[k].x, tip.y - px[k].y) < 4) k--;
    const len = Math.hypot(tip.x - px[k].x, tip.y - px[k].y);
    if (len > 0) {
      const [ux, uy] = [(tip.x - px[k].x) / len, (tip.y - px[k].y) / len];
      d[2] = `M${f(tip.x + 7 * ux)} ${f(tip.y + 7 * uy)}L${f(tip.x - 2 * ux - 5 * uy)} ${f(tip.y - 2 * uy + 5 * ux)}L${f(tip.x - 2 * ux + 5 * uy)} ${f(tip.y - 2 * uy - 5 * ux)}z`;
    }
  }
  const all = d.join('|');
  if (all === orbitDrawn) return;
  orbitDrawn = all;
  svg.querySelector('path.orbit:not(.off)')!.setAttribute('d', d[0]);
  svg.querySelector('path.orbit.off')!.setAttribute('d', d[1]);
  svg.querySelector('path.orbit-head')!.setAttribute('d', d[2]);
  svg.querySelector('path.orbit-others')!.setAttribute('d', d[3]);
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
    // A drag aims a source, or sets a sink moving; a click (or a drag of a few pixels) leaves
    // either at rest, as a fixed sink always is.
    const { at, vel } = strokeAim(s, sink && sinkFixed);
    if (sink) {
      placeMass(at.pos, vel, q);
      syncTools();
      return;
    }
    // The flux is measured against the densest gas and the time a signal takes to cross the source.
    placeSource(at.pos, vel, q, sourceFixed, { rhoMax: s.rhoMax || toolRefs().rhoMax, vRef: s.vRef || toolRefs().vRef });
    return;
  }
  if (s.kind === 'star') {
    if (!s.released) return;
    stroke = null;
    // As for a point mass: a drag sets the star moving, a click leaves it at rest.
    const { at, vel } = strokeAim(s);
    placeStar(at.pos, vel);
    return;
  }
  if (s.kind === 'query' || s.kind === 'profile') return; // the tools that read never start a stroke
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
  if (s.kind === 'pull') {
    // The field is carried after the pointer, a frame's worth at a time (ui/session.ts pullStep).
    const to = locate(s.x, s.y, s.depth).pos;
    const from = (s.pulled ??= locate(s.x0, s.y0, s.depth).pos);
    const step = pullStep(from, to, { radius: q.size, softness: q.softness, grip: q.amount }, cfg.n, cfg.ndim < 3);
    if (s.released && step.done) stroke = null;
    if (!step.op) return;
    pendingBrush = step.op;
    s.pulled = step.at;
    return;
  }
  // Mass, temperature and field strength: one full application on the press, then, if the press is held for
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
      setTimeout(() => tellMass(target).catch(fail), 0);
    }
  }
  if (strength === 0) return;
  const sign = s.shift ? -1 : 1;
  const amount = sign * strength * (s.kind === 'mass' ? q.amount * s.rhoMax : Math.log(1.5));
  pendingBrush = { ...base, pos: locate(s.x, s.y, s.depth).pos, amount, atRest: massAtRest };
}

/**
 * The field menu's Composition group: a mass fraction for each species the run carries, as
 * values "19:i" (field, species). Rebuilt when the species change; a view of a species the
 * run does not carry falls back to the density.
 */
function syncSpeciesFields(sel: HTMLSelectElement) {
  // The magnetic fields, only for a run that has a field.
  const mhd = cfg.problem.mhd !== undefined;
  const mag = [...sel.querySelectorAll('optgroup')].find((g) => g.label === 'Magnetic field');
  if (mag) mag.hidden = !mhd;
  if (!mhd && MAGNETIC_FIELDS.includes(view.field)) view.field = 0;
  // The nuclear heating, only for a run that burns.
  const burns = !!burnSpec(cfg);
  const heat = sel.querySelector<HTMLOptionElement>(`option[value="${NUCLEAR_HEATING_FIELD}"]`);
  if (heat) heat.hidden = !burns;
  if (!burns && view.field === NUCLEAR_HEATING_FIELD) view.field = 0;
  // The radiation fields, only for a run with radiation transport.
  const rad = cfg.radiation === 'm1';
  const radGroup = [...sel.querySelectorAll('optgroup')].find((g) => g.label === 'Radiation');
  if (radGroup) radGroup.hidden = !rad;
  if (!rad && RADIATION_FIELDS.includes(view.field)) view.field = 0;
  const species = cfg.problem.species ?? [];
  const key = species.join(',');
  let group = sel.querySelector<HTMLOptGroupElement>('optgroup[data-species]');
  if ((group?.dataset.species ?? '') !== key) {
    group?.remove();
    group = null;
    if (species.length) {
      group = document.createElement('optgroup');
      group.label = 'Composition';
      group.dataset.species = key;
      species.forEach((name, i) => {
        const o = document.createElement('option');
        o.value = `${MASS_FRACTION_FIELD}:${i}`;
        o.textContent = `X(${speciesLabel(name)})`;
        o.title = `${FIELD_TIPS[MASS_FRACTION_FIELD]}: ${speciesLabel(name)}`;
        group!.append(o);
      });
      sel.append(group);
    }
  }
  if (view.field === MASS_FRACTION_FIELD && !(view.species < species.length)) {
    if (species.length) view.species = 0;
    else view.field = 0;
  }
}

/** Displayed range from the latest diagnostics, unless the user typed one. */
function updateRange(d: Diagnostics) {
  if (!$<HTMLInputElement>('auto').checked) return;
  // The first fields are ranged by the diagnostics reduction; the rest by the display pass,
  // over what it last drew.
  // A projection is always ranged over the columns it drew.
  const r = renderer.range;
  const key = renderer.rangeKey(view);
  // (A view that holds one field has no state for the diagnostics to range: what it drew does.)
  const drawn = projecting() || view.field >= d.ranges.length || !!cfg.viewBaked;
  if (drawn && (!r || r.key !== key)) return;
  let [lo, hi] = drawn ? [r!.lo, r!.hi] : d.ranges[view.field];
  if (drawn) {
    // A few extreme cells (inside a sink, at the corner of an obstacle) would set the whole
    // range: the top is the 99.5th percentile of |value| when that is lower.
    const top = renderer.quantile(key, 0.995);
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
  showRange();
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

// [label, tooltip, CSV column, a count]; DIAG_SIDE puts each in the left (0) or right column.
// BURN_COL is burning's, shown only for a run that burns; LIQUID_COL a liquid's, shown only for
// one (eos/liquid.wgsl); RAD_ROWS from RAD_COL are radiation transport's, shown only with it
// (the first is the light curve); the last MAG_ROWS are a magnetic field's, shown only with one.
const DIAG_ROWS: [string, string, string, boolean?][] = [
  ['t', 'Time', 't'], ['dt', 'Timestep', 'dt'], ['steps', 'Steps taken', 'steps', true], ['mass', 'Total mass', 'mass'],
  ['Mach', 'Maximum Mach number', 'max_mach'], ['floor', 'Cells at the density or pressure floor (emptied voids, usually)', 'floored_cells', true],
  ['Jeans', 'Cells whose thermal Jeans length is under 4 cells. With the Jeans floor on these are the cells it is holding up', 'jeans_unresolved_cells', true],
  ['p x', 'Total x momentum', 'p_x'], ['p y', 'Total y momentum', 'p_y'], ['p z', 'Total z momentum', 'p_z'],
  ['E kin', 'Kinetic energy', 'e_kin'], ['E int', 'Internal energy', 'e_int'], ['E grav', 'Gravitational energy', 'e_grav'], ['E tot', 'Total energy (with a magnetic field, its energy included)', 'e_tot'],
  ['L nuc', 'Nuclear heating: the energy burning released per unit time over the last step, summed over the gas', 'l_nuc'],
  ['δρ max', 'A liquid: the largest departure of its density from the reference, |ρ / ρ₀ − 1|. It is about the Mach number squared, and is the error of treating the flow as incompressible: raise the sound speed to lower it', 'rho_deviation'],
  ['L rad', 'Luminosity: the radiation energy leaving through the open faces of the box per unit time, averaged over the time since the readout before (summed step by step, so nothing between readouts is missed). Its history is the bolometric light curve, as seen at the box’s edge: light takes the box’s size over the reduced speed of light to get there', 'l_rad'],
  ['E rad', 'Radiation energy in the box', 'e_rad'],
  ['E esc', 'Radiation energy that has left through the open faces of the box since the run started', 'e_escaped'],
  ['T rad', 'Largest radiation temperature, (E / a)^(1/4), of any cell', 't_rad_max'],
  ['ĉ slow', 'Cells where the gas’s speed plus its sound speed is over a third of the reduced speed of light: raise the reduced speed of light if this is not zero', 'chat_slow_cells', true],
  ['E mag', 'Magnetic energy, the sum of B² / 2 over the cells (code units: B_Gauss = √(4π) B)', 'e_mag'],
  ['|B| max', 'Largest field strength at a cell center (code units: B_Gauss = √(4π) B)', 'b_max'],
  ['β min', 'Least plasma beta, 2p / B², of any cell: below 1 the field’s pressure dominates the gas’s', 'beta_min'],
  ['∇·B', 'Largest |div B| dx of any cell, over the largest |B|: constrained transport keeps it at float32 round-off (about 1e-7)', 'div_b'],
  ['α M', 'Maxwell α: a disk’s Maxwell stress −B_R B_φ summed over the disk between the sink and the damping zone, over its summed pressure: the Shakura-Sunyaev α the field carries. MRI turbulence gives some 0.01 to 0.1', 'alpha_maxwell'],
  ['α R', 'Reynolds α: as the Maxwell α, for the Reynolds stress ρ v_R δv_φ, with δv_φ the orbital speed less the softened Keplerian one: typically a quarter to a half of the Maxwell stress in MRI turbulence', 'alpha_reynolds'],
  ['Q z', 'Mass-weighted mean over the disk of the MRI quality factor 2π |B_z| / (√ρ Ω dx): cells across the fastest-growing wavelength. Above about 8 the MRI grows at its proper rate (Hawley, Guan & Krolik 2011)', 'q_z'],
  ['Q φ', 'As Q z, for the field along the orbit. Above about 20 the turbulence is resolved', 'q_phi'],
];
/** A disk's rows (Problem.disk), the last of the field's. */
const DISK_ROWS = 4;
const MAG_ROWS = 4 + DISK_ROWS;
/** Burning's row (nuclear heating). */
const BURN_COL = 14;
/** A liquid's row (its largest density departure). */
const LIQUID_COL = 15;
/** Radiation transport's rows: luminosity, energy, energy escaped, largest temperature, cells too fast for chat. */
const RAD_COL = 16;
const RAD_ROWS = 5;
const radCol = (c: number) => c >= RAD_COL && c < RAD_COL + RAD_ROWS;
// Seven a side; burning's row left; a field's four rows two a side (|B| max, beta min left;
// E mag, div B right), and a disk's likewise (the alphas left, the Qs right).
// Radiation's five: luminosity, temperature and the warning left; the two energies right.
const DIAG_SIDE = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1];
/** What each diagnostic measures, in DIAG_ROWS order (the field's in code units). */
const DIAG_MEASURES: Measure[] = ['time', 'time', null, 'mass', null, null, null, 'momentum', 'momentum', 'momentum', 'energy', 'energy', 'energy', 'energy', 'power', null, 'power', 'energy', 'energy', 'temperature', null, 'energy', null, null, null, null, null, null, null];
/** A liquid's largest |rho / rho0 - 1|, from the density's range (NaN for any other gas). */
const liquidDeviation = (d: Diagnostics): number => {
  if (cfg?.eos !== 'liquid') return NaN;
  const r0 = cfg.eosParams.liqRho;
  return Math.max(Math.abs(d.ranges[0][0] / r0 - 1), Math.abs(d.ranges[0][1] / r0 - 1));
};
/** The readout before, for the luminosity: the time of its escaped total, the total, and the luminosity shown. */
let radPrev: { at: number; escaped: number; lum: number } | null = null;
/**
 * Radiation's rows (RAD_ROWS of them; NaN without it). The luminosity is the energy that left
 * since the readout before over the time between the two totals (their own times: a readout
 * made while the run goes on reads the clock some steps before the total, so over the clock's
 * interval the last readout before a pause gave a pause's the luminosity 0, and the first after
 * it a spike). The first readout's, and a run started again, is the readout's own; one that
 * repeats a time keeps the one before.
 */
function radValues(d: Diagnostics): number[] {
  const r = d.radiation;
  if (!r) return new Array<number>(RAD_ROWS).fill(NaN);
  let lum = r.luminosity;
  if (radPrev && r.escapedAt > radPrev.at) lum = (r.escaped - radPrev.escaped) / (r.escapedAt - radPrev.at);
  else if (radPrev && r.escapedAt === radPrev.at) lum = radPrev.lum; // paused: the readout repeats
  radPrev = { at: r.escapedAt, escaped: r.escaped, lum };
  return [lum, r.eRad, r.escaped, r.maxTrad, r.tooFast];
}
/** The diagnostics in DIAG_ROWS order. */
const diagValues = (d: Diagnostics): number[] => {
  const m = d.magnetic;
  const k = m?.disk;
  return [
    d.t, d.dt, d.steps, d.mass, d.maxMach, d.floored, d.jeansUnresolved, ...d.momentum, d.eKin, d.eInt, d.eGrav, d.eTot, d.nuclearHeating, liquidDeviation(d),
    ...radValues(d),
    ...(m ? [m.eMag, m.bMax, m.betaMin, m.divB] : new Array<number>(MAG_ROWS - DISK_ROWS).fill(NaN)),
    ...(k ? [k.alphaMaxwell, k.alphaReynolds, k.qz, k.qphi] : new Array<number>(DISK_ROWS).fill(NaN)),
  ];
};
/** Whether column c is one of a magnetic field's. */
const magneticCol = (c: number) => c >= DIAG_ROWS.length - MAG_ROWS && c < DIAG_ROWS.length;
/** Whether column c is one of a disk's (the last of a field's). */
const diskCol = (c: number) => c >= DIAG_ROWS.length - DISK_ROWS && c < DIAG_ROWS.length;

/** One column of the diagnostics history. */
interface DiagCol {
  label: string;
  title: string;
  /** CSV column name, or null for a column only the panel shows */
  csv: string | null;
  measure: Measure;
  count?: boolean;
}
const MASS_RATE: Measure = [1, 0, -1];
const ANGULAR_MOMENTUM: Measure = [1, 2, -1];
const FORCE: Measure = [1, 1, -2];
const TORQUE: Measure = [1, 2, -2];
/** One sink field: [CSV suffix (null: panel only), panel label, tooltip, measure]. */
type SinkField = [string | null, string, string, Measure];
const xyz = (name: string, title: string, measure: Measure): SinkField[] =>
  (['x', 'y', 'z'] as const).map((a) => [`${name}_${a}`, '', `${title}, ${a}`, measure]);
/**
 * Each sink's columns, after the diagnostics above, one set per point mass that has been in the
 * run. The first SINK_SHOWN are shown in the panel. The rates are averaged over the time since
 * the readout before.
 */
const SINK_FIELDS: SinkField[] = [
  ['mass', 'M', 'Mass, with the gas it has taken in', 'mass'],
  ['mdot', 'Ṁ', 'Accretion rate', MASS_RATE],
  ['accreted', 'ΔM', 'Gas mass taken in so far', 'mass'],
  [null, 'Δp', 'Momentum the accreted gas brought in (magnitude)', 'momentum'],
  [null, 'ΔL', '', ANGULAR_MOMENTUM], // its title depends on the dimension: see colTitle
  ...xyz('p_acc', 'Momentum the accreted gas brought in', 'momentum'),
  ...xyz('l_acc', 'Angular momentum the accreted gas brought in, about the sink in its rest frame', ANGULAR_MOMENTUM),
  ...xyz('f_acc', 'Rate the accreted gas brings momentum in', FORCE),
  ...xyz('torque_acc', 'Rate the accreted gas brings angular momentum in', TORQUE),
  ...xyz('pos', 'Position from the box center', 'length'),
  ...xyz('vel', 'Velocity', 'velocity'),
];
/** Sink fields the panel shows. */
const SINK_SHOWN = 5;
/**
 * Each body's columns (Diagnostics.bodies: forces on the setup's obstacles, hydro/forces.ts), one
 * set per body, after the diagnostics above and before the sinks'. The coefficients need a
 * reference (a tunnel's): without one they are blank.
 */
const DIMENSIONLESS: Measure = [0, 0, 0];
const BODY_FIELDS: SinkField[] = [
  ['cd', 'C_d', 'Drag coefficient: the force along the stream over rho U^2 A / 2', DIMENSIONLESS],
  ['cl', 'C_l', 'Lift coefficient: the force along the lift axis over rho U^2 A / 2', DIMENSIONLESS],
  ['cp', 'C_p', 'Power coefficient, in a rotor\u2019s frame: Omega times the torque about the axis over rho U^3 pi R^2 / 2 (positive for a turbine; the Betz limit is 0.593)', DIMENSIONLESS],
  ['fx', 'F_x', 'Force the gas exerts on the body, x', FORCE],
  ['fy', 'F_y', 'Force the gas exerts on the body, y', FORCE],
  ['fz', 'F_z', 'Force the gas exerts on the body, z', FORCE],
  ['torque', 'τ', 'Torque the gas exerts on the body about its point (z in 2D; magnitude in 3D)', TORQUE],
  ...xyz('f_visc', 'Viscosity\u2019s part of the force', FORCE),
  ...xyz('torque', 'Torque about the body\u2019s point', TORQUE),
];
/** Body fields the panel shows. */
const BODY_SHOWN = 7;
const BASE_COLS = DIAG_ROWS.length;
const bodyCol = (i: number, field: number) => BASE_COLS + BODY_FIELDS.length * i + field;
const SINK_BASE = BASE_COLS + BODY_FIELDS.length * MAX_BODIES;
const DIAG_COLS: DiagCol[] = [
  ...DIAG_ROWS.map(([label, title, csv, count], i) => ({ label, title, csv, measure: DIAG_MEASURES[i], count })),
  ...Array.from({ length: MAX_BODIES }, (_, i) =>
    BODY_FIELDS.map(([csv, label, title, measure]) => ({ label: label || (csv ?? ''), title, csv: csv && `body${i + 1}_${csv}`, measure })),
  ).flat(),
  ...Array.from({ length: MAX_MASSES }, (_, i) =>
    SINK_FIELDS.map(([csv, label, title, measure]) => ({ label: label || (csv ?? ''), title, csv: csv && `sink${i + 1}_${csv}`, measure })),
  ).flat(),
];
const sinkCol = (i: number, field: number) => SINK_BASE + SINK_FIELDS.length * i + field;
/** The tooltip of column `c`. */
function colTitle(c: number): string {
  if (c < BASE_COLS) return DIAG_COLS[c].title;
  if (c < SINK_BASE) {
    const i = Math.floor((c - BASE_COLS) / BODY_FIELDS.length);
    return `${cfg?.problem.bodies?.[i]?.id ?? `Body ${i + 1}`}: ${DIAG_COLS[c].title}`;
  }
  const i = Math.floor((c - SINK_BASE) / SINK_FIELDS.length);
  const field = (c - SINK_BASE) % SINK_FIELDS.length;
  const title =
    field !== 4
      ? DIAG_COLS[c].title
      : cfg.ndim < 3
        ? 'Angular momentum the accreted gas brought in, about the sink in its rest frame (z; positive is counterclockwise)'
        : 'Angular momentum the accreted gas brought in, about the sink in its rest frame (magnitude)';
  return `Sink ${i + 1}: ${title}`;
}

/** What each sink had at the readout before, for the rates. */
let sinkPrev: { t: number; acc: number; p: number[]; l: number[]; rates: number[] }[] = [];
const norm = (v: number[]) => Math.hypot(v[0], v[1], v[2]);
/** The sink columns of a readout: SINK_FIELDS.length per mass. */
function sinkValues(d: Diagnostics): number[] {
  const out: number[] = [];
  d.masses.forEach((m, i) => {
    const prev = sinkPrev[i];
    const p = m.accretedMomentum;
    const l = m.accretedSpin;
    let rates = new Array<number>(7).fill(NaN);
    if (prev && d.t > prev.t) {
      const h = d.t - prev.t;
      rates = [(m.accreted - prev.acc) / h, ...p.map((v, k) => (v - prev.p[k]) / h), ...l.map((v, k) => (v - prev.l[k]) / h)];
    } else if (prev && d.t === prev.t) rates = prev.rates; // paused: the readout repeats
    sinkPrev[i] = { t: d.t, acc: m.accreted, p, l, rates };
    const mass = m.massless ? 0 : cfg.G > 0 ? m.gm / cfg.G : NaN;
    const lShown = cfg.ndim < 3 ? l[2] : norm(l);
    out.push(mass, rates[0], m.accreted, norm(p), lShown, ...p, ...l, ...rates.slice(1), ...m.pos, ...m.vel);
  });
  return out;
}

/** The body columns of a readout: BODY_FIELDS.length per body, MAX_BODIES of them (NaN for none). */
function bodyValues(d: Diagnostics): number[] {
  const out: number[] = [];
  for (let i = 0; i < MAX_BODIES; i++) {
    const b = d.bodies?.[i];
    if (!b) {
      out.push(...new Array<number>(BODY_FIELDS.length).fill(NaN));
      continue;
    }
    const k = b.coefficients;
    out.push(k?.drag ?? NaN, k?.lift ?? NaN, k?.power ?? NaN, ...b.force, cfg.ndim < 3 ? b.torque[2] : norm(b.torque), ...b.viscous, ...b.torque);
  }
  return out;
}

const diagHistory = new DiagHistory(2);
/** The run the history belongs to. */
let diagOf: Simulation | undefined;
const diagCols = $('diag').querySelectorAll('dl');
const sinkPanel = $('diag-sinks');
const diagPop = $('diag-pop');
const diagPopCanvas = diagPop.querySelector('canvas')!;
const diagPopTitle = diagPop.querySelector('.title')!;
/** Column whose whole-run plot is showing, if any. */
let diagHover = -1;
interface DiagCell { col: number; dt: HTMLElement; dd: HTMLElement; spark: HTMLCanvasElement; set: (v: string) => void; warn: boolean }
function diagCell(col: number, label: string, into: Element): DiagCell {
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  const spark = document.createElement('canvas');
  spark.className = 'spark';
  into.append(dt, dd, spark);
  for (const el of [dd, spark]) {
    el.addEventListener('pointerenter', () => showDiagPlot(col, dd));
    el.addEventListener('pointerleave', () => hideDiagPlot());
  }
  return { col, dt, dd, spark, set: liveText(dd), warn: false };
}
const diagCells = DIAG_ROWS.map(([label], i) => diagCell(i, label, diagCols[DIAG_SIDE[i]]));
/** Show the magnetic field's rows with one, hide them without; a disk's only for a disk. */
/** Show a magnetic field's rows (and a disk's), and burning's. */
function showMagneticRows(on: boolean, disk = false, burn = false, rad = false) {
  $('diag').classList.toggle('rad', rad);
  $('diag').classList.toggle('mag', on);
  $('diag').classList.toggle('disk', on && disk);
  $('diag').classList.toggle('burn', burn);
  for (const c of diagCells) {
    let show: boolean;
    if (c.col === BURN_COL) show = burn;
    else if (c.col === LIQUID_COL) show = cfg?.eos === 'liquid';
    else if (radCol(c.col)) show = rad;
    else if (magneticCol(c.col)) show = on && (disk || !diskCol(c.col));
    else continue;
    for (const el of [c.dt, c.dd, c.spark]) el.style.display = show ? '' : 'none';
  }
}
showMagneticRows(false);
/** Each sink's block in the panel, shown while that mass is in the run. */
const sinkBlocks = Array.from({ length: MAX_MASSES }, (_, i) => {
  const box = document.createElement('div');
  box.className = 'sink';
  box.hidden = true;
  const head = document.createElement('div');
  head.className = 'sink-head';
  head.textContent = `Sink ${i + 1}`;
  const dl = document.createElement('dl');
  box.append(head, dl);
  sinkPanel.append(box);
  return { box, cells: SINK_FIELDS.slice(0, SINK_SHOWN).map(([, label], k) => diagCell(sinkCol(i, k), label, dl)) };
});
let sinksShown = 0;
/** Each body's block in the panel, shown while the run has that body. */
const bodyBlocks = Array.from({ length: MAX_BODIES }, (_, i) => {
  const box = document.createElement('div');
  box.className = 'sink';
  box.hidden = true;
  const head = document.createElement('div');
  head.className = 'sink-head';
  head.textContent = `Body ${i + 1}`;
  const dl = document.createElement('dl');
  box.append(head, dl);
  sinkPanel.prepend(box);
  return { box, head, cells: BODY_FIELDS.slice(0, BODY_SHOWN).map(([, label], k) => diagCell(bodyCol(i, k), label, dl)) };
});
// (Prepended one by one: put them back in order, ahead of the sinks.)
for (const b of [...bodyBlocks].reverse()) sinkPanel.prepend(b.box);
let bodiesShown = 0;
const allCells = () => [...diagCells, ...bodyBlocks.flatMap((b) => b.cells), ...sinkBlocks.flatMap((b) => b.cells)];
const shownValue = (v: number | undefined, c: DiagCol, u: Readout) => (v !== undefined && Number.isFinite(v) ? fmt3(v * u.scale(c.measure), c.count) : '–');

/** Put the unit of each diagnostic in its tooltip, and show the latest values in the current units. */
function showDiagUnits() {
  const u = units();
  const last = diagHistory.rows[diagHistory.rows.length - 1];
  for (const cell of allCells()) {
    const c = DIAG_COLS[cell.col];
    const l = u.label(c.measure);
    cell.dt.title = l ? `${colTitle(cell.col)}, in ${l}` : colTitle(cell.col);
    if (last) cell.set(shownValue(last[cell.col], c, u));
  }
  drawDiagPlot();
}
const setHud = liveText(hud);

/** The popup with column `col`'s whole run, beside its number. */
function showDiagPlot(col: number, anchor: HTMLElement) {
  diagHover = col;
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
  const c = DIAG_COLS[diagHover];
  const rows = diagHistory.rows;
  const u = units();
  const k = u.scale(c.measure);
  const l = u.label(c.measure);
  const v = rows[rows.length - 1]?.[diagHover];
  const now = v !== undefined && Number.isFinite(v) ? fmt3(v * k, c.count) : '';
  diagPopTitle.textContent = `${colTitle(diagHover)} (${c.label})${now ? ` · ${now}${l ? ` ${l}` : ''}` : ''}`;
  const [xs, ys] = diagHistory.all(diagHover);
  drawFull(diagPopCanvas, xs, k === 1 ? ys : ys.map((y) => y * k), !!c.count);
}

function showDiagnostics(d: Diagnostics, of: Simulation) {
  showToolValues();
  const last = diagHistory.rows[diagHistory.rows.length - 1];
  // A clock held short of the end (a movie's frame time, a checkpoint) reads dt = 0, the step it
  // is about to take: show the last step's instead. With burning the steps shrink to about the
  // frame interval, and every other readout would otherwise be 0.
  if (d.dt === 0 && d.steps > 0 && last && of === diagOf && d.t < cfg.tEnd * (1 - 1e-5)) d = { ...d, dt: last[1] };
  if (of !== diagOf) {
    diagHistory.clear();
    sinkPrev = [];
    radPrev = null;
    diagOf = of;
    showMagneticRows(!!of.cfg.problem.mhd, !!of.cfg.problem.disk, !!burnSpec(of.cfg), of.cfg.radiation === 'm1');
    showDiagUnits();
  } else if (last && d.steps < last[2]) {
    // The run went back: its history starts again.
    sinkPrev = [];
    radPrev = null;
  }
  const u = units();
  const values = [...diagValues(d), ...bodyValues(d), ...sinkValues(d)];
  diagHistory.push(values);
  const n = d.masses.length;
  const nb = Math.min(d.bodies?.length ?? 0, MAX_BODIES);
  if (n !== sinksShown || nb !== bodiesShown) {
    sinksShown = n;
    bodiesShown = nb;
    sinkBlocks.forEach((b, i) => (b.box.hidden = i >= n));
    bodyBlocks.forEach((b, i) => {
      b.box.hidden = i >= nb;
      if (i < nb) b.head.textContent = d.bodies![i].id;
    });
    sinkPanel.hidden = n === 0 && nb === 0;
    showDiagUnits();
  }
  const warns = [false, false, false, false, false, d.floored > 0, d.jeansUnresolved > 0 && !cfg.jeansFloor];
  // Radiation: the reduced speed of light is too slow for the gas somewhere.
  warns[RAD_COL + 4] = (d.radiation?.tooFast ?? 0) > 0;
  const cells = [...diagCells, ...bodyBlocks.slice(0, nb).flatMap((b) => b.cells), ...sinkBlocks.slice(0, n).flatMap((b) => b.cells)];
  for (const cell of cells) {
    cell.set(shownValue(values[cell.col], DIAG_COLS[cell.col], u));
    const warn = warns[cell.col] ?? false;
    if (warn !== cell.warn) {
      cell.warn = warn;
      cell.dd.classList.toggle('warn', warn);
    }
    const [xs, ys] = diagHistory.recent(cell.col);
    drawSpark(cell.spark, xs, ys);
  }
  drawDiagPlot();
}

/** Save the diagnostics of the run so far as CSV, with the columns of every sink that was in it. */
function exportDiagnostics() {
  const width = diagHistory.rows.reduce((w, r) => Math.max(w, r.length), BASE_COLS);
  // A magnetic field's columns only with one, a disk's only for a disk, burning's only for a run that burns.
  const burns = !!burnSpec(cfg);
  // (A body's columns only for the bodies the run has.)
  const bodies = Math.min(cfg.problem.bodies?.length ?? 0, MAX_BODIES);
  const cols = DIAG_COLS.slice(0, width).flatMap((c, i) => (c.csv && (i < BASE_COLS || i >= bodyCol(bodies, 0)) && (!magneticCol(i) || cfg.problem.mhd) && (!diskCol(i) || cfg.problem.disk) && (i !== BURN_COL || burns) && (i !== LIQUID_COL || cfg.eos === 'liquid') && (!radCol(i) || cfg.radiation === 'm1') ? [i] : []));
  const csv = diagHistory.csv(cols, cols.map((i) => DIAG_COLS[i].csv!));
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `shaderous-${cfg.problem.id}-${cfg.ndim}d-${cfg.n}-diagnostics.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

// --- frame loop -------------------------------------------------------------------------
// Stepping runs apart from the display. Steps go out in batches, and a new batch is submitted
// as soon as one is reported done, keeping a few queued so the GPU never waits on the page:
// not for a display refresh, and not for an adaptive run's refinement check, which is
// answered within a batch. The display draws each refresh, behind whatever steps are queued.
//
// A batch aims at half a refresh of GPU time, so the queue (and the delay of what is drawn)
// stays near one refresh. Chrome reports completed work promptly; Firefox only on a ~100 ms
// timer, so there batches are longer and more of them are queued to cover the wait.
let batch = 1; // steps per batch
let queued = 0; // batches submitted and not yet reported done
let queuedSteps = 0; // the steps in them
let resuming = false; // stepping is held, waiting on a refinement check
let msPerStep = 0; // GPU time per step, render included, measured from completed batches
let busyMs = 0; // GPU time and steps since msPerStep was last updated
let busySteps = 0;
let lastDone = 0; // when the last batch was reported done
let pollMs = 0; // how long the browser takes to report already-finished work
let frame = 0;
let lastReadout = 0;
let stepRate = 0; // steps per second of wall time
let stepsSubmitted = 0;
/**
 * The run's time for the view's turn with a rotating frame: the last readout's (with the steps
 * submitted by then), carried on by the steps submitted since at the readout's step. Readouts
 * come five times a second; the view turns every frame.
 */
let timeMark: { t: number; dt: number; submitted: number } | null = null;
/** The rotating frame's angular velocity, for the view's turn: a 3D run in one (not a shearing box), else null. */
function frameOmega(): number[] | null {
  if (!cfg || cfg.ndim !== 3 || cfg.problem.shear) return null;
  const w = rotationOf(cfg).slice(0, 3);
  return w.some((x) => x !== 0) ? w : null;
}
/** Hand the renderer the frame and the time, before it draws. */
function syncFrameTurn() {
  const t = lastDiag && timeMark ? timeMark.t + Math.max(0, stepsSubmitted - timeMark.submitted) * timeMark.dt : (cfg?.restart?.time ?? 0);
  setFrameTurn(frameOmega(), t);
}
let rateMark = { time: 0, steps: 0, frames: 0 };
// Implicit pressure: the tally of the solves of the run it was read from, and whether a read is out.
let solveTally: { of: Simulation; stats: SolveStats } | null = null;
let solveAsked = false;
let fps = 0;
let framesDrawn = 0;
let refreshMs = 16.7; // display refresh interval, estimated from requestAnimationFrame
let lastTick = 0;
const MAX_BATCH = 64;
const loadingEl = $('loading');
const loadingText = $('loading-text');

/** What the loading notice says: the stage the load is at (gpu/progress.ts), with a count while shaders compile. */
function loadingMessage(): string {
  const stage = loadStage();
  const c = compileProgress();
  const compiling = `${COMPILE_STAGE}${c.pending > 0 && c.total > 1 ? ` (${c.done} of ${c.total})` : ''}…`;
  if (stage === COMPILE_STAGE) return compiling;
  if (stage) return `${stage}…`;
  if ((reloading === 'shaders' && !sim?.preparing) || sim?.compiling) return compiling;
  // Autoload off, nothing loaded yet: the page waits for the user.
  if (!sim && pending && !building) return 'Autoload is off: press Run or Load to load it';
  return 'Loading simulation…';
}
function showLoadingMessage() {
  const text = loadingMessage();
  if (loadingText.textContent !== text) loadingText.textContent = text;
  // Waiting for the user (Autoload off), not loading: no spinner.
  loadingEl.classList.toggle('waiting', !sim && !!pending && !building);
}
// The notice follows a load as it goes, the first one included (the frame loop is not running
// yet). Before a stage that blocks the page it is given a frame to show. A page that is drawing
// no frames (hidden, or out of sight while still reporting itself visible, where timers are
// throttled too) must not hold the load up: whether frames are coming is watched directly.
let lastFrameAt = 0;
const frameBeat = () => {
  lastFrameAt = performance.now();
  requestAnimationFrame(frameBeat);
};
requestAnimationFrame(frameBeat);
watchLoading(showLoadingMessage, () => (headless || performance.now() - lastFrameAt > 250 ? Promise.resolve() : painted()));

/**
 * GPU time a batch aims at, and how much to keep queued: enough to cover the delay in
 * reporting completion, plus QUEUE_MS for the page being busy elsewhere (an adaptive run's
 * re-mesh can hold the page for a few milliseconds). Until a step has been timed, three batches;
 * never fewer than two, so a step longer than all that still has the next one queued behind it.
 */
const QUEUE_MS = 12;
/** Besides the browser's delay in reporting work done, the time the page takes over a refinement check's answer. */
const SPEC_MS = 1;
const batchMs = () => Math.max(refreshMs / 2, pollMs / 4);
const queueFull = () => queued >= 64 || (queued >= 2 && (msPerStep ? queuedSteps * msPerStep >= pollMs + QUEUE_MS : queued >= 3));

/** Submit batches of steps until enough are queued. Runs from each completion and each refresh. */
function pump() {
  const s = sim;
  while (s && s === sim && !building && !resuming && running && !driving && !finished && !queueFull()) {
    const enc = gpu.device.createCommandEncoder();
    const timed = frameTiming.open();
    // An adaptive run may take fewer steps than asked: it ends a batch at each refinement check,
    // and holds back while the flow is too far ahead of the mesh. Submit whatever it encoded
    // even so: that can include a re-mesh.
    // Steps may go on behind a refinement check, held on the GPU if it re-meshes (AmrSim): as
    // many as cover the time its answer takes to come back, and the page to act on it.
    const taken = s.encodeSteps(enc, batch, false, msPerStep > 0 ? Math.ceil((pollMs + SPEC_MS) / msPerStep) : 2);
    if (timed) frameTiming.close(enc, taken);
    gpu.device.queue.submit([enc.finish()]);
    s.submitted();
    if (timed) frameTiming.submitted();
    if (!taken) {
      // Held for a refinement check: go on the moment it is answered (or at the next refresh,
      // for any other hold, such as a star still settling).
      const resume = s.stepsResume();
      if (resume) {
        resuming = true;
        resume.then(() => {
          resuming = false;
          pump();
        });
      }
      return;
    }
    stepsSubmitted += taken;
    queued++;
    queuedSteps += taken;
    const t0 = performance.now();
    gpu.device.queue.onSubmittedWorkDone().then(() => {
      queued--;
      queuedSteps -= taken;
      const now = performance.now();
      if (s !== sim) return;
      // The batch could start once submitted and once the work before it was done.
      busyMs += now - Math.max(t0, lastDone);
      busySteps += taken;
      lastDone = now;
      if (busyMs > 4 * batchMs() && busySteps > 0) {
        const m = busyMs / busySteps;
        msPerStep = msPerStep ? 0.5 * (msPerStep + m) : m;
        batch = Math.min(MAX_BATCH, Math.max(1, Math.round(batchMs() / msPerStep)));
        busyMs = busySteps = 0;
      }
      pump();
    });
    // A short batch (an adaptive run ends one at each refinement check) just goes on to the next.
  }
}

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

let lastTiming = 0;

// A paused run with nothing changing draws no frames (see tick()). Whatever the user does to the
// page may change the state by the time it is done: frames are drawn for a moment after.
let lastDraw = 0;
let wakeUntil = 0;
for (const type of ['pointerdown', 'pointerup', 'keydown', 'keyup', 'input', 'change', 'wheel', 'drop'])
  addEventListener(type, () => (wakeUntil = performance.now() + 300), { capture: true, passive: true });

/**
 * The GPU memory field and the line under it: the budget in force, and the automatic one. Where a
 * crash has lowered the automatic budget and it is the one in force, a warning over the
 * simulation says so, and opens this panel when clicked.
 */
function showMemory() {
  const gb = (b: number) => `${+(b / GB).toFixed(2)}`;
  const auto = autoBudget();
  const user = userBudget();
  $<HTMLInputElement>('gpumem').value = gb(user ?? auto.bytes);
  $('gpumem-auto').toggleAttribute('disabled', user === null && !auto.lowered);
  const warn = $('mem-warn');
  warn.hidden = !(user === null && auto.lowered);
  warn.textContent = `⚠ Reduced GPU memory: ${gb(auto.bytes)} GB instead of ${gb(auto.full)} GB, since the page crashed. Fewer blocks fit and large grids are not offered. Check Performance to set it back.`;
  warn.title = 'The page crashed while holding a large allocation, so the automatic GPU memory budget was halved below it. Click to open Performance: set the memory there, or press Auto to go back to the full automatic value';
  const crash = lastCrash() ? ` The page crashed last time, holding about ${gb(lastCrash())} GB.` : '';
  $('gpumem-note').textContent = user === null
    ? `Automatic: ${gb(auto.bytes)} GB (${auto.reason}).${crash}`
    : `Set by you; automatic would be ${gb(auto.bytes)} GB (${auto.reason}).${crash}`;
}

/** The GPU time per step of each pipeline, slowest first. */
function showTiming(r: TimingReport) {
  const box = $('timing');
  if (!r.steps) {
    box.textContent = running ? 'Measuring…' : 'Run the simulation to time its passes.';
    return;
  }
  const table = document.createElement('table');
  const row = (cells: string[], head = false) => {
    const tr = table.insertRow();
    cells.forEach((c, i) => {
      const td = document.createElement(head ? 'th' : 'td');
      td.textContent = c;
      if (i > 0) td.className = 'num';
      tr.appendChild(td);
    });
  };
  row(['Pipeline', 'ms/step', '%', 'passes'], true);
  for (const x of r.rows) row([x.label, x.msPerStep.toFixed(3), (100 * x.share).toFixed(1), x.passesPerStep.toFixed(x.passesPerStep % 1 ? 1 : 0)]);
  row(['Total', r.msPerStep.toFixed(3), '100', '']);
  const note = document.createElement('p');
  note.className = 'timing-note';
  note.textContent = `${r.steps} steps in ${r.frames} frames${r.skipped ? `, ${r.skipped} frames untimed` : ''}`;
  box.replaceChildren(table, note);
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
  // Loading: no simulation yet, a restart or rebuild under way, or a star still settling.
  const loading = !sim || !!reloading || sim.preparing || !!baking;
  // A view that holds one field bakes the field shown when it changes (headless too); one that
  // failed is not tried again until what is shown changes.
  if (
    sim && !building && cfg?.viewBaked && viewing && !baking &&
    (viewing.field !== view.field || viewing.species !== shownSpecies() || (view.fieldLines && !!cfg.problem.mhd && (!cfg.viewFaces || !viewing.faces))) &&
    bakeFailed?.want !== viewWanted()
  ) {
    settleView().catch((e) => {
      const message = e instanceof Error ? e.message : String(e);
      console.warn(`view: ${message}`);
      alert(`Could not read ${viewFieldName().toLowerCase()}${view.fieldLines && cfg?.problem.mhd ? ' and the magnetic field' : ''} from ${viewing?.name ?? 'the file'}: ${message}`);
    });
  }
  if (loadingEl.hidden === loading) loadingEl.hidden = !loading;
  if (loading) showLoadingMessage();
  if (!sim || building || headless) return;
  const s = sim;
  // Top up the steps (normally done as batches complete), then edit and draw behind them.
  pump();
  const enc = gpu.device.createCommandEncoder();
  /** The profile on show, sampled in this frame: read back once it is submitted. */
  let readProfile: (() => void) | null = null;
  advanceStroke(now);
  updateOrbit(now);
  updatePin(now);
  if (pendingBrush) {
    // Logged after it is encoded: a re-mesh that waits goes first (AmrSim.encodeBrush) and logs
    // itself, and the log keeps that order. The stamps are taken before this frame's work is
    // submitted, so both read the step they are applied at.
    s.encodeBrush(enc, pendingBrush);
    logAction({ kind: 'brush', op: pendingBrush });
    pendingBrush = null;
  }
  // The state may have changed since the last frame drawn: steps, an edit, a load, or whatever
  // the user just did to the page. Otherwise a frame is drawn only if the view changed, and
  // twice a second in case the state changed by a path not counted here.
  const changed = running || queued > 0 || driving || resuming || loading || !!pendingBrush || stepOnce || now < wakeUntil || now - lastDraw >= 500;
  let timed = false;
  if (stepOnce && !running) {
    timed = frameTiming.open();
    const taken = s.encodeSteps(enc, 1);
    if (timed) frameTiming.close(enc, taken);
    stepsSubmitted += taken;
  }
  stepOnce = false;
  // While a movie records, its frames are all the canvas shows: the recorder draws them. A
  // native screenshot holds the canvas from its draw until draw(null).
  if (!recordingSize && !nativeShot) {
    updateGuides();
    refreshContourHover();
    renderer.densityCut = densityCut();
    renderer.slice = sliceLayout();
    syncFrameTurn();
    if (renderer.encode(enc, s, view, changed)) {
      lastDraw = now;
      framesDrawn++;
    }
    drawSources();
    drawProfileLine();
    readProfile = encodeFrameProfile(now, enc, s);
  }
  gpu.device.queue.submit([enc.finish()]);
  readProfile?.();
  s.submitted();
  renderer.submitted();
  if (timed) frameTiming.submitted();

  // The masses move every step: their markers follow from a readback of their own, each frame.
  if (sim) {
    const of = sim;
    of.readMasses().then((list) => {
      if (of !== sim) return;
      if (list) liveMasses = { of, list };
      else if (!lastDiag?.masses.length) liveMasses = null;
    });
  }

  // GPU timing, when on: once a second, over the second just past.
  if (now - lastTiming >= 1000) {
    lastTiming = now;
    const r = frameTiming.report();
    if (r) showTiming(r);
  }

  // Readouts refresh five times a second, whatever the display rate.
  frame++;
  if (now - lastReadout >= 200) {
    lastReadout = now;
    const mark = stepsSubmitted;
    s.readDiagnostics().then((d) => {
      if (!d || s !== sim) return;
      lastDiag = d;
      timeMark = { t: d.t, dt: d.dt, submitted: mark };
      session.trace({
        steps: d.steps, t: d.t, dt: d.dt, mass: d.mass, eTot: d.eTot, maxMach: d.maxMach, floored: d.floored,
        ...(d.magnetic ? { bMax: d.magnetic.bMax, betaMin: d.magnetic.betaMin, divB: d.magnetic.divB } : {}),
      });
      // dt = 0 after some progress means the clock reached the end time.
      if (d.dt === 0 && !d.held && d.steps > 0 && running && !finished) {
        if (ckpt.on && cfg.tStop !== undefined && d.t >= cfg.tStop * (1 - 1e-5)) {
          // Held at a checkpoint: write it, and the run goes on.
          if (!ckpt.busy) saveCheckpoint(s);
        } else if (!ckpt.on || !(d.t < cfg.tEnd * (1 - 1e-5))) {
          // (With checkpoints on, a stale readout from a checkpoint just written is no end.)
          finished = true;
          running = false;
          syncControls();
        }
      }
      // A clock held at a checkpoint reads dt = 0 (the step it is about to take) at a time
      // already shown: leave it out of the readouts and their history.
      if (d.dt === 0 && (d.held || (ckpt.on && d.t < cfg.tEnd * (1 - 1e-5)))) return;
      updateRange(d);
      showDiagnostics(d, s);
      showDensityCut();
    });
    if (now - rateMark.time > 500) {
      stepRate = ((stepsSubmitted - rateMark.steps) * 1000) / (now - rateMark.time);
      // Frames drawn, not refreshes of the page: a paused run with a still view draws few.
      fps = ((framesDrawn - rateMark.frames) * 1000) / (now - rateMark.time);
      rateMark = { time: now, steps: stepsSubmitted, frames: framesDrawn };
      if (s.prs && !solveAsked) {
        solveAsked = true;
        // (A run torn down meanwhile fails the read: its tally is of no use anyway.)
        s.prs.stats().then((stats) => (solveTally = { of: s, stats }), () => {}).finally(() => (solveAsked = false));
      }
    }
    setHud(statusLine(s, true));
  }
}

/**
 * The status line: the grid, the simulation time, the point masses and the mesh, in the units
 * the view shows. `live` adds what only the page has (frame rate, step rate, notices); without
 * it, the line is the one painted onto a movie's frames.
 */
function statusLine(s: Simulation, live: boolean): string {
  const u = units();
  const parts = [gridLabel(cfg.n, cfg.ndim)];
  if (live) parts.push(`${fps.toFixed(0)} fps`);
  if (lastDiag) parts.push(`t = ${u.fmt(lastDiag.t, 'time')}`);
  // Comoving coordinates: where the expansion is.
  if (lastDiag?.scale !== undefined) parts.push(`a = ${lastDiag.scale.toPrecision(4)}, z = ${lastDiag.redshift!.toPrecision(4)}`);
  if (live) {
    const rec = recordProgress;
    parts.push(recordingSize ? `recording: ${rec?.frames ?? 0} frames, ${(rec?.seconds ?? 0).toFixed(1)} s of movie` : pending ? 'press Run to load' : finished ? 'finished' : running ? `${batch} step${batch > 1 ? 's' : ''}/batch, ${stepRate.toFixed(0)} steps/s` : 'paused');
  }
  const pm = lastDiag?.masses ?? [];
  const acc = pm.reduce((a, m) => a + m.accreted, 0);
  const took = acc > 0 ? `, accreted ${u.fmt(acc, 'mass')}` : '';
  if (pm.length === 1) parts.push(`point mass at r = ${u.fmt(Math.hypot(...pm[0].pos), 'length')}${took}`);
  else if (pm.length) parts.push(`${pm.length} point masses, separation ${u.fmt(Math.hypot(...pm[0].pos.map((x, i) => x - pm[1].pos[i])), 'length')}${took}`);
  if (solveTally?.of === s && pressureNote(solveTally.stats)) parts.push(pressureNote(solveTally.stats));
  const mesh = s.status();
  if (mesh) parts.push(mesh);
  if (live && toolNotice) parts.push(toolNotice);
  return parts.join('  •  ');
}

/** The page is up: WebGPU found and the first run built (or failed, which window.shaderous reports). */
const ready = (async () => {
  try {
    setLoadStage('Starting WebGPU');
    gpu = await initGpu();
    setAdapter(gpu.info);
    initTiming(gpu.device);
    // Firefox reports no adapter details; the fallback flag is the one thing worth flagging.
    const name = `${gpu.info.vendor} ${gpu.info.architecture}`.trim() || 'WebGPU adapter';
    setAdapterInfo(`${name}${gpu.info.description ? ' (' + gpu.info.description + ')' : ''}`);
    $('adapter').textContent = gpu.info.fallback ? `${name} (software fallback: expect it to be slow)` : name;
    setLoadStage('Preparing the display');
    renderer = new Renderer(gpu, canvas);
    await renderer.init();
    await measurePollLatency();
    editor.set(defaultOpacity(), 'inferno');
    pushTransfer();
    bind();
    // With Autoload off the page opens on the problem's settings, and loads it on Run or Load.
    // A link with a setup of its own: unpacked and made the custom setup before it is read.
    let first = firstLink;
    const packed = new URLSearchParams(location.search).get('setup');
    if (packed) {
      try {
        const { spec, report } = readSetup(await decodeSetupParam(packed), (id) => PRESET_FILES[id]);
        if (!spec) throw new Error(report.error);
        for (const n of report.notes) console.warn(`setup: ${n}`);
        setCustomSetup(spec);
        packCustom();
        first = decodeLink(new URLSearchParams(location.search));
      } catch (e) {
        console.warn(`the setup in the link could not be read: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    await build(first.problem, first.n, first.ndim, first.amr, false, first, undefined, !autoload);
    requestAnimationFrame(tick);
  } catch (e) {
    fail(e);
    throw e;
  }
})();

// window.shaderous: the page as a script or an agent drives it (ui/api.ts, docs/api.md).
const api = installApi(
  {
    gpu: () => gpu,
    adapter: () => `${gpu.info.vendor} ${gpu.info.architecture}${gpu.info.description ? ` (${gpu.info.description})` : ''}${gpu.info.fallback ? ' [software fallback]' : ''}`.trim(),
    sim: () => sim,
    cfg: () => cfg,
    view,
    canvas,
    viewSize: () => [canvas.width, canvas.height] as [number, number],
    viewBounds() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (!(w > 0 && h > 0)) return null;
      if (flatView()) {
        // The box is a square, as locate() maps the canvas onto it.
        const { span, shift } = sliceLayout();
        const side = span * view.zoom;
        const x0 = w / 2 - shift - view.cx * side;
        const y0 = h / 2 - (1 - view.cy) * side;
        return [x0 / w, y0 / h, (x0 + side) / w, (y0 + side) / h];
      }
      // The volume: the box's eight corners, as the camera projects them.
      const f = cameraFrame(view, w / h);
      const b: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
      for (let i = 0; i < 8; i++) {
        const rel = [0, 1, 2].map((k) => ((i >> k) & 1) - 0.5 - f.eye[k]);
        const along = (v: number[]) => rel[0] * v[0] + rel[1] * v[1] + rel[2] * v[2];
        const z = along(f.fwd);
        // A corner beside or behind the camera has no place on the screen.
        if (z < 0.05) return null;
        const x = (along(f.right) / (z * (w / h) * f.tan) + 1) / 2;
        const y = (1 - along(f.up) / (z * f.tan)) / 2;
        b[0] = Math.min(b[0], x);
        b[1] = Math.min(b[1], y);
        b[2] = Math.max(b[2], x);
        b[3] = Math.max(b[3], y);
      }
      return b;
    },
    background: () => backgroundCss(view.light),
    resolutions: () => ({ 1: RES_1D, 2: RES_2D, 3: RES_3D, uniformMax: { 1: uniformMax(1), 2: uniformMax(2), 3: uniformMax(3) } }),
    idle,
    build: (link) => build(link.problem, link.n, link.ndim, link.amr, false, link),
    importAthdf,
    importSnapshot,
    viewFile: viewState,
    viewProbe,
    settleView,
    profile: apiProfile,
    restart: () => restartRun(),
    running: () => running,
    finished: () => finished,
    setRunning(on) {
      if (on === running) return;
      running = on && !finished;
      syncControls();
    },
    setFinished(on) {
      finished = on;
      if (on) running = false;
      syncControls();
    },
    setDriving(on) {
      driving = on;
    },
    rearm: (t) => armCheckpoints(t),
    query: () => linkQuery(),
    refresh(d) {
      lastDiag = d;
      timeMark = { t: d.t, dt: d.dt, submitted: stepsSubmitted };
      updateRange(d);
      showDensityCut();
    },
    draw(size, asPage, capture = false) {
      if (!sim) return;
      const enc = gpu.device.createCommandEncoder();
      renderer.fixedSize = size;
      renderer.capture = capture;
      renderer.densityCut = densityCut();
      // At a size of its own the slice is centered, since the panel and rail are not in the
      // picture; unless the picture is the view area itself, framed as the page frames it.
      const k = size ? size[0] / Math.max(1, canvas.clientWidth) : 1;
      const page = sliceLayout();
      renderer.slice = !size ? page : asPage ? { span: page.span * k, shift: page.shift * k } : { span: 0.94 * Math.min(...size), shift: 0 };
      syncFrameTurn();
      renderer.encode(enc, sim, view);
      renderer.capture = false;
      gpu.device.queue.submit([enc.finish()]);
      sim.submitted();
      renderer.submitted();
      // Back from a native draw: the canvas is the page's again (or the movie's).
      if (!size && nativeShot) {
        nativeShot = false;
        setRecordingStyle(recordingSize);
      }
    },
    drawNative(region, capture = false) {
      if (!sim) throw new Error('no run is loaded');
      if (!flatView()) throw new Error('the volume view has no pixel per cell: use slice or projection');
      // A 1D run's picture: its rings drawn on a square of at most NATIVE_RINGS a side in a curved
      // run (which has no pixel a cell anyway); in a Cartesian one, a strip (nativeStrip).
      const side = cfg.ndim === 1 && curvedView(cfg) ? Math.min(renderer.mapSide, NATIVE_RINGS) : renderer.mapSide;
      const r = cfg.ndim === 1 && !curvedView(cfg) ? nativeStrip(nativeRegion(region, side)) : nativeRegion(region, side);
      const w = r.i1 - r.i0;
      const h = r.j1 - r.j0;
      if (!(w >= 1 && h >= 1)) throw new Error('the box is not in view');
      const max = gpu.device.limits.maxTextureDimension2D;
      if (w > max || h > max) throw new Error(`a picture of ${w} × ${h} pixels is larger than the GPU draws (${max} a side)`);
      const enc = gpu.device.createCommandEncoder();
      // One pixel per texel: the slice spans `side` pixels at zoom 1, centered so that pixel
      // centers fall on texel centers (render/slice.wgsl), on a copy of the view.
      renderer.fixedSize = [w, h];
      renderer.capture = capture;
      renderer.densityCut = densityCut();
      renderer.slice = { span: side, shift: 0 };
      // (A texel a pixel of the map over the whole box, not a 1D run's window of it.)
      renderer.fullMap = true;
      const hover = renderer.hoverLevel;
      renderer.hoverLevel = 0;
      // The canvas grows to the picture's size until draw(null): shown fitted, the pointer waiting.
      setRecordingStyle([w, h]);
      nativeShot = true;
      try {
        renderer.encode(enc, sim, { ...view, zoom: 1, cx: (r.i0 + r.i1) / (2 * side), cy: (r.j0 + r.j1) / (2 * side) });
      } finally {
        renderer.hoverLevel = hover;
        renderer.capture = false;
        renderer.fullMap = false;
      }
      gpu.device.queue.submit([enc.finish()]);
      sim.submitted();
      renderer.submitted();
      return { width: w, height: h, side, n: sim.n };
    },
    flatView: () => flatView(),
    captured: () => renderer.captured(),
    rangeSettled: () => renderer.rangeSettled(),
    rangedByDrawing: (d) => $<HTMLInputElement>('auto').checked && (projecting() || view.field >= d.ranges.length || !!cfg.viewBaked),
    athdf: () => athdfOf(sim!),
    snapshot: () => snapshotOf(sim!),
    setColormap(name) {
      editor.set(editor.points, name);
      pushTransfer();
    },
    opacityCurve: () => editor.points.map((p) => ({ ...p })),
    setOpacityCurve(points) {
      editor.set(points, editor.colormap);
      pushTransfer();
    },
    setAutoRange(on) {
      $<HTMLInputElement>('auto').checked = on;
    },
    sync: () => syncControls(),
    runningSpec: () => runningSpec(cfg),
    applySpec,
    adopt,
    tool: (name, opts) => toolApi(name, opts),
    sessionLog: () => session.snapshot(),
    replayAction,
    setReplaying(on) {
      replaying = on;
      if (sim?.autoRegrid !== undefined) sim.autoRegrid = !on;
      if (sim?.autoBins !== undefined) sim.autoBins = !on;
    },
    gridMemory: (spec) => setupHost.memory(spec)!,
    refusal: (spec) => setupHost.refusal(spec),
    overlayInfo() {
      const u = units();
      const m = viewMeasure();
      const k = u.scale(m);
      const unit = u.label(m);
      const name = viewFieldName();
      const { right, up } = cameraFrame(view, 1);
      return {
        colormap: editor.colormap,
        lo: view.lo * k,
        hi: view.hi * k,
        log: view.log && !LINEAR_FIELDS.has(view.field),
        title: unit ? `${name} (${unit})` : name,
        mode: cfg.ndim === 3 && view.mode === 'volume' ? 'volume' : 'plane',
        sliceAxis: cfg.ndim < 3 ? 2 : view.sliceAxis,
        right,
        up,
        twoD: cfg.ndim < 3,
        light: view.light,
        status: sim ? statusLine(sim, false) : '',
        oneD: oneDChip(cfg)?.text,
      };
    },
    setRecording(size) {
      recordingSize = size;
      setRecordingStyle(size);
      syncRecordButton();
    },
    recordProgress(p) {
      recordProgress = p;
      syncRecordButton();
    },
    checkpointsOn: () => ckpt.on,
    writeCheckpoint: () => (sim ? saveCheckpoint(sim) : Promise.resolve()),
  },
  ready,
);

// Handles for the console and for tests.
Object.assign(window, { shaderousApp: { get sim() { return sim; }, get cfg() { return cfg; }, view, get diag() { return lastDiag; }, get batch() { return batch; }, get stepRate() { return stepRate; }, get fps() { return fps; }, get pollMs() { return pollMs; }, timing: { ...frameTiming, set: setTiming },
  /** The star tool, as a click with these settings would use it: resolves once the star is placed, settled and magnetized. */
  placeStar(pos: Vec3, o: { n?: number; size?: number; logRho?: number; field?: StarField; logBeta?: number; vel?: Vec3 } = {}) {
    if (o.n !== undefined) starN = o.n;
    if (o.size !== undefined || o.logRho !== undefined) starSet = true;
    if (o.size !== undefined) toolParams.star.size = o.size;
    if (o.logRho !== undefined) starLogRho = o.logRho;
    if (o.field !== undefined) starField = o.field;
    if (o.logBeta !== undefined) starLogBeta = o.logBeta;
    placeStar(pos, o.vel ?? [0, 0, 0]);
    return enqueue(async () => {}).then(() => toolNotice);
  },
} });

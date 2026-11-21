// The scripting API's methods as data: each one's name, what it does in a sentence or two, and
// its arguments as JSON Schema. The MCP server takes its tools' input schemas and the opening
// of their descriptions from here (through tools/headless/schema.json), and `describe()` returns
// the list, so a method's arguments are written once. Results are not described here: they are
// in docs/api.md.
//
// A page test checks that this names exactly the methods `window.shaderous` has.
import { DERIVED_VARIABLES, VARIABLES } from '../io/snapshot';
import { VIEW_EXTRAS, VIEW_OPTIONS } from '../render/view_options';

/** As much of JSON Schema as the arguments need. */
export interface ArgSchema {
  type?: string | string[];
  enum?: string[];
  description?: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  minItems?: number;
  maxItems?: number;
  items?: ArgSchema;
  properties?: Record<string, ArgSchema>;
  required?: string[];
}

export interface ApiMethod {
  name: string;
  /** what it does: the opening of the matching MCP tool's description */
  summary: string;
  /** its arguments, in order. One named `options` is an object of named settings (`properties`). */
  args: (ArgSchema & { name: string; optional?: boolean })[];
  /** it steps, restarts or replaces the run (the rest only read it, or change what is drawn) */
  changesRun: boolean;
}

const VIEW_KEYS = [...VIEW_OPTIONS, ...VIEW_EXTRAS].map((o) => o.key).join(', ');
const size = (what: string): ArgSchema => ({ type: 'integer', minimum: 16, maximum: 4096, description: what });

export const API_METHODS: ApiMethod[] = [
  { name: 'describe', summary: 'List the problems (presets) with their ids, dimensions, end times and parameters, and the fields, variables and choices a setup can make.', args: [], changesRun: false },
  {
    name: 'check',
    summary: 'Check a setup document before running it. Returns `ok`, an `error` if it cannot be read at all (with the key at fault), `notes` (keys dropped, values held to their range), and `issues`: each with a level (`error`: a rule broken, such as a star without isolated self-gravity, or a value that cannot work; `warning`: a choice to look at again, such as a softening under two cells, bodies that overlap or leave the box, a grid past the GPU memory), a sentence, and the path of the part or option it is about. Use it after writing or changing a document and before `load`.',
    args: [{ name: 'document', type: 'object', description: 'The setup document (format "shaderous-setup")' }],
    changesRun: false,
  },
  {
    name: 'meshFile',
    summary: 'Give the page a mesh file a setup document refers to (a `mesh` shape’s `file`: STL, OBJ, PLY, glTF, GLB, or an airfoil .dat or SVG outline), by that name, before `load`. Returns its size in bytes.',
    args: [
      { name: 'name', type: 'string', description: 'The name the setup gives the file' },
      { name: 'base64', type: 'string', description: 'The file’s bytes, base64-encoded' },
    ],
    changesRun: false,
  },
  { name: 'preset', summary: 'A preset’s whole setup document: a starting point for one of your own.', args: [{ name: 'id', type: 'string', description: 'A problem id' }], changesRun: false },
  {
    name: 'load',
    summary: 'Build a new run from a setup and leave it paused at t = 0. A setup is a preset with overrides, or a whole setup document (`document`) that assembles a run from parts: point masses, stars, disks, gas regions, obstacles, a magnetic field. The setup uses the keys of a shareable link, as a query string ("problem=sedov&dim=2&n=512&gamma=1.4") or an object ({"problem": "sedov", "dim": 2, "n": 512}). Returns the run’s state, including `query`, the setup in force.',
    args: [
      { name: 'setup', type: ['string', 'object'], description: 'Query string, link, or object of setup keys; or a setup document itself' },
      { name: 'document', optional: true, type: 'object', description: 'A setup document (format "shaderous-setup", see docs/setups.md and describe().setup): the run is built from it, and `setup` then says the grid and what differs. Start from preset(id) to make one' },
    ],
    changesRun: true,
  },
  { name: 'loadAthdf', summary: 'Start a run from an .athdf file: one Shaderous wrote continues exactly; another code’s is loaded into the current problem. Paused at the file’s time.', args: [{ name: 'base64', type: 'string', description: 'The file’s bytes, base64' }], changesRun: true },
  { name: 'state', summary: 'The run: problem, dimensions, grid, amr levels, time, timestep, steps, end time, whether it has finished, the GPU adapter, and `query`, the setup that reproduces it.', args: [], changesRun: false },
  {
    name: 'run',
    summary: 'Advance the run: to time `until` (the last step is shortened to land on it exactly), by `steps` steps, or whichever comes first. Stops at the problem’s end time (stopped: "end"); to go further, load with a later `tend`. Returns t, dt, steps, steps taken, wall seconds and why it stopped.',
    args: [{
      name: 'options', type: 'object',
      properties: {
        until: { type: 'number', description: 'Run until the clock reads this time' },
        steps: { type: 'integer', minimum: 0, description: 'Take at most this many steps' },
        timeoutMs: { type: 'number', description: 'Give up after this much wall time, in milliseconds' },
      },
    }],
    changesRun: true,
  },
  { name: 'step', summary: 'Take n steps (one, without n).', args: [{ name: 'n', optional: true, type: 'integer', minimum: 0 }], changesRun: true },
  { name: 'runTo', summary: 'Run until the clock reads t, landed on exactly.', args: [{ name: 't', type: 'number' }, { name: 'timeoutMs', optional: true, type: 'number' }], changesRun: true },
  { name: 'restart', summary: 'Start the run again from its initial conditions, paused.', args: [], changesRun: true },
  { name: 'play', summary: 'Let the page’s own loop step the run, as the Run button does.', args: [], changesRun: true },
  { name: 'pause', summary: 'Stop the page’s own loop from stepping the run.', args: [], changesRun: true },
  { name: 'diagnostics', summary: 'Totals and extremes read back now: mass, momentum, kinetic/internal/gravitational/total energy, max Mach number, floored cells, Jeans-unresolved cells, center of mass, point masses, and the [min, max] of every field.', args: [], changesRun: false },
  { name: 'debris', summary: 'Debris energies, for problems that track them (describe: problems[].debris, e.g. tidal-disruption): the gas binned by its specific orbital energy about the point mass, both what has left the box (tallied as it left) and what is in it, a surviving self-bound core set apart, and the return rate of the bound debris to pericenter, dM/dt (Kepler).', args: [], changesRun: false },
  { name: 'impact', summary: 'What a giant impact has left (3D runs with gravity, e.g. moon-forming-impact): the gas sorted by the Kepler orbit each cell is on about the densest body into the planet (pericenter inside it), the disk around it and what escapes, with each part’s mass, the share of it that is each species and the mean dye, the disk’s angular momentum, and the mass of the moon the disk would gather into.', args: [], changesRun: false },
  { name: 'stats', summary: 'Least, greatest and volume-weighted mean of each variable over the box (rho, press, vel1-3, r0 and species mass fractions).', args: [], changesRun: false },
  {
    name: 'slice',
    summary: 'A variable on a res x res grid on a plane through the box (perpendicular to `axis` at fraction `pos`).',
    args: [{
      name: 'options', optional: true, type: 'object',
      properties: {
        variable: { type: 'string', description: `One of ${[...VARIABLES, ...DERIVED_VARIABLES].join(', ')}, or the name of a species. temp is the temperature in code units, tempK in kelvin (a setup with physical units). Default rho` },
        axis: { type: 'integer', minimum: 0, maximum: 2, description: 'Default 2 (z); always 2 in 2D' },
        pos: { type: 'number', minimum: 0, maximum: 1, description: 'Default 0.5' },
        res: { type: 'integer', minimum: 2, maximum: 2048, description: 'Default: the size of the grid, at most 256 (the MCP tool: 32)' },
        encoding: { type: 'string', enum: ['json', 'base64'], description: "json (default): the values as a list; base64: as float32 bytes in `array`, many times faster to bring out of the page at a large res" },
      },
    }],
    changesRun: false,
  },
  {
    name: 'probe',
    summary: 'The gas at points given as fractions of the box ([x, y, z], z ignored in 2D): density, pressure, velocity, passive scalar, temperature (temp in code units; tempK in kelvin, with physical units), refinement level and species mass fractions.',
    args: [{ name: 'points', type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 }, minItems: 1 }],
    changesRun: false,
  },
  {
    name: 'screenshot',
    summary: 'A picture of the run as a PNG image.',
    args: [{
      name: 'options', optional: true, type: 'object',
      properties: {
        width: size('Default 768'),
        height: size('Default: width'),
        native: { type: ['boolean', 'string'], description: "One pixel per cell of the finest level (the grid's own resolution, up to the largest texture): true or 'box' for the whole box, 'view' for the part in view. A plane only (a 2D run, a slice or a projection); not with width or height" },
        overlays: { type: 'object', description: '{ colorbar, triad, status }: draw the color bar, the axes and the status line (all off by default)' },
        layout: { type: 'string', enum: ['over', 'beside'], description: "'over': the overlays over the picture's edges, as in a movie; 'beside': in margins around it, covering nothing. Default 'beside' with native, else 'over'" },
      },
    }],
    changesRun: false,
  },
  {
    name: 'record',
    summary: 'Record the run from where it is as an MP4 movie (H.264), stepping it as it goes: the current view with a color bar, x/y/z axes and the status line (grid, simulation time) over it. A frame every `every` steps, or near every `dt` of code time (at the nearest step boundary: no step is shortened, so the run is the one it would have been unrecorded). Stops at `until`, after `frames` frames, or at the end time. Set the view first.',
    args: [{
      name: 'options', optional: true, type: 'object',
      properties: {
        dt: { type: 'number', exclusiveMinimum: 0, description: 'A frame near every multiple of this much code time (default: a 600th of the dynamical time of the problem, so one dynamical time plays in 10 s at 60 fps)' },
        every: { type: 'integer', minimum: 1, description: 'Or a frame every this many steps' },
        until: { type: 'number', description: 'Stop when the clock reaches this time (landed on exactly)' },
        frames: { type: 'integer', minimum: 1, description: 'Stop after this many frames' },
        fps: { type: 'number', minimum: 1, maximum: 240, description: 'Playback rate (default 60)' },
        width: size('Default 1920'),
        height: size('Default 1080 (9/16 of width when width is given)'),
        bitrate: { type: 'number', exclusiveMinimum: 0, description: 'Bits per second of movie (default 0.2 bits per pixel per frame)' },
        overlays: { type: 'object', description: '{ colorbar, triad, status }, all on by default' },
        timeoutMs: { type: 'number', description: 'Give up after this much wall time, keeping what was recorded' },
        download: { type: 'boolean', description: 'Hand the file to the browser as a download (default true)' },
        base64: { type: 'boolean', description: 'Return the file in the result as `base64`' },
      },
    }],
    changesRun: true,
  },
  { name: 'stopRecording', summary: 'End the recording under way: the file is finished with the frames recorded so far.', args: [], changesRun: false },
  { name: 'recordingStatus', summary: 'The recording under way, or null.', args: [], changesRun: false },
  {
    name: 'view',
    summary: 'Change what screenshots show without touching the run. Returns the view settings.',
    args: [{ name: 'changes', type: 'object', description: `Any of ${VIEW_KEYS}` }],
    changesRun: false,
  },
  { name: 'athdf', summary: 'The state of the gas as an .athdf file (HDF5 in Athena++’s layout; yt reads it), base64. The file carries the setup, so loading it continues the run exactly.', args: [], changesRun: false },
  {
    name: 'athdfChunk',
    summary: 'Bytes of the .athdf file athdf({ chunked: true }) holds, base64: a file of gigabytes is read out in pieces.',
    args: [{ name: 'offset', type: 'integer', minimum: 0 }, { name: 'length', type: 'integer', minimum: 1 }],
    changesRun: false,
  },
  {
    name: 'athdfPut',
    summary: 'One piece of an .athdf file for loadAthdf() with no argument: its bytes from `offset`, base64, of a file of `total` bytes.',
    args: [{ name: 'offset', type: 'integer', minimum: 0 }, { name: 'base64', type: 'string' }, { name: 'total', type: 'integer', minimum: 1 }],
    changesRun: false,
  },
  {
    name: 'snapshotOut',
    summary: 'The state of the gas as a snapshot held in the page, for a caller that writes the .athdf file itself: its fields but the arrays, and each array’s key, type and length. Read the arrays with snapshotChunk, then call snapshotDone.',
    args: [],
    changesRun: false,
  },
  {
    name: 'snapshotChunk',
    summary: 'Bytes of an array of the snapshot snapshotOut() holds, base64.',
    args: [{ name: 'key', type: 'string' }, { name: 'offset', type: 'integer', minimum: 0 }, { name: 'length', type: 'integer', minimum: 1 }],
    changesRun: false,
  },
  { name: 'snapshotDone', summary: 'Let the snapshot snapshotOut() holds go.', args: [], changesRun: false },
  {
    name: 'snapshotIn',
    summary: 'Set out a snapshot for loadSnapshot(): its fields but the arrays, and the arrays’ keys, types and lengths, as snapshotOut gives them.',
    args: [{ name: 'header', type: 'object' }, { name: 'arrays', type: 'array', items: { type: 'object' } }],
    changesRun: false,
  },
  {
    name: 'snapshotPut',
    summary: 'One piece of an array of the snapshot snapshotIn() set out: its bytes from `offset`, base64.',
    args: [{ name: 'key', type: 'string' }, { name: 'offset', type: 'integer', minimum: 0 }, { name: 'base64', type: 'string' }],
    changesRun: false,
  },
  { name: 'loadSnapshot', summary: 'Start a run from the snapshot snapshotIn() and snapshotPut() put together, as loadAthdf does from a file. Paused at its time.', args: [], changesRun: true },
  {
    name: 'viewAthdf',
    summary: 'Open an .athdf checkpoint for viewing only, for runs too large to continue here: read without copying it whole, held with its state (every field) or holding only the field shown (baked from the file again when the field changes), whichever keeps more of its levels, its finest levels merged until it fits the GPU memory budget. Uniform grids too, as blocks of 16. It never steps (run refuses); probes, screenshots and the view work, and with the state slices and statistics too. Returns the state and `view`: the level kept, the file’s finest, leaf blocks after and before, and `oneField`.',
    args: [
      { name: 'url', optional: true, type: 'string', description: 'A URL the page can fetch the file from (the dev server serves the project’s files); without it, the file athdfPut put together' },
      { name: 'opts', optional: true, type: 'object', description: '{ oneField }: hold only the field shown (true) or the state (false), instead of whichever keeps more levels' },
    ],
    changesRun: true,
  },
  { name: 'viewBudget', summary: 'The GPU memory the adaptive mesh may use, in bytes: what a caller sizing a view-only snapshot (snapshotIn with a `view` field) plans against.', args: [], changesRun: false },
  { name: 'controls', summary: 'The control panel as it stands: every control’s id, kind, label, tooltip, value and options. Use with `control` to change settings mid-run.', args: [], changesRun: false },
  {
    name: 'control',
    summary: 'Set a panel control (or press a button, with no value) exactly as a person would, mid-run, and wait for what it starts (a shader rebuild, a restart). Examples: control("riemann", "hll"), control("eos", "gas_radiation"), control("reset"). Number fields take values in the units the panel shows; set control("units", "code") first for code units. The run stays paused.',
    args: [{ name: 'id', type: 'string' }, { name: 'value', optional: true, type: ['string', 'number', 'boolean'] }],
    changesRun: true,
  },
  {
    name: 'tool',
    summary: 'Use one of the editing tools, as a person would with the pointer, applied now between steps and logged: mass (add or remove gas), velocity, temperature, obstacle (a shape in a plane of the box, cut through it along the plane normal), field (make the magnetic field stronger or weaker), pull (drag the field lines from one point to another), source, sink, star (3D), clear (the sources or the sinks), or query (the gas at a point). Positions are fractions of the box ([x, y] or [x, y, z]), velocities code units; the rest as the tool’s sliders take them. Unknown options are an error.',
    args: [
      { name: 'name', type: 'string', enum: ['mass', 'velocity', 'temperature', 'obstacle', 'field', 'pull', 'source', 'sink', 'star', 'clear', 'query'] },
      {
        name: 'options', optional: true, type: 'object',
        properties: {
          pos: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 3, description: 'Center, in fractions of the box (z defaults to the middle)' },
          radius: { type: 'number', exclusiveMinimum: 0, description: 'Brush radius, or the radius of a source, sink or star, as a fraction of the box' },
          softness: { type: 'number', minimum: 0, maximum: 1, description: 'Brush edge: 0 hard, 1 fading all the way from the center' },
          amount: { type: 'number', minimum: -1, maximum: 1, description: 'mass: density added at the center as a share of the densest gas (negative removes; default 0.25)' },
          atRest: { type: 'boolean', description: 'mass: new gas arrives at rest, sharing the momentum there (default true)' },
          vel: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 3, description: 'velocity: the velocity to set; source, sink, star: its velocity (code units)' },
          factor: { type: 'number', exclusiveMinimum: 0, description: 'temperature, field: the factor applied at the center (default 1.5; below 1 lowers it)' },
          shape: { type: 'string', enum: ['rect', 'ellipse', 'polygon'], description: 'obstacle: its shape' },
          center: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2, description: 'obstacle: rect or ellipse center, in the plane’s two coordinates' },
          half: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2, description: 'obstacle: rect or ellipse half-widths' },
          points: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }, minItems: 3, description: 'obstacle: a polygon’s corners' },
          axis: { type: 'integer', minimum: 0, maximum: 2, description: 'obstacle: the plane’s normal, along which it is cut through the box (default 2, z; its coordinates are then x and y, for 0 y and z, for 1 x and z)' },
          erase: { type: 'boolean', description: 'obstacle: turn obstacle back into gas' },
          from: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 3, description: 'pull: where the drag starts' },
          to: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 3, description: 'pull: where it ends (taken frame by frame, as the tool would)' },
          grip: { type: 'number', minimum: 0.01, maximum: 0.9, description: 'pull: the share of the drag the field at the center moves by (default 0.5)' },
          rate: { type: 'number', minimum: 0, maximum: 1, description: 'source, sink: the tool’s Flux (default 0.25)' },
          mass: { type: 'number', minimum: 0, description: 'sink: its mass as a multiple of the gas in the box (default 0: a massless particle)' },
          soften: { type: 'number', exclusiveMinimum: 0, description: 'sink: softening length of its gravity, as a fraction of the box' },
          fixed: { type: 'boolean', description: 'source, sink: held where it is placed (default false)' },
          n: { type: 'number', description: 'star: polytropic index' },
          rhoc: { type: 'number', exclusiveMinimum: 0, description: 'star: central density (code units)' },
          field: { type: 'string', enum: ['none', 'dipole', 'random'], description: 'star: its magnetic field' },
          beta: { type: 'number', minimum: 1, description: 'star: plasma beta at its center' },
          what: { type: 'string', enum: ['sources', 'sinks'], description: 'clear: which to remove' },
        },
      },
    ],
    changesRun: true,
  },
  { name: 'actions', summary: 'The session log of the run: the setup it started from, and every edit since (the tools, settings changed, the adaptive mesh’s re-meshes), each with the step it was applied at, with the run’s diagnostics along the way. The page’s Save session log button saves the same.', args: [], changesRun: false },
  {
    name: 'replay',
    summary: 'Run a session log again: load the setup it started from, step to each action and apply it, re-meshing where it re-meshed, so the run comes out as it did (to the bit, on the same machine). Returns how many actions it applied, and where it stopped if the run went another way.',
    args: [
      { name: 'log', type: 'object', description: 'A session log, as actions() returns it' },
      { name: 'options', optional: true, type: 'object', properties: { untilSteps: { type: 'integer', minimum: 0, description: 'Stop at this many steps (default: after the last action)' } } },
    ],
    changesRun: true,
  },
  { name: 'errors', summary: 'What the page has reported going wrong (uncaught errors, failed shader builds, a lost GPU device), oldest first.', args: [], changesRun: false },
];

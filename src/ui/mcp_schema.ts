// The MCP server's tools as data: each tool's name, the method of the scripting API it wraps,
// the arguments it passes on to it (described as the API describes them, api_schema.ts) and
// what it adds of its own (a file to save to, seconds where the API takes milliseconds).
//
// `npm run gen` writes the finished definitions (name, description, input schema) to
// tools/headless/schema.json, which the server (tools/headless/mcp.mjs) serves as they are: it
// holds only what each tool does when called. The tables of tools in docs/headless.md and in
// the agent skill are written from `brief`.
import { VIEW_EXTRAS, VIEW_OPTIONS } from '../render/view_options';
import { API_METHODS, type ArgSchema } from './api_schema';

export interface McpTool {
  name: string;
  /** the API method it wraps: its description opens with that method's summary */
  method: string;
  /** one line for the tables of the docs */
  brief: string;
  /** what follows the method's summary in its description; with `alone`, the whole description */
  more?: string;
  alone?: boolean;
  /** arguments passed on to the method, by the name the method gives them */
  pass?: string[];
  /** arguments of its own */
  own?: Record<string, ArgSchema>;
  required?: string[];
}

const VIEW_KEYS = [...VIEW_OPTIONS, ...VIEW_EXTRAS].map((o) => (o.key === 'mode' ? 'mode (volume|slice|projection)' : o.key === 'field' ? 'field (index, see describe)' : o.key)).join(', ');

export const MCP_TOOLS: McpTool[] = [
  {
    name: 'describe', method: 'describe',
    brief: 'Lists problems (id, dimensions, end time, parameters), fields, variables and choices. Give `problem` for one problem’s full entry, with its setup document',
    more: 'Give `problem` for one problem in full, with its description.',
    own: { problem: { type: 'string', description: 'A problem id for its full entry' } },
  },
  { name: 'check_setup', method: 'check', brief: 'Checks a setup document without running it: whether it can be read, what reading left out, and the errors and warnings the Setup dialog would list', pass: ['document'], required: ['document'],
    own: { dir: { type: 'string', description: 'The folder the document’s mesh files are in (its `mesh` shapes name them); default the server’s working folder' } } },
  { name: 'load', method: 'load', brief: 'Builds a run from a setup (query string or object) or a setup document, paused at t = 0', pass: ['setup', 'document'],
    own: { dir: { type: 'string', description: 'The folder the document’s mesh files are in (its `mesh` shapes name them); default the server’s working folder' } } },
  {
    name: 'run', method: 'run',
    brief: 'Advances to a time (`until`, landed on exactly), by a number of `steps`, or whichever comes first, with an optional `timeout_seconds`',
    pass: ['until', 'steps'],
    own: { timeout_seconds: { type: 'number', description: 'Give up after this much wall time (default 600)' } },
  },
  { name: 'state', method: 'state', brief: 'Returns the problem, grid, time, steps, end time, GPU adapter, and `query`, the setup in force' },
  { name: 'diagnostics', method: 'diagnostics', brief: 'Reads back totals and extremes: mass, momentum, energies, Mach number, point masses, field ranges' },
  { name: 'stats', method: 'stats', brief: 'Gives each variable’s min, max and volume-weighted mean over the box' },
  {
    name: 'slice', method: 'slice',
    brief: 'Samples a variable on a plane. Returns min, max and mean, with the values as rows when `res` is at most 64; `save_path` writes the full slice to a file instead: a `.npy` file (float32, shape `[res, res]`) if the name ends in `.npy`, else JSON',
    more: 'Returns min, max and mean, and the values (row j, column i: values[j][i]) when res is at most 64 or include_values is set; with save_path the full slice is written there instead: as a .npy file (float32, shape [res, res], for numpy’s np.load) when the name ends in .npy, else as JSON.',
    pass: ['variable', 'axis', 'pos', 'res'],
    own: {
      include_values: { type: 'boolean', description: 'Return the values at any res (default: only when res is at most 64)' },
      save_path: { type: 'string', description: 'Write the slice to this file: .npy (float32 for numpy), else JSON' },
    },
  },
  { name: 'probe', method: 'probe', brief: 'Gives the gas at given points', pass: ['points'], required: ['points'] },
  {
    name: 'screenshot', method: 'screenshot',
    brief: 'Returns a PNG image of the view, optionally changing the `view` first and saving to `save_path`',
    more: `\`view\` changes what is drawn first: ${VIEW_KEYS}. Settings persist for later screenshots. \`native\` saves a plane at the grid's own resolution (one pixel per finest cell), with any overlays in margins beside it.`,
    pass: ['width', 'height', 'native', 'layout'],
    own: {
      view: { type: 'object', description: 'View settings to apply first' },
      save_path: { type: 'string', description: 'Also write the PNG to this file' },
      colorbar: { type: 'boolean', description: 'The color bar (default false)' },
      axes: { type: 'boolean', description: 'The axes (default false)' },
      status: { type: 'boolean', description: 'The status line: grid, simulation time, point masses, mesh (default false)' },
    },
  },
  { name: 'view', method: 'view', brief: 'Changes what screenshots show, leaving the run alone', pass: ['changes'], required: ['changes'] },
  { name: 'controls', method: 'controls', brief: 'Lists the panel’s controls and the settings of the option table that can change mid-run' },
  { name: 'control', method: 'control', brief: 'Sets one of them (or presses a button) mid-run, as a person would', pass: ['id', 'value'], required: ['id'] },
  { name: 'restart', method: 'restart', brief: 'Starts again from the initial conditions' },
  {
    name: 'save_athdf', method: 'athdf', alone: true,
    brief: 'Writes the state of the gas to an `.athdf` file',
    more: 'Save the state of the gas to an .athdf file (HDF5 in Athena++’s layout; yt reads it). The file carries the setup, so load_athdf continues the run exactly.',
    own: { path: { type: 'string', description: 'File or folder to write to (default: the current folder, under the file’s own name)' } },
  },
  {
    name: 'record', method: 'record',
    brief: 'Records the run from where it is as an MP4 movie to `path`, with a color bar, axes and the status line (with the simulation time): a frame every `every` steps or near every `dt` of code time, until `until` or `frames`',
    more: 'Returns the frame count, movie length and where the file went.',
    pass: ['path', 'dt', 'every', 'until', 'frames', 'fps', 'width', 'height'],
    own: {
      path: { type: 'string', description: 'File or folder to write to (default: the current folder, under the name of the movie)' },
      bitrate_mbps: { type: 'number', exclusiveMinimum: 0, description: 'Megabits per second of movie (default 0.2 bits per pixel per frame: 25 at 1080p and 60 fps)' },
      colorbar: { type: 'boolean', description: 'Default true' },
      axes: { type: 'boolean', description: 'Default true' },
      status: { type: 'boolean', description: 'The status line (grid, simulation time, point masses, mesh). Default true' },
      timeout_s: { type: 'number', description: 'Give up after this much wall time, keeping what was recorded' },
    },
  },
  {
    name: 'load_athdf', method: 'loadAthdf', brief: 'Starts a run from an `.athdf` file, or opens it for viewing only',
    own: {
      path: { type: 'string', description: 'The file to read' },
      view: { type: 'boolean', description: 'Open it only to look at (a run too large to continue here): read in parts, its finest levels merged until it fits the GPU memory budget, never stepped (run refuses). slice, probe, screenshot and view work. The result says the level kept (`view`)' },
    },
    required: ['path'],
  },
  {
    name: 'debris', method: 'debris',
    brief: 'Gives debris energies and the return rate dM/dt, for problems that track them: totals, the surviving core, the first return and the peak. `save_path` writes the full report',
    more: 'Returns totals, the core, the first return and the peak of dM/dt (in days and solar masses per year where the problem has units). save_path writes the full report (bins, dmdt.t and dmdt.rate) as JSON.',
    own: { save_path: { type: 'string', description: 'Write the full report as JSON to this file' } },
  },
  {
    name: 'impact', method: 'impact',
    brief: 'Gives what a giant impact has left: the masses of the planet, the disk around it and the escaping gas, what each is made of, and the moon the disk would gather into',
    more: 'For 3D runs with gravity (the moon-forming-impact preset). Masses in code units (Earth masses in that preset); shares are mass fractions by species, and the mean dye, which tells the two bodies apart.',
  },
  {
    name: 'edit', method: 'tool',
    brief: 'Uses an editing tool on the run, as a person would: `name` (mass, velocity, temperature, obstacle, field, pull, source, sink, star, clear, query) and its `options`',
    pass: ['name', 'options'],
    required: ['name'],
  },
  {
    name: 'session_log', method: 'actions',
    brief: 'Summarizes what has been done to the run since it started (actions by kind, steps), and writes the full session log to `save_path`',
    more: 'Returns a summary: the setup, whether it can be replayed, the actions by kind and the steps they span. save_path writes the full log as JSON, for replay.',
    own: { save_path: { type: 'string', description: 'Write the session log as JSON to this file' } },
  },
  {
    name: 'replay', method: 'replay',
    brief: 'Runs a saved session log (a file written by session_log or the page’s Save session log button) again, step for step',
    own: { path: { type: 'string', description: 'The session log file' } },
    pass: ['path', 'untilSteps'],
    required: ['path'],
  },
  { name: 'errors', method: 'errors', brief: 'Lists what the page has reported going wrong' },
];

/** The schema of one argument of an API method: positional, or a setting of its options. */
function argOf(method: string, name: string): ArgSchema {
  const m = API_METHODS.find((x) => x.name === method);
  if (!m) throw new Error(`the API has no method "${method}"`);
  const found = m.args.find((a) => a.name === name) ?? m.args.find((a) => a.name === 'options')?.properties?.[name];
  if (!found) throw new Error(`the API's ${method} has no argument "${name}"`);
  const { name: _name, optional: _optional, ...schema } = found as ArgSchema & { name?: string; optional?: boolean };
  return schema;
}

/** The tools as an MCP client is told of them (tools/list). */
export function mcpTools() {
  return MCP_TOOLS.map((t) => {
    const summary = API_METHODS.find((m) => m.name === t.method)?.summary;
    if (summary === undefined) throw new Error(`${t.name}: the API has no method "${t.method}"`);
    // Arguments in the order a call reads best: what is passed on, then its own; a tool may place its own first by naming it in `pass`.
    const properties: Record<string, ArgSchema> = {};
    for (const k of t.pass ?? []) properties[k] = t.own?.[k] ?? argOf(t.method, k);
    for (const [k, v] of Object.entries(t.own ?? {})) properties[k] ??= v;
    return {
      name: t.name,
      method: t.method,
      brief: t.brief,
      description: t.alone ? t.more! : t.more ? `${summary} ${t.more}` : summary,
      inputSchema: { type: 'object', properties, required: t.required ?? [], additionalProperties: false },
    };
  });
}

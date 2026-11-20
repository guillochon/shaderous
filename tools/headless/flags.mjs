// The command-line runner's options (cli.mjs), apart from the runner so that the generator of
// the docs can read them without starting a run.

// Every option, written once: this table makes both the parser's options and the help text, so
// an option cannot be read and not documented, or the other way round. F(flags, help, more): a
// flag is 'name' (a switch) or 'name VALUE' (it takes a value); several flags share a line of
// help when they belong together; `many` lets a flag repeat. A { show } entry is a line of help
// alone (a positional argument, or flags listed a second time under another heading).
const F = (flags, help = [], more = {}) => ({ flags: [flags].flat(), help: [help].flat(), ...more });
export const SECTIONS = [
  {
    title: 'Setup (the same keys as a shared link; see docs/api.md)',
    entries: [
      { show: '[setup]', help: ['A query string or link: "problem=sedov&dim=2&n=512"'] },
      F(['problem ID', 'dim 2|3', 'n CELLS', 'amr LEVELS'], 'The preset, and its grid'),
      F('set KEY=VALUE', 'Any other setup key, repeatable: --set gamma=1.4 --set p.mach=3', { many: true }),
      F('setup FILE', ['A setup document (name.shaderous.json, see docs/setups.md) in place of', "--problem: a preset's file, or one saved from the Setup dialog. The", 'keys above then say the grid and what differs from the file']),
    ],
  },
  {
    title: "Stopping (default: the problem's end time)",
    entries: [
      F('until T', 'Run to this time (landed on exactly)'),
      F('steps N', 'Run at most this many steps'),
      F('every DT', 'Write outputs every DT of simulated time (and at the start)'),
      F('every-wall SECONDS', 'Or write outputs every SECONDS of wall time (and at the start)'),
      F('timeout SECONDS', 'Give up on a run after this much wall time'),
      F('wall SECONDS', ['Stop after this much wall time in all (loading included), and write the', 'last outputs then. A file named STOP in the output folder stops the', 'run the same way at its next output']),
    ],
  },
  {
    title: 'Outputs (diag.jsonl and manifest.json are always written)',
    entries: [
      F('out DIR', 'Folder (default: runs/<problem>-<date>)'),
      F('png', 'A screenshot at each output'),
      F(['width PX', 'height PX'], 'Screenshot size (default 1024 square)'),
      F('overlays', 'The color bar, axes and status line (with the time) on each screenshot'),
      F('native', ["Screenshots at the grid's own resolution (a plane: a 2D run, a slice or", 'a projection): one pixel per cell of the finest level, with the color bar,', 'axes and status line beside it. Not with --width or --height']),
      F('view KEY=VALUE', 'View setting for screenshots, repeatable: --view field=3 --view colormap=viridis', { many: true }),
      F('shot NAME:KEY=VALUE,..', ['A screenshot of its own at each output, NNNN-tT-NAME.png: the --view', 'settings with these on top. Repeatable, one per view:', '--shot vol:mode=volume --shot rho:mode=slice,sliceAxis=2'], { many: true }),
      F('slice VAR', 'A slice of this variable at each output (repeatable)', { many: true }),
      F(['slice-axis A', 'slice-pos P', 'slice-res R'], 'Its axis (0 to 2), position (0 to 1) and cells per side'),
      F('slice-format F', ['json (default): NNNN-tT-VAR.json, the whole slice() result; or npy:', "NNNN-tT-VAR.npy, float32 of shape [res, res] for numpy's np.load,", 'with the slice\'s other fields under that output\'s "slices" in', 'manifest.json']),
      F('athdf', 'An .athdf file at each output'),
      F('athdf-every N', 'Or an .athdf file every Nth output and at the last (0: the last only)'),
      F('debris', ['Debris energies and dM/dt at each output, as JSON (problems that track', 'them, such as tidal-disruption); plot with tools/headless/plot_dmdt.py']),
    ],
  },
  {
    title: 'Movie (--mp4 records the whole run as movie.mp4, with a color bar, axes and the status line; not with --every\nor --steps; outputs are written at the start and the end)',
    entries: [
      F('mp4', 'Record the run as an MP4 movie'),
      F('frame-dt DT', ['A frame near every DT of simulated time (no step is shortened for it);', "default a 600th of the problem's dynamical time"]),
      F('frame-every N', 'Or a frame every N steps'),
      F('fps F', 'Playback rate (default 60)'),
      F('bitrate M', ['Megabits per second of movie (default 0.2 bits per pixel', 'per frame: 25 at 1080p and 60 fps)']),
      { show: '--width PX  --height PX', help: ['Movie size (default 1920 x 1080 with --mp4)'] },
      F('no-overlays', 'Leave out the color bar, the axes and the status line'),
    ],
  },
  { title: 'Sweeps', entries: [F('vary KEY=V1,V2,..', 'Values of a setup key, repeatable; every combination is run', { many: true })] },
  {
    title: 'Other',
    entries: [
      F('url URL', 'Use the page served here (default: serve this checkout with Vite)'),
      F('headed', 'Show the browser (if headless Chrome finds no GPU adapter)'),
      F('low-priority', 'Run at below-normal CPU priority (the browser inherits it on Windows)'),
      F('gpu-memory GB', ["GPU memory the simulation may take (the panel's setting; default 2.5 on a", 'desktop): how many blocks the adaptive mesh can hold']),
      F('quiet', 'No progress messages'),
    ],
  },
];

/** The help: each section's entries, a flag's help beside it and carried on under itself. */
export function helpText() {
  const column = 23;
  const lines = [];
  for (const s of SECTIONS) {
    lines.push('', s.title);
    for (const e of s.entries) {
      const left = `  ${e.show ?? e.flags.map((f) => `--${f}`).join('  ')}`;
      const [first, ...rest] = e.help;
      lines.push(first === undefined ? left : left.length < column - 1 ? left.padEnd(column) + first : `${left}   ${first}`);
      for (const more of rest) lines.push(' '.repeat(column) + more);
    }
  }
  return lines.join('\n');
}

/** The parser's options: a flag with a value's name takes text, one without is a switch. */
export const OPTIONS = { help: { type: 'boolean', short: 'h' } };
for (const s of SECTIONS) {
  for (const e of s.entries) {
    for (const f of e.flags ?? []) {
      const [name, value] = f.split(' ');
      OPTIONS[name] = { type: value ? 'string' : 'boolean', ...(e.many ? { multiple: true } : {}) };
    }
  }
}

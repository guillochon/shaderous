// The flow around the blade sections on the GPU (plans/wind-tunnel.md 8.1.6), through the
// command-line runner: for each section and grid, slices of the two velocities and the pressure at
// the grid's own resolution (one sample a cell), every 0.5 to t = 24 (every 1 on 2048^2, to keep
// a run's files near 1 GB), for the means over t = 8 to 24 and the instants.
// The forces come from the GPU test cut-sections, not from these runs: the runner lands on each
// output time with a short step, whose force is not the flow's (8.1.4).
//
//   node tools/cutcell/section_fields.mjs [sections,...] [cellsOnChord,...] [--url URL]
//   (defaults: all five sections; 32,64,128,256)
// Then: node tools/cutcell/section_cells.mjs (the gas shares) and python tools/cutcell/section_plots.py.
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const args = process.argv.slice(2);
const url = args.includes('--url') ? args.splice(args.indexOf('--url'), 2)[1] : null;
const sections = (args[0] ?? 'section-tip,section-075,section-round,naca0020,naca0012').split(',');
const grid = (args[1] ?? '32,64,128,256').split(',').map(Number);
const BOX = 8;

for (const cells of grid)
  for (const s of sections) {
    const n = BOX * cells;
    const out = `runs/sections/${s}_${cells}`;
    if (existsSync(`${out}/manifest.json`)) {
      console.log(`${out}: done before, skipped`);
      continue;
    }
    const cmd = [
      'tools/headless/cli.mjs', 'run', '--setup', `tools/cutcell/sections/${s}.shaderous.json`, '--n', String(n),
      '--until', '24', '--every', n > 1024 ? '1' : '0.5', '--set', 'cfl=0.4',
      '--slice', 'vel1', '--slice', 'vel2', '--slice', 'press', '--slice-format', 'npy', '--slice-res', String(n),
      '--low-priority', '--quiet', '--out', out,
      ...(url ? ['--url', url] : []),
    ];
    console.log(`${out}: n = ${n}`);
    const t0 = Date.now();
    const r = spawnSync(process.execPath, cmd, { stdio: 'inherit' });
    console.log(`${out}: exit ${r.status} in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
  }

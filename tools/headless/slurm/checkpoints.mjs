// The checkpoints of a run split over several jobs (segments: RUN_DIR/seg-*/), as the runner's
// manifests list them. A file is listed only once it is written whole, so a job killed while
// writing one leaves a file here that is never picked.
//   node checkpoints.mjs latest RUN_DIR       prints the path of the newest checkpoint (or nothing)
//   node checkpoints.mjs prune RUN_DIR KEEP   deletes all but the newest KEEP
//   node checkpoints.mjs stopped RUN_DIR      prints why the newest finished segment stopped
import { existsSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const [cmd, dir, keepArg] = process.argv.slice(2);
const segments = existsSync(dir) ? readdirSync(dir).filter((d) => d.startsWith('seg-')).map((d) => path.join(dir, d)) : [];
const manifests = segments
  .map((d) => path.join(d, 'manifest.json'))
  .filter((f) => existsSync(f))
  .map((f) => ({ dir: path.dirname(f), m: JSON.parse(readFileSync(f, 'utf8')) }));
// Every listed checkpoint with its simulated time, oldest first.
const listed = manifests
  .flatMap(({ dir: d, m }) => (m.outputs ?? []).flatMap((o) => o.files.filter((f) => f.endsWith('.athdf')).map((f) => ({ file: path.join(d, f), t: o.t }))))
  .filter((c) => existsSync(c.file))
  .sort((a, b) => a.t - b.t);

if (cmd === 'latest') {
  if (listed.length) console.log(listed.at(-1).file);
} else if (cmd === 'prune') {
  const keep = Math.max(1, Number(keepArg ?? 2));
  for (const c of listed.slice(0, Math.max(0, listed.length - keep))) {
    rmSync(c.file);
    console.log(`removed ${c.file} (t = ${c.t})`);
  }
} else if (cmd === 'stopped') {
  const done = manifests.filter(({ m }) => m.finished).sort((a, b) => a.m.finished.localeCompare(b.m.finished));
  if (done.length) console.log(done.at(-1).m.stopped);
} else {
  console.error('usage: checkpoints.mjs latest|prune|stopped RUN_DIR [KEEP]');
  process.exit(1);
}

// The state of one fill's runs in a sweep (dwd-sweep.sbatch): the fill's folder and the folders
// below it (resume-*/, switched/), as the runner's manifests list them. A file is listed only
// once it is written whole, so a run killed while writing one leaves a file here that is never
// picked.
//   node sweep-state.mjs latest OUT        the newest checkpoint (.athdf) of any of them, or nothing
//   node sweep-state.mjs stopped DIR       why the run in DIR stopped (its manifest), or nothing
//   node sweep-state.mjs last DIR          the last checkpoint the run in DIR wrote, or nothing
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const [cmd, dir] = process.argv.slice(2);
const manifest = (d) => {
  const f = path.join(d, 'manifest.json');
  return existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')) : null;
};
const checkpoints = (d) =>
  (manifest(d)?.outputs ?? [])
    .flatMap((o) => o.files.filter((f) => f.endsWith('.athdf')).map((f) => ({ file: path.join(d, f), t: o.t })))
    .filter((c) => existsSync(c.file));

if (cmd === 'latest') {
  const dirs = existsSync(dir) ? [dir, ...readdirSync(dir).map((e) => path.join(dir, e)).filter((d) => statSync(d).isDirectory())] : [];
  const all = dirs.flatMap(checkpoints).sort((a, b) => a.t - b.t);
  if (all.length) console.log(all.at(-1).file);
} else if (cmd === 'stopped') {
  const s = manifest(dir)?.stopped;
  if (s) console.log(s);
} else if (cmd === 'last') {
  const c = checkpoints(dir);
  if (c.length) console.log(c.at(-1).file);
} else {
  console.error('usage: sweep-state.mjs latest|stopped|last DIR');
  process.exit(1);
}

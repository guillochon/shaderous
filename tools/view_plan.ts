// How a checkpoint would be viewed: its blocks per level and, for a few atlas sizes, the level
// its blocks would be merged to (io/view.ts). With --read, also time reading it merged.
//   npx tsx tools/view_plan.ts runs/x/0010.athdf [--read LEVEL]
import h5wasm from 'h5wasm/node';
import { library, openAthdf, type H5 } from '../src/io/athdf';
import { countAtCap, readForView } from '../src/io/view';

const file = process.argv[2];
if (!file) throw new Error('usage: tools/view_plan.ts FILE [--read LEVEL]');
const lib = await library(h5wasm as unknown as H5);
const r = openAthdf(lib, file);
const head = r.header();
const per: Record<number, number> = {};
for (const l of head.levels) per[l] = (per[l] ?? 0) + 1;
console.log(`${file}: ${r.blocks} blocks of ${head.blockSize.join('x')}, root ${head.rootSize.join('x')}, variables ${r.variables.join(' ')}${r.hasFaces ? ', faces' : ''}`);
console.log('leaves per level', per);
const top = Math.max(...Object.keys(per).map(Number));
for (let cap = top; cap >= 0; cap--) {
  const c = countAtCap(head, cap);
  console.log(`cap ${cap}: ${c.leaves} leaves, ${c.nodes} tree blocks`);
}
const at = process.argv.indexOf('--read');
if (at > 0) {
  const cap = Number(process.argv[at + 1]);
  const t0 = performance.now();
  const s = readForView(r, cap, { progress: (f) => process.stdout.write(`\r${(100 * f).toFixed(0)}%`) });
  console.log(`\nread merged to level ${cap}: ${s.levels.length} blocks in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}
r.close();

// The gas share of each cell and face of a blade section's grid, as the GPU makes them
// (hydro/cut.ts cutGeometry, on the CPU from the body's distances), for the plots of
// plans/wind-tunnel.md 8.1.6: written to runs/sections/<section>_<cells>/cut.npy, float32 of
// shape [n, n, 4] (the gas share phi, the upper x face's open share, the upper y face's, the
// upper z face's: 1 in 2D), and wall.npy, [n, n] (each cell center's distance to the wall,
// positive in the gas), row j (y) first.
// It imports the app's TypeScript, so it runs bundled (as tools/cutcell/section_cells.mjs does):
//   node tools/cutcell/section_cells.mjs [sections,...] [cellsOnChord,...]
import { mkdirSync, writeFileSync } from 'node:fs';
import { cutGeometry } from '../../src/hydro/cut';
import { compileSetup } from '../../src/setup/compile';
import { sectionSpec, SECTIONS } from '../../src/validation/cut';

/** A float32 .npy file of the given shape. */
function npy(path: string, data: Float32Array, shape: number[]) {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const pad = 64 - ((10 + header.length + 1) % 64);
  header += ' '.repeat(pad % 64) + '\n';
  const h = Buffer.alloc(10);
  h.write('\x93NUMPY', 0, 'latin1');
  h[6] = 1;
  h[7] = 0;
  h.writeUInt16LE(header.length, 8);
  writeFileSync(path, Buffer.concat([h, Buffer.from(header, 'latin1'), Buffer.from(data.buffer, data.byteOffset, data.byteLength)]));
}

export function main(argv: string[]) {
  const sections = (argv[0] ?? SECTIONS.join(',')).split(',');
  const grid = (argv[1] ?? '32,64,128,256').split(',').map(Number);
  for (const cells of grid)
    for (const s of sections) {
      const spec = sectionSpec(s, cells);
      const box = spec.domain!.box as number, n = spec.domain!.n as number;
      const { problem } = compileSetup(spec);
      const t0 = Date.now();
      const g = cutGeometry(problem.obstacleDistances!(n, 2), [n, n, 1], box / n, [-box / 2, -box / 2, 0]);
      const dir = `runs/sections/${s}_${cells}`;
      mkdirSync(dir, { recursive: true });
      npy(`${dir}/cut.npy`, g.cut, [n, n, 4]);
      npy(`${dir}/wall.npy`, g.wall.filter((_, k) => k % 4 === 0), [n, n]);
      writeFileSync(`${dir}/cut.json`, JSON.stringify({ section: s, cells, n, box, lo: [-box / 2, -box / 2], dx: box / n, cutCells: g.cutCells }));
      console.log(`${dir}/cut.npy: ${g.cutCells} cut cells, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    }
}

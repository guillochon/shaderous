// Runs tools/cutcell/section_cells.ts, which imports the app's TypeScript: bundled first by Vite
// (as the app's build does) into a temporary folder, then run with node.
//   node tools/cutcell/section_cells.mjs [sections,...] [cellsOnChord,...]
import { build } from 'vite';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const out = mkdtempSync(join(tmpdir(), 'section-cells-'));
try {
  await build({ logLevel: 'warn', configFile: false, build: { ssr: 'tools/cutcell/section_cells.ts', outDir: out, emptyOutDir: true, rollupOptions: { output: { entryFileNames: 'section_cells.mjs' } } } });
  const m = await import(pathToFileURL(join(out, 'section_cells.mjs')).href);
  m.main(process.argv.slice(2));
} finally {
  rmSync(out, { recursive: true, force: true });
}

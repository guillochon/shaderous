// Files made from the tables in src: the schema the headless tools read, and (between markers)
// the tables of the docs. `npm run gen` writes them; this test fails when one is out of date.
import { describe, expect, it } from 'vitest';
import { fillGenerated } from '../../src/schema/docs';
import { schemaText } from '../../src/schema/schema';
import { setupJsonSchema } from '../../src/setup/codec';

// Node's own modules, without their type package (the project does not install it).
const fs: { readFileSync(u: URL, enc: string): string; writeFileSync(u: URL, text: string): void; existsSync(u: URL): boolean } = await import(/* @vite-ignore */ 'node:fs' as string);
declare const process: { env: Record<string, string | undefined> };
const writing = !!process.env.SHADEROUS_GEN;
const root = new URL('../../', import.meta.url);
const lf = (text: string) => text.replace(/\r\n/g, '\n');

/** Each generated file, by its path in the checkout: what it should hold, given what it holds now. */
const FILES: Record<string, (now: string) => string> = {
  'tools/headless/schema.json': () => schemaText(),
  'docs/api.md': fillGenerated,
  'docs/setups.md': fillGenerated,
  'docs/headless.md': fillGenerated,
  '.claude/skills/shaderous-sim/SKILL.md': fillGenerated,
  'plugins/shaderous/skills/simulate/SKILL.md': fillGenerated,
  'public/shaderous-setup.schema.json': () => `${JSON.stringify(setupJsonSchema(), null, 1)}\n`,
};

describe('generated files', () => {
  it.each(Object.keys(FILES))('%s is up to date (else: npm run gen)', (path) => {
    const url = new URL(path, root);
    const now = fs.existsSync(url) ? lf(fs.readFileSync(url, 'utf8')) : '';
    const want = FILES[path](now);
    if (writing && want !== now) fs.writeFileSync(url, want);
    else expect(now === want, `${path} is out of date: run "npm run gen" and commit the result`).toBe(true);
  });
});

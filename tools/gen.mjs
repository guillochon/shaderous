// npm run gen: write the files made from the tables in src (tools/headless/schema.json, the
// generated tables of the docs). The work is in tests/unit/generated.test.ts, run through
// Vitest so that it can import the page's TypeScript; the same test, run plainly, checks that
// the files are up to date.
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const r = spawnSync(process.execPath, ['node_modules/vitest/vitest.mjs', 'run', 'tests/unit/generated.test.ts'], { cwd: root, stdio: 'inherit', env: { ...process.env, SHADEROUS_GEN: '1' } });
process.exit(r.status ?? 1);

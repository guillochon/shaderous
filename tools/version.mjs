#!/usr/bin/env node
// Set the release version everywhere it is written: package.json and package-lock.json (the npm
// package), server.json (the MCP Registry listing, twice) and the Claude Code plugin's manifest.
// The release workflow (.github/workflows/npm.yml) refuses a tag that any of them disagrees with.
//
//   node tools/version.mjs 0.1.2      set it
//   node tools/version.mjs            print each file's version (exit 1 if they differ)
import { readFileSync, writeFileSync } from 'node:fs';

const root = new URL('..', import.meta.url);
const FILES = {
  'package.json': (j) => [j, 'version'],
  'package-lock.json': (j) => [j, 'version', j.packages[''], 'version'],
  'server.json': (j) => [j, 'version', ...j.packages.flatMap((p) => (p.identifier === 'shaderous' ? [p, 'version'] : []))],
  'plugins/shaderous/.claude-plugin/plugin.json': (j) => [j, 'version'],
};

const next = process.argv[2];
if (next !== undefined && !/^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/.test(next)) {
  console.error(`not a version: ${next}`);
  process.exit(1);
}
const seen = new Set();
for (const [file, slots] of Object.entries(FILES)) {
  const url = new URL(file, root);
  const text = readFileSync(url, 'utf8');
  const json = JSON.parse(text);
  const s = slots(json);
  for (let i = 0; i < s.length; i += 2) {
    seen.add(s[i][s[i + 1]]);
    if (next) s[i][s[i + 1]] = next;
  }
  const versions = [...new Set(Array.from({ length: s.length / 2 }, (_, k) => s[2 * k][s[2 * k + 1]]))];
  if (next) {
    const indent = text.match(/^\{\n(\s+)/)?.[1] ?? '  ';
    writeFileSync(url, JSON.stringify(json, null, indent) + '\n');
  }
  console.log(`${file}: ${versions.join(', ')}`);
}
if (!next && seen.size > 1) {
  console.error(`versions differ: ${[...seen].join(', ')}`);
  process.exit(1);
}

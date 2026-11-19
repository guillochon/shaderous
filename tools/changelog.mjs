#!/usr/bin/env node
// A release's notes, from CHANGELOG.md: the section under "## X.Y.Z" (a date may follow the
// version), up to the next "## ". The release workflow (.github/workflows/npm.yml) refuses a tag
// whose version has no section here, and makes the section the body of the GitHub release;
// CHANGELOG.md itself ships in the npm package.
//
//   node tools/changelog.mjs 0.3.0    print the notes of 0.3.0 (exit 1 if there are none)
//   node tools/changelog.mjs          the same for package.json's version
import { readFileSync } from 'node:fs';

const root = new URL('..', import.meta.url);

/** The body of a version's section, trimmed, or undefined where the changelog has none. */
export function notes(changelog, version) {
  const lines = changelog.split('\n');
  const heading = (l) => /^## +\[?v?([0-9][^\]\s]*)\]?/.exec(l)?.[1];
  const start = lines.findIndex((l) => heading(l) === version);
  if (start < 0) return undefined;
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((l) => /^## /.test(l));
  return (end < 0 ? rest : rest.slice(0, end)).join('\n').trim();
}

if (import.meta.url === new URL(process.argv[1], 'file:').href) {
  const version = process.argv[2] ?? JSON.parse(readFileSync(new URL('package.json', root), 'utf8')).version;
  const body = notes(readFileSync(new URL('CHANGELOG.md', root), 'utf8'), version);
  if (!body) {
    console.error(`CHANGELOG.md has no notes for ${version}: add a section "## ${version} (YYYY-MM-DD)" before releasing`);
    process.exit(1);
  }
  console.log(body);
}

// One source for every list that describes the interface: the option table, the view's
// settings and the link's keys. These tests fail when a copy of one of them (in the document
// codec, the docs or the headless tools) says something else.
import { describe, expect, it } from 'vitest';
import { readSetup } from '../../src/setup/codec';
import { flatten, OPTIONS, PART_OPTIONS, type Option } from '../../src/setup/options';
import { emptySetup, setPath, type SetupSpec } from '../../src/setup/spec';
import { defaultView } from '../../src/render/renderer';
import { decodeLink, type LinkReport } from '../../src/ui/link';
import apiDoc from '../../docs/api.md?raw';

const selects = (options: Option[]) => flatten(options).filter((o) => o.control.type === 'select');
const choicesOf = (o: Option) => (o.control.type === 'select' ? o.control.choices.map((c) => c.value) : []);
/** A value of the same type as the option's choices, which is not one of them. */
const stranger = (o: Option) => (typeof choicesOf(o)[0] === 'number' ? 97 : 'no-such-choice');

/** Whether a document with this value at this path reads, and keeps the value. */
function reads(build: (v: unknown) => SetupSpec, path: string, v: unknown): boolean {
  const { spec } = readSetup(JSON.parse(JSON.stringify(build(v))));
  if (!spec) return false;
  const kept = path.split('.').reduce<unknown>((at, k) => (at as Record<string, unknown> | undefined)?.[k], spec);
  return kept === v;
}

/** The backticked names in the first column of the table that follows a heading. */
function tableKeys(doc: string, heading: string): string[] {
  const from = doc.indexOf(heading);
  expect(from, heading).toBeGreaterThanOrEqual(0);
  const rows = doc.slice(from + heading.length).replace(/\r/g, '').split('\n');
  const start = rows.findIndex((r) => r.startsWith('|'));
  const out: string[] = [];
  for (let i = start; i < rows.length && rows[i].startsWith('|'); i++) {
    const cell = rows[i].split('|')[1];
    for (const m of cell.matchAll(/`([^`]+)`/g)) out.push(m[1]);
  }
  return out;
}

describe('the document codec and the option table', () => {
  // physics.species and the units are read by rules of their own.
  const general = selects(OPTIONS).filter((o) => !/^domain\.units\./.test(o.path));
  it.each(general.map((o) => [o.path, o] as const))('%s: the codec takes each choice and no other', (path, o) => {
    const build = (v: unknown) => {
      const s = emptySetup();
      setPath(s, path, v);
      return s;
    };
    for (const v of choicesOf(o)) expect(reads(build, path, v), `${path} = ${v}`).toBe(true);
    expect(reads(build, path, stranger(o)), `${path} = a value that is no choice`).toBe(false);
  });

  const ofParts = Object.entries(PART_OPTIONS).flatMap(([kind, k]) => selects(k.fields).map((o) => [`${kind}.${o.path}`, kind, o] as const));
  it.each(ofParts)('%s: the codec takes each choice and no other', (_, kind, o) => {
    if (kind === 'flow' && o.path === 'name') return; // a flow's name picks its own fields
    const build = (v: unknown) => {
      const s = emptySetup();
      s.parts = [{ kind } as never];
      setPath(s, `parts.0.${o.path}`, v);
      return s;
    };
    for (const v of choicesOf(o)) expect(reads(build, `parts.0.${o.path}`, v), `${o.path} = ${v}`).toBe(true);
    expect(reads(build, `parts.0.${o.path}`, stranger(o)), `${o.path} = a value that is no choice`).toBe(false);
  });
});

describe('link keys', () => {
  it('no two options share one', () => {
    const keys = [...flatten(OPTIONS), ...Object.values(PART_OPTIONS).flatMap((k) => flatten(k.fields))].flatMap((o) => (o.linkKey ? [o.linkKey] : []));
    expect(keys.filter((k, i) => keys.indexOf(k) !== i)).toEqual([]);
  });

  it('every key the API reference lists is one a link reads', () => {
    const keys = tableKeys(apiDoc, 'The main keys are listed below.').filter((k) => k !== 'p.<id>');
    expect(keys.length).toBeGreaterThan(30);
    const q = new URLSearchParams(keys.map((k) => [k, '1']));
    q.set('problem', 'kelvin-helmholtz');
    const report: LinkReport = { ignored: [] };
    decodeLink(q, report);
    // A key the link does not know is reported by its name alone; a value it turns down, as key=value.
    expect(report.ignored.filter((k) => !k.includes('='))).toEqual([]);
  });
});

describe('the view', () => {
  it('the API reference lists each of its settings, and no other', () => {
    const listed = tableKeys(apiDoc, '### `view(changes)');
    expect([...listed].sort()).toEqual([...Object.keys(defaultView()), 'colormap', 'autoRange'].sort());
  });
});

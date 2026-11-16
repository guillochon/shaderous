// One source for every list that describes the interface: the option table, the view's
// settings and the link's keys. These tests fail when a copy of one of them (in the document
// codec, the docs or the headless tools) says something else.
import { describe, expect, it } from 'vitest';
import { PRESET_FILES, PRESET_IDS, presetSpec } from '../../src/presets/index';
import { formatSetup, readSetup, SETUP_SCHEMA_URL, setupJsonSchema } from '../../src/setup/codec';
import { flatten, OPTIONS, PART_OPTIONS, type Option } from '../../src/setup/options';
import { emptySetup, setPath, type SetupSpec } from '../../src/setup/spec';
import { defaultView } from '../../src/render/renderer';
import { VIEW_EXTRAS, VIEW_OPTIONS } from '../../src/render/view_options';
import { decodeLink, linkKeys, type LinkReport } from '../../src/ui/link';
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
    const keys = tableKeys(apiDoc, '<!-- gen:link-keys -->').filter((k) => k !== 'p.<id>');
    expect(keys.length).toBeGreaterThan(30);
    const q = new URLSearchParams(keys.map((k) => [k, '1']));
    q.set('problem', 'kelvin-helmholtz');
    const report: LinkReport = { ignored: [] };
    decodeLink(q, report);
    // A key the link does not know is reported by its name alone; a value it turns down, as key=value.
    expect(report.ignored.filter((k) => !k.includes('='))).toEqual([]);
  });
});

describe('the list of link keys', () => {
  it('names only keys a link reads, each once', () => {
    const keys = linkKeys().map((k) => k.key);
    expect(new Set(keys).size).toBe(keys.length);
    const q = new URLSearchParams(keys.filter((k) => k !== 'p.<id>').map((k) => [k, '1']));
    q.set('problem', 'kelvin-helmholtz');
    const report: LinkReport = { ignored: [] };
    decodeLink(q, report);
    expect(report.ignored.filter((k) => !k.includes('='))).toEqual([]);
  });
});

describe('the view', () => {
  it('the API reference lists each of its settings, and no other', () => {
    const listed = tableKeys(apiDoc, '<!-- gen:view -->');
    expect([...listed].sort()).toEqual([...VIEW_OPTIONS, ...VIEW_EXTRAS].map((o) => o.key).sort());
  });

  it('its table is the state of the renderer, in order, and no two settings share a link key', () => {
    expect(VIEW_OPTIONS.map((o) => o.key)).toEqual(Object.keys(defaultView()));
    const links = [...VIEW_OPTIONS, ...VIEW_EXTRAS].map((o) => o.link).filter(Boolean);
    expect(new Set(links).size).toBe(links.length);
    const general = [...flatten(OPTIONS), ...Object.values(PART_OPTIONS).flatMap((k) => flatten(k.fields))].flatMap((o) => (o.linkKey ? [o.linkKey] : []));
    expect(links.filter((k) => general.includes(k))).toEqual([]);
  });

  it('a link carries every setting there and back', () => {
    const q = new URLSearchParams({ problem: 'kelvin-helmholtz', mode: 'slice', field: '2', axis: '1', slice: '0.25', rhocut: '0.5', opacity: '80', yaw: '9', pitch: '1', dist: '3', tx: '0.1', ty: '0.2', tz: '0.3', zoom: '4', cx: '0.6', cy: '0.7', ncont: '12', isoa: '0.25', nflines: '300.4', log: '0', blocks: '0', contours: '1', fog: '0', light: '1', box: '0', flines: '1' });
    const report: LinkReport = { ignored: [] };
    expect(decodeLink(q, report).view).toEqual({ mode: 'slice', field: 2, sliceAxis: 1, slicePos: 0.25, minRho: 0.5, opacity: 80, yaw: 9, pitch: 1, dist: 3, tx: 0.1, ty: 0.2, tz: 0.3, zoom: 4, cx: 0.6, cy: 0.7, nContours: 12, surfaceOpacity: 0.25, fieldLineCount: 300, log: false, showBlocks: false, contours: true, fog: false, light: true, box: false, fieldLines: true });
    expect(report.ignored).toEqual([]);
    // Out of range: held, and said so.
    const held = decodeLink(new URLSearchParams({ problem: 'kelvin-helmholtz', zoom: '1000', axis: '7', field: '99', ncont: '0' }), report).view;
    expect(held).toMatchObject({ zoom: 256, sliceAxis: 2, nContours: 1 });
    expect(report.ignored.length).toBe(4);
  });
});

// As much of a JSON Schema validator as the generated schema uses: what is wrong with a value, as paths.
type Js = Record<string, any>;
function invalid(v: unknown, s: Js, path = ''): string[] {
  const out: string[] = [];
  const kind = v === null ? 'null' : Array.isArray(v) ? 'array' : Number.isInteger(v) ? 'integer' : typeof v;
  if (s.const !== undefined && v !== s.const) return [`${path}: not ${JSON.stringify(s.const)}`];
  if (s.enum && !s.enum.includes(v)) return [`${path}: ${JSON.stringify(v)} is not a choice`];
  for (const key of ['anyOf', 'oneOf'] as const) {
    if (!s[key]) continue;
    const fits = s[key].filter((alt: Js) => invalid(v, alt, path).length === 0).length;
    if (key === 'anyOf' ? fits < 1 : fits !== 1) return [`${path}: fits ${fits} of ${key}`, ...(s[key].length < 40 ? [] : s[key].flatMap((alt: Js) => (alt.properties?.kind?.const === (v as Js)?.kind && (alt.properties?.name?.const ?? (v as Js)?.name) === (v as Js)?.name ? invalid(v, alt, path) : [])))];
  }
  if (s.type && !(s.type === kind || (s.type === 'number' && kind === 'integer'))) return [`${path}: ${kind}, not ${s.type}`];
  if (typeof v === 'number' && ((s.minimum !== undefined && v < s.minimum) || (s.maximum !== undefined && v > s.maximum))) out.push(`${path}: ${v} out of range`);
  if (Array.isArray(v)) {
    if ((s.minItems !== undefined && v.length < s.minItems) || (s.maxItems !== undefined && v.length > s.maxItems)) out.push(`${path}: ${v.length} entries`);
    if (s.items) v.forEach((x, i) => out.push(...invalid(x, s.items, `${path}.${i}`)));
  } else if (kind === 'object') {
    for (const k of s.required ?? []) if (!(k in (v as Js))) out.push(`${path}: lacks ${k}`);
    for (const [k, x] of Object.entries(v as Js)) {
      const sub = s.properties?.[k] ?? (typeof s.additionalProperties === 'object' ? s.additionalProperties : undefined);
      if (sub) out.push(...invalid(x, sub, path ? `${path}.${k}` : k));
      else if (s.additionalProperties === false) out.push(`${path ? `${path}.` : ''}${k}: not a key`);
    }
  }
  return out;
}

describe('the JSON Schema of a setup document', () => {
  const schema = setupJsonSchema();

  it.each(PRESET_IDS)('takes the preset %s, as its file and whole', (id) => {
    expect(invalid(PRESET_FILES[id], schema)).toEqual([]);
    expect(invalid(JSON.parse(formatSetup(presetSpec(id)!)), schema)).toEqual([]);
  });

  it('turns down what the codec turns down', () => {
    const doc = (change: (s: Js) => void) => {
      const s = JSON.parse(JSON.stringify(presetSpec('binary')));
      change(s);
      return s;
    };
    for (const change of [(s: Js) => (s.physics.riemann = 'exact'), (s: Js) => (s.parts[0].kind = 'planet'), (s: Js) => (s.domain.ndim = 4), (s: Js) => (s.physics.cfl = 'small'), (s: Js) => delete s.format]) {
      expect(invalid(doc(change), schema)).not.toEqual([]);
      expect(readSetup(doc(change)).spec).toBeNull();
    }
    // A key this build does not know: dropped with a note when read, an error for an editor.
    expect(invalid(doc((s) => (s.physics.magic = 1)), schema)).toEqual(['physics.magic: not a key']);
  });

  it('a written file names it, and reads back without a note', () => {
    const text = formatSetup(presetSpec('sedov')!);
    expect(JSON.parse(text).$schema).toBe(SETUP_SCHEMA_URL);
    const { spec, report } = readSetup(text);
    expect(report.notes).toEqual([]);
    expect(spec).toEqual(presetSpec('sedov'));
  });
});

// The option table, the codec and the compiler of setup documents (src/setup).
import { describe, expect, it } from 'vitest';
import { PRESET_FILES, PRESET_IDS, presetSpec } from '../../src/presets/index';
import { decodeSetupParam, encodeSetupParam, formatSetup, readSetup } from '../../src/setup/codec';
import { compileSetup, paramFeatures } from '../../src/setup/compile';
import { BC_CHOICES, FACTS, flatten, flowOptions, OPTIONS, optionAt, PART_OPTIONS, ruleIssues, type Option, type Rule } from '../../src/setup/options';
import { FLOWS } from '../../src/setup/parts/flow';
import { emptySetup, featuredOf, featuredPaths, getPath, setPath, type SetupSpec } from '../../src/setup/spec';

const find = (id: string) => PRESET_FILES[id];

function rulesOf(o: Option): Rule[] {
  const own = [...(o.requires ?? []), ...(o.excludes ?? [])];
  const choices = o.control.type === 'select' ? o.control.choices.flatMap((c) => [...(c.requires ?? []), ...(c.excludes ?? [])]) : [];
  return [...own, ...choices];
}

describe('the option table', () => {
  const all = [...flatten(OPTIONS), ...Object.values(PART_OPTIONS).flatMap((p) => flatten(p.fields)), ...Object.keys(FLOWS).flatMap(flowOptions)];

  it('names only facts that exist in its rules', () => {
    const rules = [...all.flatMap(rulesOf), ...Object.values(PART_OPTIONS).flatMap((p) => [...(p.requires ?? []), ...(p.excludes ?? [])]), ...BC_CHOICES.flatMap((c) => [...(c.requires ?? []), ...(c.excludes ?? [])])];
    expect(rules.length).toBeGreaterThan(20);
    for (const r of rules) expect(FACTS[r.fact], r.fact).toBeDefined();
  });

  it('lists each general option once, with a label and a tip', () => {
    const paths = flatten(OPTIONS).map((o) => o.path);
    expect(new Set(paths).size).toBe(paths.length);
    for (const o of all) expect(o.label && o.tip, o.path).toBeTruthy();
  });

  it('gives each link key to one option', () => {
    const keys = all.flatMap((o) => (o.linkKey ? [o.linkKey] : []));
    expect(new Set(keys).size).toBe(keys.length);
  });

  it('has defaults its own controls accept, and the codec with them', () => {
    const spec = emptySetup();
    for (const o of flatten(OPTIONS)) {
      if (o.default === undefined || o.default === null || o.control.type === 'group') continue;
      if (o.control.type === 'number') {
        expect(o.default as number, o.path).toBeGreaterThanOrEqual(o.control.min ?? -Infinity);
        expect(o.default as number, o.path).toBeLessThanOrEqual(o.control.max ?? Infinity);
      }
      if (o.control.type === 'select') expect(o.control.choices.map((c) => c.value), o.path).toContain(o.default);
      if (!o.path.startsWith('domain.units.') && !o.path.startsWith('physics.burn.')) setPath(spec, o.path, o.default);
    }
    const { report } = readSetup(JSON.stringify(spec));
    expect(report.error).toBeUndefined();
    expect(report.notes).toEqual([]);
  });

  it('is satisfied by every preset', () => {
    for (const id of PRESET_IDS) expect(ruleIssues(presetSpec(id)!), id).toEqual([]);
  });

  it('says why an option is not available', () => {
    const spec: SetupSpec = { ...emptySetup(), domain: { ndim: 2 }, physics: { gravity: 'isolated', eos: 'barotropic', cooling: 'ism' } };
    const text = ruleIssues(spec).map((i) => i.text).join('\n');
    expect(text).toMatch(/Gravity: Self, isolated: needs Dimensions: 3D/);
    expect(text).toMatch(/Cooling function.*needs An equation of state with an energy/);
  });

  it('finds the option a featured path leads to, in every preset', () => {
    for (const id of PRESET_IDS) {
      const spec = presetSpec(id)!;
      for (const f of (spec.featured ?? []).map(featuredOf))
        for (const path of featuredPaths(f)) {
          expect(optionAt(spec, path), `${id}: ${path}`).toBeDefined();
          expect(getPath(spec, path), `${id}: ${path}`).toBeDefined();
        }
    }
  });
});

describe('the setup codec', () => {
  it('reads every preset file without a note, and writes it back the same', () => {
    for (const id of PRESET_IDS) {
      const { spec, report } = readSetup(JSON.stringify(PRESET_FILES[id]), find);
      expect(report.error, id).toBeUndefined();
      expect(report.notes, id).toEqual([]);
      expect(spec, id).toEqual(presetSpec(id));
      const again = readSetup(formatSetup(spec!));
      expect(again.spec, id).toEqual(spec);
      expect(again.report.notes, id).toEqual([]);
    }
  });

  it('resolves extends, the child over the parent, lists replaced whole', () => {
    const child = presetSpec('polytrope-dipole')!;
    expect(child.extends).toBeUndefined();
    expect(child.physics?.gravity).toBe('isolated');
    expect(child.name).toBe('Polytrope with a dipole field');
    expect(child.featured).toHaveLength(5);
    expect(readSetup({ ...emptySetup(), extends: 'no-such' }, find).report.error).toMatch(/does not have/);
  });

  it('drops what it does not know, holds values to their range, and says so', () => {
    const { spec, report } = readSetup({ ...emptySetup(), surprise: 1, physics: { cfl: 7, magic: true }, parts: [{ kind: 'mass', gm: 1, hat: 'red' }, { kind: 'flow', name: 'kelvin-helmholtz', extra: 2 }] });
    expect(report.error).toBeUndefined();
    expect(spec!.physics!.cfl).toBe(1);
    const notes = report.notes.join('\n');
    for (const want of [/surprise: dropped/, /physics.cfl: 7 held to 1/, /physics.magic: dropped/, /parts.0.hat: dropped/, /parts.1.extra: dropped/]) expect(notes).toMatch(want);
    expect(JSON.stringify(spec)).not.toMatch(/surprise|magic|hat|extra/);
  });

  it('ends malformed and hostile input in a clean report', () => {
    const bad: unknown[] = [
      'not json', '[]', 'null', '{"format":"other","version":1}', { format: 'shaderous-setup' }, { ...emptySetup(), physics: { cfl: 'fast' } },
      { ...emptySetup(), parts: 'many' }, { ...emptySetup(), parts: [{ kind: 'bomb' }] }, { ...emptySetup(), parts: [{ kind: 'flow', name: 'no-such-flow' }] },
      { ...emptySetup(), parts: Array.from({ length: 5000 }, () => ({ kind: 'mass', gm: 1 })) },
      { ...emptySetup(), domain: { box: Number.NaN } }, { ...emptySetup(), name: 'x'.repeat(100000) },
      { ...emptySetup(), parts: [{ kind: 'mass', gm: 1, id: 'a' }, { kind: 'mass', gm: 1, id: 'a' }] },
      { ...emptySetup(), parts: [{ kind: 'mass', gm: 1, pos: [1, 2] }] },
    ];
    let deep: unknown = 1;
    for (let i = 0; i < 200; i++) deep = { eosParams: deep };
    bad.push({ ...emptySetup(), physics: deep });
    for (const input of bad) expect(readSetup(input).report.error, JSON.stringify(input)?.slice(0, 80)).toBeTruthy();
    const proto = JSON.parse('{"format":"shaderous-setup","version":1,"__proto__":{"polluted":1},"physics":{"__proto__":{"polluted":1}}}');
    expect(readSetup(proto).report.error).toBeUndefined();
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it('warns about a newer version and still reads what it knows', () => {
    const { spec, report } = readSetup({ ...emptySetup(), version: 99, future: {}, domain: { box: 2 } });
    expect(spec!.domain!.box).toBe(2);
    expect(report.notes[0]).toMatch(/newer build/);
  });

  it('packs a setup into a link and back', async () => {
    for (const id of ['binary', 'tidal-disruption-random', 'forward-step']) {
      const spec = presetSpec(id)!;
      const param = await encodeSetupParam(spec);
      expect(param).toMatch(/^[A-Za-z0-9_-]+$/);
      expect(param.length).toBeLessThan(8000);
      expect(readSetup(await decodeSetupParam(param)).spec).toEqual(spec);
    }
    await expect(decodeSetupParam('AAAA')).rejects.toThrow();
  });
});

describe('the setup compiler', () => {
  it('compiles the same document to the same problem twice', () => {
    for (const id of ['binary', 'sedov', 'polytrope']) {
      const [a, b] = [compileSetup(presetSpec(id)!), compileSetup(JSON.parse(JSON.stringify(presetSpec(id))) as SetupSpec)];
      expect(JSON.stringify(a.problem)).toBe(JSON.stringify(b.problem));
      expect(a.problem.damping?.wgsl).toBe(b.problem.damping?.wgsl);
    }
  });

  it('makes parameters of the featured options of parts, and settings of the rest', () => {
    const spec: SetupSpec = { ...presetSpec('binary')!, featured: ['parts.2.ecc', 'parts.0.soft', 'physics.viscosity'] };
    expect(paramFeatures(spec).map((f) => f.path)).toEqual(['parts.2.ecc', 'parts.0.soft']);
    const { problem } = compileSetup(spec);
    expect(problem.params!.map((p) => [p.id, p.value, !!p.live])).toEqual([['ecc', 0, false], ['soft', 0.06, true]]);
  });

  it('compiles an empty document, and one with a mass, a region and an obstacle', () => {
    expect(compileSetup(emptySetup()).problem.box).toBe(1);
    const { spec, report } = readSetup({ ...emptySetup(), domain: { box: 2 }, background: { rho: 1, p: 1 }, parts: [
      { kind: 'mass', gm: 1, pos: [0.5, 0, 0], soft: 0.1 }, { kind: 'region', shape: { type: 'box', min: [-0.2, -0.2, null], max: [0.2, 0.2, null] }, rho: 5 }, { kind: 'obstacle', shape: { type: 'sphere', center: [-0.5, 0, 0], radius: 0.1 } }] });
    expect(report.error).toBeUndefined();
    const { problem } = compileSetup(spec!);
    expect(problem.masses!({}, 1, 0.01, 3)).toHaveLength(1);
  });
});

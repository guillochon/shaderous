// The built-in flows (src/setup/parts/flows): each has parameters, and each parameter does something.
import { beforeAll, describe, expect, it } from 'vitest';
import { makeEos } from '../../src/eos/eos';
import { loadHelmholtz } from '../../src/eos/helmholtz';
import { PRESET_IDS, presetSpec } from '../../src/presets/index';
import { readSetup } from '../../src/setup/codec';
import { compileSetup } from '../../src/setup/compile';
import { FLOWS } from '../../src/setup/parts/flow';
import { cloneSetup, type FlowPart, type SetupSpec } from '../../src/setup/spec';
import { configFor } from '../../src/sim';

/** A preset that uses each flow, to try the flow in the setup it was made for. */
const home = new Map<string, { spec: SetupSpec; index: number }>();
for (const id of PRESET_IDS) {
  const spec = presetSpec(id)!;
  (spec.parts ?? []).forEach((p, index) => {
    if (p.kind === 'flow' && !home.has(p.name)) home.set(p.name, { spec, index });
  });
}

/** Parameters that only act with another switched on: what to switch on to try them. */
const WITH: Record<string, Record<string, boolean>> = { 'field-loop.rhoIn': { dense: true } };
/**
 * Parameters that only act with a gas that carries an energy, where their preset's does not: the
 * slow flows' Mach number sets an ideal gas's pressure, and a liquid has its own sound speed.
 */
const WITH_GAS = new Set(['gresho.mach', 'taylor-green.mach', 'shear-layers.mach']);

/** Everything a setup's flow decides, at sample points: the gas, the field, the burning and the heat. */
function fingerprint(spec: SetupSpec): string {
  const { problem } = compileSetup(spec);
  const out: unknown[] = [];
  for (const ndim of (problem.supports2d ? [2, 3] : [3]) as (2 | 3)[]) {
    const cfg = configFor(problem, 32, { ndim });
    const eos = makeEos(cfg.eos, cfg.eosParams);
    const at = problem.init(32, eos, ndim, cfg.params);
    const A = problem.mhd?.potential?.(cfg.params, ndim);
    const F = problem.mhd?.face?.(cfg.params, ndim);
    const heat = ndim === 3 ? problem.heat?.(cfg.params, eos) : undefined;
    const b = problem.box;
    // A lattice that is not the grid's, and the three axes closely: a hot spot or a pocket is
    // small, and sits on one.
    const pts: number[][] = [];
    for (let i = 0; i < 3000; i++) pts.push([b * (((i * 0.6180339887) % 1) - 0.5), b * (((i * 0.7548776662) % 1) - 0.5), ndim === 2 ? 0 : b * (((i * 0.5698402910) % 1) - 0.5)]);
    for (let i = 0; i < 400; i++) for (let a = 0; a < ndim; a++) pts.push([0, 1, 2].map((k) => (k === a ? b * ((i + 0.5) / 400 - 0.5) : 0)));
    for (const [x, y, z] of pts) {
      const w = at(x, y, z);
      out.push(w.rho, w.vx, w.vy, w.vz, w.p, w.s, w.X, A?.(x, y, z), F ? [F(0, x, y, z), F(1, x, y, z), F(2, x, y, z)] : 0, heat?.(x, y, z, w.rho));
    }
    out.push(problem.burn?.(cfg.params));
  }
  return JSON.stringify(out);
}

describe('the built-in flows', () => {
  beforeAll(async () => {
    await loadHelmholtz();
  });

  it('each have a preset that uses them', () => {
    expect([...home.keys()].sort()).toEqual(Object.keys(FLOWS).sort());
  });

  it('each have parameters, with defaults inside their ranges', () => {
    for (const def of Object.values(FLOWS)) {
      expect(Object.keys(def.fields).length, def.name).toBeGreaterThan(0);
      for (const [key, f] of Object.entries(def.fields)) {
        expect(f.label && f.tip, `${def.name}.${key}`).toBeTruthy();
        if (typeof f.default !== 'number') continue;
        expect(f.default, `${def.name}.${key}`).toBeGreaterThanOrEqual(f.min ?? -Infinity);
        expect(f.default, `${def.name}.${key}`).toBeLessThanOrEqual(f.max ?? Infinity);
        expect(f.min !== undefined && f.max !== undefined, `${def.name}.${key}: a range`).toBe(true);
      }
    }
  });

  for (const def of Object.values(FLOWS)) {
    it(`${def.name}: every parameter changes the setup, and the defaults are the preset`, () => {
      const { spec, index } = home.get(def.name)!;
      const base = fingerprint(spec);
      // The preset names few of its flow's parameters, or none: the defaults are its numbers.
      const part = spec.parts![index] as FlowPart;
      const spelled = cloneSetup(spec);
      for (const [key, f] of Object.entries(def.fields)) (spelled.parts![index] as FlowPart)[key] = part[key] ?? f.default;
      expect(fingerprint(spelled)).toBe(base);

      for (const [key, f] of Object.entries(def.fields)) {
        const now = (part[key] ?? f.default) as number | boolean;
        let next: number | boolean;
        if (typeof now === 'boolean') next = !now;
        else {
          // A step from where it is, inside its range.
          const step = f.factor ? now * (f.factor - 1) : (f.step ?? 0.1 * ((f.max ?? 1) - (f.min ?? 0)));
          next = now + step <= (f.max ?? Infinity) ? now + step : now - step;
        }
        const doc = cloneSetup(spec);
        const before = cloneSetup(spec);
        const gas = WITH_GAS.has(`${def.name}.${key}`);
        for (const d of [doc, before]) {
          Object.assign(d.parts![index], WITH[`${def.name}.${key}`] ?? {});
          if (gas) d.physics = { ...d.physics, eos: 'ideal' };
        }
        (doc.parts![index] as FlowPart)[key] = next;
        // The edited document is still one the reader takes as written.
        const read = readSetup(doc);
        expect(read.report.error, `${def.name}.${key}`).toBeUndefined();
        expect(read.report.notes, `${def.name}.${key}`).toEqual([]);
        expect(fingerprint(read.spec!) !== (WITH[`${def.name}.${key}`] || gas ? fingerprint(before) : base), `${def.name}.${key} = ${next} changes something`).toBe(true);
      }
    });
  }
});

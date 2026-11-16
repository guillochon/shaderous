// A run and its setup document, both ways (src/setup/running.ts).
import { describe, expect, it } from 'vitest';
import { PRESET_IDS, preset, presetSpec } from '../../src/presets/index';
import { compileSetup } from '../../src/setup/compile';
import { changeBetween, runningSpec, sameSetup, settingsOf, setSetting } from '../../src/setup/running';
import { cloneSetup, setPath } from '../../src/setup/spec';
import { configFor, type SimConfig } from '../../src/sim';
import { physicalEosParams } from '../../src/eos/eos';
import { problemUnits } from '../../src/problems/types';

/** A preset's configuration on its own grid (or a small one). */
function config(id: string): SimConfig {
  const { problem } = preset(id);
  const ndim = problem.grid?.ndim ?? 3;
  return configFor(problem, 64, { ndim, amrLevels: problem.grid?.amr ?? 0 });
}

/** A configuration as data: without the functions of its problem. */
const plain = (c: SimConfig) => JSON.parse(JSON.stringify({ ...c, problem: c.problem.id, tEnd: Number.isFinite(c.tEnd) ? c.tEnd : 'inf' }));

describe('the running document', () => {
  it('of an untouched preset is the preset, on the run\'s grid', () => {
    for (const id of PRESET_IDS) {
      const spec = runningSpec(config(id));
      expect(sameSetup(spec, presetSpec(id)!), id).toBe(true);
      expect(spec.domain?.n).toBe(64);
    }
  });

  it('holds what was changed since, and compiles back to the run\'s configuration', () => {
    for (const id of ['binary', 'kelvin-helmholtz', 'polytrope', 'thermal-instability', 'mri-disk', 'rayleigh-taylor']) {
      const cfg = config(id);
      for (const q of cfg.problem.params ?? []) if (q.step && q.value + q.step <= q.max) cfg.params[q.id] = q.value + q.step;
      // Units that follow the parameters set the equation of state's physical constants, as the page keeps them.
      if (typeof cfg.problem.units === 'function') Object.assign(cfg.eosParams, physicalEosParams(problemUnits(cfg.problem, cfg.params), cfg.problem.composition), cfg.problem.eosParams);
      cfg.eosParams.gamma = 1.3;
      cfg.cfl = 0.4;
      cfg.viscosity = 2e-3;
      cfg.fluff = { ...cfg.fluff, rho: 1e-7 };
      cfg.tEnd = 3;
      cfg.riemann = 'hll';
      cfg.refineHi = 0.6;
      if (cfg.problem.damping) cfg.dampRate = 1.5;
      const spec = runningSpec(cfg);
      expect(sameSetup(spec, presetSpec(id)!), id).toBe(false);
      const { problem } = compileSetup(spec);
      const back = configFor(problem, cfg.n, { ndim: cfg.ndim, amrLevels: cfg.amrLevels });
      const [a, b] = [plain(cfg), plain(back)];
      // The parameters are written into the document: its defaults are the run's values.
      expect(b, id).toEqual(a);
    }
  });

  it('keeps a 2D run\'s faces apart from the 3D ones', () => {
    const { problem } = preset('rayleigh-taylor');
    const cfg = configFor(problem, 64, { ndim: 2 });
    cfg.bcLo = [1, 2, 0];
    cfg.bcHi = [1, 2, 0];
    const spec = runningSpec(cfg);
    expect(spec.domain?.bc2d).toEqual({ lo: [1, 2, 0], hi: [1, 2, 0] });
    expect(spec.domain?.bc).toEqual(presetSpec('rayleigh-taylor')!.domain!.bc);
  });
});

describe('what a change costs', () => {
  const base = () => runningSpec(config('binary'));
  const edit = (path: string, value: unknown) => {
    const s = cloneSetup(base());
    setPath(s, path, value);
    return changeBetween(base(), s);
  };

  it('is nothing for the same document', () => {
    expect(changeBetween(base(), base()).cost).toBe('none');
  });
  it('is a uniform write for a live setting', () => {
    const c = edit('physics.viscosity', 5e-3);
    expect([c.cost, c.settings.map((s) => s.setting.path)]).toEqual(['live', ['physics.viscosity']]);
    expect(edit('physics.cfl', 0.3).cost).toBe('live');
    expect(edit('parts.3.damp.rate', 1).cost).toBe('live');
    // One point mass's softening: the run takes it up where the masses are.
    const soft = edit('parts.0.soft', 0.1);
    expect([soft.cost, soft.structure, soft.retune]).toEqual(['live', false, true]);
    expect(edit('name', 'Mine').cost).toBe('live');
  });
  it('is new shaders for the solver and the equation of state', () => {
    expect(edit('physics.riemann', 'hll').cost).toBe('shaders');
    expect(edit('physics.eos', 'ideal').cost).toBe('shaders');
  });
  it('is a fresh start for the grid, the parts and the box', () => {
    expect(edit('domain.n', 128).cost).toBe('restart');
    expect(edit('parts.2.ecc', 0.3).cost).toBe('restart');
    expect(edit('domain.box', 20).cost).toBe('restart');
    expect(edit('parts.3.sigma0', 2e-4).structure).toBe(true);
  });

  it('compares setups with different parts without touching either', () => {
    const [a, b] = [runningSpec(config('sedov')), runningSpec(config('binary'))];
    const before = JSON.stringify([a, b]);
    expect(changeBetween(a, b).structure).toBe(true);
    expect(changeBetween(b, a).structure).toBe(true);
    expect(JSON.stringify([a, b])).toBe(before);
  });

  it('applies settings to a configuration as the document has them', () => {
    const cfg = config('binary');
    const spec = base();
    for (const s of settingsOf(spec)) {
      if (s.path === 'physics.fluff.rho') setSetting(cfg, s, 1e-6);
      if (s.path === 'run.tEnd') setSetting(cfg, s, null);
      if (s.path === 'domain.bc') setSetting(cfg, s, { lo: [2, 2, 2], hi: [2, 2, 2] });
    }
    expect(cfg.fluff.rho).toBe(1e-6);
    expect(cfg.tEnd).toBe(Infinity);
    expect(cfg.bcLo).toEqual([2, 2, 2]);
  });
});

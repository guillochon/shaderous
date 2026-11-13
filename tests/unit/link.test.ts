import { describe, expect, it } from 'vitest';
import { configFor } from '../../src/sim';
import { FIELDS, problemById } from '../../src/problems';
import { defaultOpacity } from '../../src/render/transfer';
import { DEFAULT_SETUP, decodeLink, encodeLink, viewDefaults } from '../../src/ui/link';

const state = (id: string, n: number, ndim: 2 | 3, amrLevels = 0) => {
  const problem = problemById(id);
  return { cfg: configFor(problem, n, { ndim, amrLevels }), view: viewDefaults(problem, ndim), auto: true, colormap: 'inferno', opacity: defaultOpacity() };
};

describe('shareable links', () => {
  it('a bare visit opens the default setup', () => {
    const l = decodeLink(new URLSearchParams(''));
    expect([l.problem, l.ndim, l.n]).toEqual([DEFAULT_SETUP.problem, 2, 1024]);
  });

  it('a link that only names a preset opens it in 3D at the default resolution', () => {
    const l = decodeLink(new URLSearchParams('problem=sedov'));
    expect([l.problem, l.ndim, l.n]).toEqual(['sedov', 3, 0]);
  });

  it('a link that only names a preset with a grid of its own opens on that grid', () => {
    const l = decodeLink(new URLSearchParams('problem=tidal-disruption-n3'));
    expect([l.ndim, l.n, l.amr]).toEqual([3, 512, 4]);
  });

  it('a link that gives the cells but no levels is uniform, whatever the preset opens on', () => {
    const l = decodeLink(new URLSearchParams('problem=tidal-disruption-n3&dim=3&n=128'));
    expect([l.ndim, l.n, l.amr]).toEqual([3, 128, 0]);
    expect(decodeLink(new URLSearchParams('problem=tidal-disruption-n3&n=128&amr=1')).amr).toBe(1);
  });

  it('untouched settings give a link with nothing but the preset and grid', () => {
    expect(encodeLink(state('sedov', 128, 3)).toString()).toBe('problem=sedov&dim=3&n=128');
  });

  it('changed settings survive a round trip', () => {
    const s = state('tidal-disruption', 128, 3, 2);
    s.cfg.tEnd = 20;
    s.cfg.riemann = 'hll';
    s.cfg.limiter = 'mc';
    s.cfg.cfl = 0.5;
    s.cfg.eosParams.gamma = 1.4;
    s.cfg.gravEnergyFlux = false;
    s.cfg.viscosity = 0.002;
    s.cfg.conductivity = 0.5;
    s.cfg.params.beta = 2;
    s.view.mode = 'slice';
    s.view.field = 3;
    s.view.yaw = -1.25;
    s.view.lo = 1e-4;
    s.view.hi = 2.5;
    s.auto = false;
    s.colormap = Object.keys({ viridis: 1 })[0];
    s.opacity = [{ x: 0, a: 0 }, { x: 0.5, a: 0.4 }, { x: 1, a: 1 }];

    const l = decodeLink(new URLSearchParams(encodeLink(s).toString()));
    expect([l.problem, l.ndim, l.n, l.amr]).toEqual(['tidal-disruption', 3, 128, 2]);
    expect(l.cfg).toEqual({ tEnd: 20, riemann: 'hll', limiter: 'mc', cfl: 0.5, gravEnergyFlux: false, viscosity: 0.002, conductivity: 0.5 });
    expect(l.eosParams).toEqual({ gamma: 1.4 });
    expect(l.params).toEqual({ beta: 2 });
    expect(l.view).toMatchObject({ mode: 'slice', field: 3, yaw: -1.25, lo: 1e-4, hi: 2.5 });
    expect(l.auto).toBe(false);
    expect(l.opacity).toEqual(s.opacity);
  });

  it('ignores values it does not recognize and clamps the rest', () => {
    const l = decodeLink(new URLSearchParams('problem=sedov&riemann=exact&cfl=9&eos=<script>&field=99&tf=2:2,x:y&p.nope=3'));
    expect(l.cfg).toEqual({ cfl: 1 });
    expect(l.view.field).toBe(FIELDS.length - 1);
    expect(l.opacity).toBeUndefined();
    expect(l.params).toEqual({});
  });
});

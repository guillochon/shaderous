import { describe, expect, it } from 'vitest';
import { COOLING_MODULES } from '../../src/cooling/cooling';
import { preprocess } from '../../src/gpu/preprocess';
import { assemble, shaderFiles } from '../../src/gpu/shaders';
import { EOS_MODULES } from '../../src/eos/eos';

describe('shader preprocessor', () => {
  const files = {
    'a.wgsl': '#include "b.wgsl"\n#include <slot>\nfn a() {}',
    'b.wgsl': 'fn b() {}',
    'c.wgsl': 'fn c() {}',
  };

  it('resolves file includes and slots mapped to files', () => {
    const out = preprocess('a.wgsl', files, { slot: 'c.wgsl' });
    expect(out).toContain('fn b() {}');
    expect(out).toContain('fn c() {}');
    expect(out).toContain('fn a() {}');
    expect(out).not.toMatch(/^\s*#include/m);
  });

  it('accepts literal WGSL as a slot value', () => {
    expect(preprocess('a.wgsl', files, { slot: 'const K = 3u;' })).toContain('const K = 3u;');
  });

  it('includes each file once', () => {
    const out = preprocess('a.wgsl', files, { slot: 'b.wgsl' });
    expect(out.match(/fn b\(\) \{\}/g)).toHaveLength(1);
  });

  it('reports missing files, missing slots and cycles', () => {
    expect(() => preprocess('nope.wgsl', files)).toThrow(/not found/);
    expect(() => preprocess('a.wgsl', files)).toThrow(/slot <slot>/);
    expect(() => preprocess('l.wgsl', { 'l.wgsl': '#include <s>' }, { s: '#include <s>\n' })).toThrow(/cycle/);
  });
});

describe('shader assembly', () => {
  it('finds the shader sources', () => {
    expect(Object.keys(shaderFiles)).toContain('hydro/update.wgsl');
    expect(Object.keys(shaderFiles)).toContain('gpu/reduce.wgsl');
  });

  it('assembles the hydro passes for every equation of state, solver and limiter', () => {
    for (const eos of Object.values(EOS_MODULES))
      for (const riemann of ['hllc', 'hll'])
        for (const limiter of ['koren', 'mc', 'vanleer', 'minmod']) {
          for (const cooling of Object.values(COOLING_MODULES)) {
            const slots = { eos, riemann: `hydro/riemann/${riemann}.wgsl`, limiter: `hydro/limiters/${limiter}.wgsl`, cooling };
            for (const entry of ['hydro/rates.wgsl', 'hydro/update.wgsl', 'amr/update.wgsl', 'gravity/correct.wgsl', 'amr/correct.wgsl']) {
              const src = assemble(entry, slots);
              expect(src).not.toMatch(/^\s*#include/m);
              expect(src).toContain('fn eos_pressure');
            }
          }
        }
  });

  it('assembles the gravity and render passes', () => {
    const entries = [
      'gravity/smooth.wgsl', 'gravity/residual.wgsl', 'gravity/restrict.wgsl', 'gravity/prolong.wgsl',
      'gravity/rhs.wgsl', 'hydro/clock.wgsl', 'render/raymarch.wgsl', 'render/slice.wgsl',
    ];
    for (const entry of entries) expect(assemble(entry)).not.toMatch(/^\s*#include/m);
  });
});

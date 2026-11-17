import { describe, expect, it } from 'vitest';
import { outlineContains } from '../../src/sim';
import { base64ToF32, f32ToBase64, isSessionLog, planeOutline, pullOps, pullStep, queryDiff, SessionRecorder } from '../../src/ui/session';

describe('session log', () => {
  it('diffs two setup queries into the keys that changed', () => {
    expect(queryDiff('problem=sedov&n=128&cfl=0.3', 'problem=sedov&n=128&cfl=0.4&tend=2')).toEqual({ cfl: '0.4', tend: '2' });
    expect(queryDiff('problem=sedov&lim=mc', 'problem=sedov')).toEqual({ lim: null });
    expect(queryDiff('a=1', 'a=1')).toEqual({});
  });

  it('round-trips float32 values through base64, exactly', () => {
    const a = new Float32Array([0, -1, 1e-30, 3.4e38, Math.PI, NaN, -0]);
    const b = base64ToF32(f32ToBase64(a));
    expect(b.length).toBe(a.length);
    for (let i = 0; i < a.length; i++) expect(Object.is(b[i], a[i]) || (Number.isNaN(a[i]) && Number.isNaN(b[i]))).toBe(true);
    // Long arrays are encoded in chunks.
    const big = Float32Array.from({ length: 70_000 }, (_, i) => i * 0.25);
    expect(Array.from(base64ToF32(f32ToBase64(big)))).toEqual(Array.from(big));
  });

  it('records actions with the steps their stamps read back, in the order applied', async () => {
    const r = new SessionRecorder();
    expect(r.active).toBe(false);
    r.add({ kind: 'massChanged', mass: 1 }, Promise.resolve({ steps: 1, t: 0 })); // no log yet: ignored
    r.start('problem=sedov', 'test');
    let release!: (v: { steps: number; t: number }) => void;
    const late = new Promise<{ steps: number; t: number }>((res) => (release = res));
    const list = [{ pos: [0.5, 0.5, 0.5] as [number, number, number], radius: 0.1, softness: 0, vel: [0, 0, 0] as [number, number, number], rate: 1, decay: 0 }];
    r.add({ kind: 'sources', list }, late);
    r.add({ kind: 'massChanged', mass: 2 }, Promise.resolve({ steps: 12, t: 0.5 }));
    // The action is copied: the caller's list may change afterwards.
    list[0].rate = 99;
    setTimeout(() => release({ steps: 10, t: 0.25 }), 5);
    const log = (await r.snapshot())!;
    expect(isSessionLog(log)).toBe(true);
    expect(log.query).toBe('problem=sedov');
    expect(log.replayable).toBe(true);
    expect(log.entries.map((e) => [e.steps, e.t, e.action.kind])).toEqual([
      [10, 0.25, 'sources'],
      [12, 0.5, 'massChanged'],
    ]);
    expect((log.entries[0].action as { list: { rate: number }[] }).list[0].rate).toBe(1);
  });

  it('keeps one trace row per step', () => {
    const r = new SessionRecorder();
    r.start('q', 'test', false, 'from a file');
    const row = (steps: number) => ({ steps, t: steps, dt: 1, mass: 1, eTot: 1, maxMach: 0, floored: 0 });
    r.trace(row(1));
    r.trace(row(1));
    r.trace(row(2));
    return r.snapshot().then((log) => {
      expect(log!.trace.map((x) => x.steps)).toEqual([1, 2]);
      expect(log!.replayable).toBe(false);
      expect(log!.why).toBe('from a file');
    });
  });

  it('cuts a drag of the pull tool into frames the tool would take, which add up to the drag', () => {
    const s = { radius: 0.1, softness: 0.7, grip: 0.5 };
    const from: [number, number, number] = [0.2, 0.3, 0.5];
    const to: [number, number, number] = [0.6, 0.4, 0.5];
    const ops = pullOps(from, to, s, 1024, true);
    const cap = Math.min((0.15 * 0.7 * 0.1) / 0.5, 0.05, 30 / (1024 * 0.5));
    expect(ops.length).toBe(Math.ceil(Math.hypot(0.4, 0.1) / cap));
    const sum = ops.reduce((a, op) => a.map((v, i) => v + op.vel[i]), [0, 0, 0]);
    expect(sum[0]).toBeCloseTo(0.4, 6);
    expect(sum[1]).toBeCloseTo(0.1, 6);
    for (const op of ops) {
      expect(op.kind).toBe('pull');
      expect(Math.hypot(...op.vel)).toBeLessThanOrEqual(cap * (1 + 1e-9));
      expect(op.amount).toBe(0.5);
      expect(op.vel[2]).toBe(0); // in 2D, nothing along z
    }
    // Each op is centered half way along its own stretch.
    expect(ops[0].pos[0]).toBeCloseTo(from[0] + 0.5 * ops[0].vel[0], 9);
  });

  it('holds the rim to 0.3 of the radius and the grip below 0.9, and does nothing without a drag', () => {
    const r = pullStep([0.5, 0.5, 0.5], [0.6, 0.5, 0.5], { radius: 0.1, softness: 0, grip: 2 }, 256, false);
    expect(r.op!.softness).toBe(0.3);
    expect(r.op!.amount).toBe(0.9);
    expect(r.done).toBe(false);
    const still = pullStep([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], { radius: 0.1, softness: 0.7, grip: 0.5 }, 256, false);
    expect(still.op).toBeNull();
    expect(still.done).toBe(true);
  });

  it('draws an obstacle in a plane of the box and cuts it along the plane normal', () => {
    // A rectangle in the x-y plane (axis z): inside at any z, outside beside it.
    const rect = planeOutline({ shape: 'rect', center: [0.5, 0.4], half: [0.1, 0.05] }, 2);
    expect(outlineContains(rect, 0.55, 0.42, 0.1)).toBe(true);
    expect(outlineContains(rect, 0.55, 0.42, 0.9)).toBe(true);
    expect(outlineContains(rect, 0.65, 0.42, 0.5)).toBe(false);
    expect(outlineContains(rect, 0.55, 0.46, 0.5)).toBe(false);
    // An ellipse in the plane normal to x: its axes are y and z.
    const ell = planeOutline({ shape: 'ellipse', center: [0.5, 0.5], half: [0.2, 0.1] }, 0);
    expect(outlineContains(ell, 0.9, 0.65, 0.5)).toBe(true);
    expect(outlineContains(ell, 0.1, 0.5, 0.65)).toBe(false);
    // A triangle in the x-z plane (axis y).
    const tri = planeOutline({ shape: 'polygon', points: [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]] }, 1);
    expect(outlineContains(tri, 0.5, 0.0, 0.4)).toBe(true);
    expect(outlineContains(tri, 0.25, 0.9, 0.7)).toBe(false);
  });
});

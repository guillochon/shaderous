import { describe, expect, it } from 'vitest';
import { colorbarTicks, formatValue, normalOutOfScreen, PLANE_AXES, triadArrows } from '../../src/render/overlays';
import { cameraFrame, defaultView } from '../../src/render/renderer';
import { defaultBitrate, movieSize } from '../../src/io/movie';

describe('color bar ticks', () => {
  it('keeps both ends and puts round values between', () => {
    const t = colorbarTicks(0, 1, false);
    expect(t[0]).toMatchObject({ value: 0, frac: 0 });
    expect(t[t.length - 1]).toMatchObject({ value: 1, frac: 1 });
    expect(t.slice(1, -1).map((x) => x.value)).toEqual([0.2, 0.4, 0.6, 0.8]);
  });

  it('places ticks in proportion along a linear bar', () => {
    for (const x of colorbarTicks(-3, 7, false)) expect(x.frac).toBeCloseTo((x.value + 3) / 10, 12);
  });

  it('uses decades on a log scale, and keeps clear of the ends', () => {
    const t = colorbarTicks(0.03, 2e3, true);
    const mids = t.slice(1, -1);
    expect(mids.map((x) => x.value)).toEqual([0.1, 1, 10, 100, 1000].filter((v) => {
      const f = (Math.log10(v) - Math.log10(0.03)) / (Math.log10(2e3) - Math.log10(0.03));
      return f > 0.08 && f < 0.92;
    }));
    expect(mids.map((x) => x.label)).toContain('10²');
    for (const x of mids) expect(x.frac).toBeCloseTo((Math.log10(x.value) - Math.log10(0.03)) / (Math.log10(2e3) - Math.log10(0.03)), 12);
  });

  it('thins decades over a wide log range', () => {
    const mids = colorbarTicks(1e-10, 1e10, true).slice(1, -1);
    expect(mids.length).toBeLessThanOrEqual(6);
    expect(mids.length).toBeGreaterThanOrEqual(3);
  });

  it('falls back to round values over less than a decade and a half', () => {
    const mids = colorbarTicks(1, 5, true).slice(1, -1);
    expect(mids.length).toBeGreaterThan(0);
    for (const x of mids) expect(x.value).toBeGreaterThan(1);
  });

  it('copes with an empty range', () => {
    expect(colorbarTicks(2, 2, false)).toHaveLength(1);
    expect(colorbarTicks(0, 1, true)).toHaveLength(1);
  });

  it('formats labels compactly', () => {
    expect(formatValue(0.25)).toBe('0.25');
    expect(formatValue(1e5)).toBe('10⁵');
    expect(formatValue(2.5e-6)).toBe('2.5×10⁻⁶');
    expect(formatValue(-12.345)).toBe('−12.3');
  });
});

describe('triad', () => {
  const plane = (sliceAxis: number, twoD = false) => ({ mode: 'plane' as const, sliceAxis, right: [1, 0, 0], up: [0, 0, 1], twoD });

  it('shows a plane with its first axis to the right and its second up', () => {
    for (const k of [0, 1, 2]) {
      const [a, b] = triadArrows(plane(k), 10);
      expect([a.axis, b.axis]).toEqual(PLANE_AXES[k]);
      expect([a.dx, a.dy]).toEqual([10, 0]);
      expect([b.dx, b.dy]).toEqual([0, -10]);
    }
    expect(triadArrows(plane(0, true), 10).map((a) => a.label)).toEqual(['x', 'y']);
  });

  it('marks a plane normal coming out of the screen for a right-handed pair', () => {
    for (const k of [0, 1, 2]) {
      const [a, b] = PLANE_AXES[k];
      const e = (i: number) => [0, 1, 2].map((j) => (j === i ? 1 : 0));
      const [p, q] = [e(a), e(b)];
      const cross = [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
      expect(normalOutOfScreen(k)).toBe(cross[k] > 0);
    }
  });

  it('projects the box axes with the camera, nearest last', () => {
    const v = { ...defaultView(), yaw: 0.6, pitch: 0.4 };
    const { right, up, fwd } = cameraFrame(v, 1);
    const arrows = triadArrows({ mode: 'volume', sliceAxis: 2, right, up, twoD: false }, 100);
    expect(arrows).toHaveLength(3);
    for (const a of arrows) {
      expect(a.dx).toBeCloseTo(100 * right[a.axis], 10);
      expect(a.dy).toBeCloseTo(-100 * up[a.axis], 10);
      // up x right is the camera's forward direction.
      expect(a.depth).toBeCloseTo(fwd[a.axis], 10);
    }
    for (let i = 1; i < 3; i++) expect(arrows[i].depth).toBeLessThanOrEqual(arrows[i - 1].depth);
    // z is up on screen whenever the camera is not looking straight down.
    expect(arrows.find((a) => a.axis === 2)!.dy).toBeLessThan(0);
  });
});

describe('movie size', () => {
  it('rounds sides down to even numbers and caps them', () => {
    expect(movieSize(1921, 1081)).toEqual([1920, 1080]);
    expect(movieSize(9000, 3, 4096)).toEqual([4096, 16]);
    expect(defaultBitrate(1920, 1080, 30)).toBe(12441600);
  });
});

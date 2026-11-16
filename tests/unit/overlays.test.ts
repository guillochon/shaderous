import { describe, expect, it } from 'vitest';
import { colorbarTicks, formatValue, normalOutOfScreen, overlayGutters, PLANE_AXES, triadArrows } from '../../src/render/overlays';
import { cameraFrame, defaultView } from '../../src/render/renderer';
import { defaultBitrate, frameLayout, movieSize } from '../../src/io/movie';

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

describe('movie frame around the picture', () => {
  const none = () => ({ left: 0, right: 0 });
  it('takes the picture with a small margin, not the empty sides of a wide frame', () => {
    // A square picture in the middle of a wide frame.
    const l = frameLayout(2000, 1000, [0.265, 0.03, 0.735, 0.97], none);
    expect([l.sx, l.sy, l.sw, l.sh]).toEqual([530, 30, 940, 940]);
    expect([l.dx, l.dy, l.dw, l.dh]).toEqual([19, 19, 940, 940]);
    expect([l.width, l.height]).toEqual([978, 978]);
  });
  it('makes room beside the picture for the overlays', () => {
    const l = frameLayout(2000, 1000, [0.265, 0.03, 0.735, 0.97], (h) => ({ left: 0.2 * h, right: 0.1 * h }));
    expect(l.dx).toBe(196);
    expect(l.width).toBe(196 + 940 + 98);
  });
  it('leaves no margin where the picture runs off the frame, and takes it all when there are no bounds', () => {
    const l = frameLayout(1000, 800, [-0.5, -0.2, 1.5, 1.2], none);
    expect(l).toMatchObject({ width: 1000, height: 800, sx: 0, sy: 0, sw: 1000, sh: 800, dx: 0, dy: 0 });
    expect(frameLayout(1000, 800, null, none)).toMatchObject({ width: 1000, height: 800, sw: 1000, sh: 800 });
  });
  it('scales down to the largest side allowed', () => {
    const l = frameLayout(8000, 4000, null, none, 4096);
    expect([l.width, l.height, l.dw, l.dh]).toEqual([4096, 2048, 4096, 2048]);
  });
  it('asks for room only for the overlays shown, more for the triad of a volume', () => {
    const info = { mode: 'plane' as const, twoD: true, title: 'Density' };
    expect(overlayGutters(1000, info, {})).toEqual({ left: 0, right: 0 });
    const plane = overlayGutters(1000, info, { triad: true, colorbar: true });
    expect(plane.left).toBeGreaterThan(150);
    expect(plane.right).toBeGreaterThan(150);
    expect(plane.right).toBeLessThan(330);
    expect(overlayGutters(1000, { ...info, mode: 'volume', twoD: false }, { triad: true }).left).toBeGreaterThan(plane.left);
  });
});

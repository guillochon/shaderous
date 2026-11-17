// Turbulence driving (src/hydro/drive.ts): the modes of the band, and the first amplitudes.
import { describe, expect, it } from 'vitest';
import { DRIVE_BYTES, DRIVE_MODES, driveModes, packDrive, type DriveParams } from '../../src/hydro/drive';

const P: DriveParams = { accel: 2, tau: 0.5, kMin: 1, kMax: 2, zeta: 1, seed: 1 };

/** The modes as packed: wavevectors and both amplitudes. */
function unpack(buf: ArrayBuffer) {
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  const n = u[0];
  const modes = [...Array(n).keys()].map((m) => ({
    k: [...f.slice(8 + 12 * m, 11 + 12 * m)],
    ar: [...f.slice(12 + 12 * m, 15 + 12 * m)],
    ai: [...f.slice(16 + 12 * m, 19 + 12 * m)],
  }));
  return { n, header: [...u.slice(0, 4)], par: [...f.slice(4, 8)], modes };
}
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

describe('turbulence driving', () => {
  it('takes one of each pair of wavevectors in the band, shortest first', () => {
    // |n|^2 = 1, 2, 3, 4: 6 + 12 + 8 + 6 vectors, half of them.
    expect(driveModes(3, 1, 2)).toHaveLength(16);
    expect(driveModes(2, 1, 2)).toHaveLength(6);
    expect(driveModes(3, 1, 3)).toHaveLength(61);
    expect(driveModes(3, 1, 8)).toHaveLength(DRIVE_MODES);
    const modes = driveModes(3, 1, 3);
    const keys = new Set(modes.map((n) => n.join()));
    for (const n of modes) {
      expect(keys.has(n.map((c) => -c).join())).toBe(false);
      expect(Math.hypot(...n)).toBeGreaterThanOrEqual(1);
    }
    for (let i = 1; i < modes.length; i++) expect(Math.hypot(...modes[i])).toBeGreaterThanOrEqual(Math.hypot(...modes[i - 1]));
    expect(driveModes(2, 1, 3).every((n) => n[2] === 0)).toBe(true);
  });

  it('packs no modes when the driving is off', () => {
    const buf = packDrive({ ...P, accel: 0 }, 3, 1);
    expect(buf.byteLength).toBe(DRIVE_BYTES);
    expect(unpack(buf).n).toBe(0);
  });

  it('draws solenoidal or compressive amplitudes as zeta says, in the plane in 2D', () => {
    for (const m of unpack(packDrive({ ...P, zeta: 1 }, 3, 1)).modes) {
      expect(Math.abs(dot(m.k, m.ar)) / Math.hypot(...m.k)).toBeLessThan(1e-5);
      expect(Math.abs(dot(m.k, m.ai)) / Math.hypot(...m.k)).toBeLessThan(1e-5);
    }
    for (const m of unpack(packDrive({ ...P, zeta: 0 }, 3, 1)).modes) {
      expect(Math.hypot(...cross(m.k, m.ar)) / Math.hypot(...m.k)).toBeLessThan(1e-5);
    }
    for (const m of unpack(packDrive({ ...P, zeta: 0.5 }, 2, 1)).modes) {
      expect(m.ar[2]).toBe(0);
      expect(m.ai[2]).toBe(0);
    }
  });

  it('gives the box a mean square acceleration of accel^2 on average over seeds', () => {
    for (const [ndim, zeta] of [[3, 1], [3, 0.5], [2, 1], [2, 0]] as const) {
      let sum = 0;
      const seeds = 400;
      for (let seed = 1; seed <= seeds; seed++) {
        const { modes } = unpack(packDrive({ ...P, zeta, seed }, ndim, 1));
        // Box mean of |ar cos(k.x) - ai sin(k.x)|^2 summed over distinct modes.
        sum += modes.reduce((s, m) => s + 0.5 * (dot(m.ar, m.ar) + dot(m.ai, m.ai)), 0);
      }
      expect(sum / seeds / P.accel ** 2, `${ndim}D, zeta ${zeta}`).toBeCloseTo(1, 1);
    }
  });

  it('scales the wavevectors with the box and is the same from the same seed', () => {
    const a = unpack(packDrive(P, 3, 2));
    expect(Math.hypot(...a.modes[0].k)).toBeCloseTo(Math.PI, 5);
    expect(new Uint8Array(packDrive(P, 3, 1))).toEqual(new Uint8Array(packDrive(P, 3, 1)));
    expect(new Uint8Array(packDrive({ ...P, seed: 2 }, 3, 1))).not.toEqual(new Uint8Array(packDrive(P, 3, 1)));
  });
});

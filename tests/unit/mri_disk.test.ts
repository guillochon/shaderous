import { describe, expect, it } from 'vitest';
import { diskState, MRI, midplane, mriDisk } from '../../src/presets/names';

const params = { beta: 50, soft: 0.15 };
const eps = 0.15;
const h = 1e-4;

/** The curl of the preset's vector potential at a point, by central differences. */
function curl(x: number, y: number, z: number): number[] {
  const A = mriDisk.mhd!.potential!(params, 3);
  const d = (i: number, j: number) => {
    const e = [0, 0, 0];
    e[j] = h;
    return (A(x + e[0], y + e[1], z + e[2])[i] - A(x - e[0], y - e[1], z - e[2])[i]) / (2 * h);
  };
  return [d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)];
}

describe('magnetized accretion disk', () => {
  it('has plasma beta `beta` at the midplane where two loops meet', () => {
    for (const k of [1, 2, 3]) {
      const R = MRI.rIn * MRI.loop ** k;
      const b = curl(R, 0, 0);
      const beta = (2 * MRI.cs ** 2 * midplane(R)) / (b[0] ** 2 + b[1] ** 2 + b[2] ** 2);
      expect(beta).toBeCloseTo(50, 0);
    }
  });

  it('confines the field to the loops', () => {
    expect(curl(0.5, 0, 0).every((c) => c === 0)).toBe(true);
    expect(curl(3, 0, 0).every((c) => c === 0)).toBe(true);
    const H = MRI.cs * 1.2 ** 1.5; // the scale height at R = 1.2
    expect(curl(1.2, 0, 2.5 * H).every((c) => c === 0)).toBe(true);
  });

  it('starts in vertical and radial balance', () => {
    const cs2 = MRI.cs ** 2;
    const phi = (R: number, z: number) => -1 / Math.sqrt(R * R + z * z + eps * eps);
    for (const [R, z] of [[1, 0.1], [1.6, 0.3], [2.2, 0]]) {
      // Vertical: cs^2 dln(rho)/dz = -dPhi/dz (the disk, far above the ambient gas).
      const lnr = (zz: number) => Math.log(diskState(R, zz, eps).disk);
      const dz = (cs2 * (lnr(z + h) - lnr(z - h))) / (2 * h);
      expect(dz).toBeCloseTo(-(phi(R, z + h) - phi(R, z - h)) / (2 * h), 5);
      // Radial: v^2 / R = dPhi/dR + cs^2 dln(rho)/dR.
      const v = diskState(R, z, eps).v;
      const lnR = (RR: number) => Math.log(diskState(RR, z, eps).rho);
      const rhs = (phi(R + h, z) - phi(R - h, z)) / (2 * h) + (cs2 * (lnR(R + h) - lnR(R - h))) / (2 * h);
      expect((v * v) / R).toBeCloseTo(rhs, 4);
    }
  });

  it('holds about a hundredth of the central mass', () => {
    const init = mriDisk.init(256, undefined as never, 3, params);
    const n = 96;
    const dx = MRI.box / n;
    let m = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        for (let k = 0; k < n; k++) m += init(-4 + (i + 0.5) * dx, -4 + (j + 0.5) * dx, -4 + (k + 0.5) * dx).rho * dx ** 3;
    expect(m).toBeGreaterThan(0.007);
    expect(m).toBeLessThan(0.012);
  });
});

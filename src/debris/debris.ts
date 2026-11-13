// Debris energies: the gas binned by its specific orbital energy about one point mass (a black
// hole, for a tidal disruption), for the rate at which bound debris comes back to it, dM/dt.
//
// Gas that leaves the box is gone from the grid, but once out it moves almost ballistically in
// the mass's potential, so its energy no longer changes. The update passes therefore bin what
// crosses each open face of the box as it crosses (the ledger, kept for the whole run and saved
// in snapshots). What is still in the box is binned on demand by a census pass, which first
// sets apart a surviving self-bound core, found by iteration: gas is bound to the core where
// 1/2 |v - v_core|^2 + phi_self < 0, and v_core is the mean velocity of that gas.
//
// The energy is e = 1/2 |v - v_bh|^2 + phi, phi being the full potential at the gas (the
// mass's, and the gas's own, which matters where the gas is still near what is left of the
// star). Kepler's third law gives the time a bound parcel comes back to pericenter,
//   t(e) = t_peri + 2 pi G M (-2 e)^(-3/2),
// and so the return rate dM/dt = dM/de * (2 pi G M)^(2/3) / 3 * (t - t_peri)^(-5/3).
//
// Bins: DEBRIS_BINS across +-DEBRIS_SPAN times the problem's energy scale (Problem.debris),
// plus one below and one above. GPU layout in debris/debris.wgsl.
import type { SimConfig } from '../sim';

export const DEBRIS_BINS = 500;
export const DEBRIS_SPAN = 5;
const HEAD_BYTES = 48;
/** Fixed-point units per reference mass: 2^52 leaves 2^11 reference masses of headroom. */
const FIX = 2 ** 52;
const TWO32 = 4294967296;

export interface DebrisSpec {
  mass: number;
  de: number;
  tPeri: number;
  refMass: number;
  /** bin only the gas the passive scalar marks (a star's), the rest counted apart as ambient */
  scalar?: boolean;
}

/** The run's debris settings, or null if its problem does not ask for them. */
export function debrisSpec(c: SimConfig): DebrisSpec | null {
  const s = c.problem.debris?.(c.params, c.G);
  return s && s.de > 0 && s.refMass > 0 && c.problem.masses ? s : null;
}

/** The update passes' debris slot (debris/debris_on.wgsl or the stub). */
export const debrisSlot = (c: SimConfig): string => (debrisSpec(c) ? 'debris/debris_on.wgsl' : 'debris/debris_off.wgsl');

/** The ledger as a snapshot keeps it: the bins' layout and the mass that left in each slot. */
export interface DebrisLedger {
  lo: number;
  width: number;
  /** below the first bin, the bins, above the last, and the gas left out of the bins */
  mass: number[];
}

export interface DebrisReport {
  /** time of the run, of pericenter, G M of the mass and the energy scale */
  t: number;
  tPeri: number;
  gm: number;
  de: number;
  /** bin k covers energies lo + k width to lo + (k + 1) width */
  lo: number;
  width: number;
  /** mass per bin: what left the box, and what is in it (the core apart) */
  escaped: number[];
  inBox: number[];
  /** mass below the first bin and above the last, each [escaped, in the box] */
  below: [number, number];
  above: [number, number];
  /** gas left out of the bins (where the scalar marks the debris: the ambient medium), [escaped, in the box] */
  ambient: [number, number];
  /** the surviving core, if the run has self-gravity and something is bound to it */
  core: { mass: number; vel: number[] } | null;
  /**
   * escaped and inBox: all the gas that has left and is in the box, ambient included, so that
   * escaped + inBox + core (+ what sinks took) is the starting mass; bound and unbound: the
   * binned debris with energy below and above zero; ambient: escaped and in-box ambient gas
   */
  totals: { escaped: number; inBox: number; core: number; bound: number; unbound: number; ambient: number };
  /** the bound debris by the time it returns to pericenter, earliest first */
  dmdt: { t: number[]; rate: number[] };
}

/** The buffers of a run's debris energies, and the readbacks. */
export class DebrisBuffers {
  readonly nb = DEBRIS_BINS;
  readonly lo: number;
  readonly width: number;
  private readonly fix: number;
  private readonly fixP: number;
  /** slots in the ledger (below, bins, above, ambient) and in a census (those, then core mass, momentum) */
  private readonly ledgerSlots = DEBRIS_BINS + 3;
  private readonly censusSlots = DEBRIS_BINS + 7;
  readonly ledger: GPUBuffer;
  readonly census: GPUBuffer;

  constructor(
    private readonly device: GPUDevice,
    readonly spec: DebrisSpec,
  ) {
    this.lo = -DEBRIS_SPAN * spec.de;
    this.width = (2 * DEBRIS_SPAN * spec.de) / DEBRIS_BINS;
    this.fix = FIX / spec.refMass;
    // Momentum in units of the reference mass moving at 64 sqrt(de): ample for any core.
    this.fixP = this.fix / (64 * Math.sqrt(spec.de));
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.ledger = device.createBuffer({ label: 'debris ledger', size: HEAD_BYTES + 8 * this.ledgerSlots, usage });
    this.census = device.createBuffer({ label: 'debris census', size: HEAD_BYTES + 8 * this.censusSlots, usage });
  }

  /** The header: the core's velocity guess (or none), or with `seed` a density above which gas is the core. */
  private head(core: number[] | null, seed?: number): ArrayBuffer {
    const b = new ArrayBuffer(HEAD_BYTES);
    const mode = seed !== undefined ? 2 : core ? 1 : 0;
    new Float32Array(b, 0, 8).set([this.lo, 1 / this.width, this.fix, this.fixP, ...(core ?? [0, 0, 0]), mode]);
    new Uint32Array(b, 32, 3).set([this.spec.mass, this.nb, this.spec.scalar ? 1 : 0]);
    new Float32Array(b, 44, 1)[0] = seed ?? 0;
    return b;
  }

  /** Empty the ledger, or fill it from a snapshot's if that has the same bins. */
  reset(seed?: DebrisLedger) {
    const q = this.device.queue;
    q.writeBuffer(this.ledger, 0, this.head(null));
    const words = new Uint32Array(2 * this.ledgerSlots);
    const same = seed && seed.mass.length === this.ledgerSlots && Math.abs(seed.lo - this.lo) <= 1e-6 * Math.abs(this.lo) && Math.abs(seed.width - this.width) <= 1e-6 * this.width;
    if (seed && !same) console.warn('debris: the snapshot bins its debris differently; starting the ledger empty');
    if (same) {
      seed.mass.forEach((m, i) => {
        const v = Math.round(m * this.fix);
        const hi = Math.floor(v / TWO32);
        words[2 * i] = v - hi * TWO32;
        words[2 * i + 1] = hi >>> 0;
      });
    }
    q.writeBuffer(this.ledger, HEAD_BYTES, words);
  }

  /** What the update pipelines bind for the tally. */
  bindings(bh: GPUBuffer): Record<number, GPUBuffer> {
    return { 20: bh, 21: this.ledger };
  }

  /** The fixed-point totals of a buffer's first `slots` slots, as numbers of units. */
  private async units(buf: GPUBuffer, slots: number, enc = this.device.createCommandEncoder()): Promise<number[]> {
    const staging = this.device.createBuffer({ size: 8 * slots, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(buf, HEAD_BYTES, staging, 0, 8 * slots);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    const i = new Int32Array(u.buffer);
    return Array.from({ length: slots }, (_, k) => i[2 * k + 1] * TWO32 + u[2 * k]);
  }

  /** The ledger in mass units, as a snapshot keeps it. */
  async readLedger(): Promise<DebrisLedger> {
    const u = await this.units(this.ledger, this.ledgerSlots);
    return { lo: this.lo, width: this.width, mass: u.map((x) => x / this.fix) };
  }

  /**
   * Census of the gas in the box. `encode` dispatches the grid's census pass with this census
   * buffer bound. With `withCore`, the core is found by iteration: a first pass takes the mean
   * velocity of the gas denser than a tenth of the peak density `rhoMax` (the gas's mean
   * velocity `v0` if there is none), and each pass after it gives the gas bound to the current
   * guess, whose mean velocity is the next guess. Starting from all the gas instead, fast
   * debris can pull the first guess so far off that nothing is bound to it.
   */
  async takeCensus(encode: (enc: GPUCommandEncoder) => void, withCore: boolean, v0: number[], rhoMax: number) {
    let core: number[] | null = null;
    if (withCore) {
      const s = await this.pass(encode, null, 0.1 * rhoMax);
      core = s.core ? s.core.vel : v0.slice();
    }
    let last = -1;
    for (let k = 0; ; k++) {
      const { bins, core: found } = await this.pass(encode, core);
      if (!core || !found) return { bins, core: null };
      const { mass: m, vel: v } = found;
      const settled = Math.abs(m - last) <= 1e-6 * m && Math.hypot(v[0] - core[0], v[1] - core[1], v[2] - core[2]) <= 1e-6 * Math.sqrt(this.spec.de);
      if (settled || k >= 12) return { bins, core: found };
      core = v;
      last = m;
    }
  }

  /** One census pass with the given core guess (or seed threshold). */
  private async pass(encode: (enc: GPUCommandEncoder) => void, core: number[] | null, seed?: number) {
    const q = this.device.queue;
    q.writeBuffer(this.census, 0, this.head(core, seed));
    q.writeBuffer(this.census, HEAD_BYTES, new Uint32Array(2 * this.censusSlots));
    const enc = this.device.createCommandEncoder();
    encode(enc);
    const u = await this.units(this.census, this.censusSlots, enc);
    const bins = u.slice(0, this.ledgerSlots).map((x) => x / this.fix);
    const m = u[this.ledgerSlots] / this.fix;
    const p = u.slice(this.ledgerSlots + 1).map((x) => x / this.fixP);
    return { bins, core: m > 0 ? { mass: m, vel: p.map((x) => x / m) } : null };
  }

  /** The report from the ledger, a census, and the mass's G M at time t. */
  report(t: number, gm: number, ledger: DebrisLedger, census: { bins: number[]; core: { mass: number; vel: number[] } | null }): DebrisReport {
    const nb = this.nb;
    const escaped = ledger.mass.slice(1, nb + 1);
    const inBox = census.bins.slice(1, nb + 1);
    const center = (k: number) => this.lo + (k + 0.5) * this.width;
    let bound = ledger.mass[0] + census.bins[0];
    let unbound = ledger.mass[nb + 1] + census.bins[nb + 1];
    const ts: number[] = [];
    const rate: number[] = [];
    const tp = this.spec.tPeri;
    const c = (2 * Math.PI * gm) ** (2 / 3) / 3;
    for (let k = 0; k < nb; k++) {
      const m = escaped[k] + inBox[k];
      const e = center(k);
      if (e < 0) bound += m;
      else unbound += m;
      if (e >= 0) continue;
      const dt = (2 * Math.PI * gm) / (-2 * e) ** 1.5;
      ts.push(tp + dt);
      rate.push((m / this.width) * c * dt ** (-5 / 3));
    }
    // Most bound first: the bins run from the most bound up, so the times are already rising.
    const sum = (a: number[]) => a.reduce((s, x) => s + x, 0);
    return {
      t,
      tPeri: tp,
      gm,
      de: this.spec.de,
      lo: this.lo,
      width: this.width,
      escaped,
      inBox,
      below: [ledger.mass[0], census.bins[0]],
      above: [ledger.mass[nb + 1], census.bins[nb + 1]],
      ambient: [ledger.mass[nb + 2], census.bins[nb + 2]],
      core: census.core,
      totals: {
        escaped: sum(ledger.mass), inBox: sum(census.bins), core: census.core?.mass ?? 0, bound, unbound, ambient: ledger.mass[nb + 2] + census.bins[nb + 2],
      },
      dmdt: { t: ts, rate },
    };
  }

  destroy() {
    this.ledger.destroy();
    this.census.destroy();
  }
}

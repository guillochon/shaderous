// Obstacles on the adaptive mesh (plans/wind-tunnel.md 4.3, 9.2): every block within a band of a
// body's surface is kept at the level the body asks for (the finest by default), decided from
// the body's shape (setup/parts/surfaces.ts), and a block made by refinement there has its solid
// cells painted from the shape at its own level (amr/regrid.wgsl), not prolonged from the coarse
// cells, which see a part thinner than themselves in pieces or not at all.
//
// The band is wider than the ghost cells reach (SURFACE_BAND_CELLS finest cells against NG), so
// no coarser block's cells, and no ghost cell filled from them, lie on the body: the coarse grid's
// own idea of a thin body never reaches the flow that sees it.
import type { BodySurface } from '../setup/parts/surfaces';
import { BS, RG_PROLONG, type Block, type RegridOp } from './tree';

/** Blocks painted in one re-mesh at most (beyond that a new block keeps its parent's cells). */
export const PAINT_BLOCKS = 1024;
/** Words before the marks: the marks' offset, the mask of the bodies' marks, then one per slot. */
const HEAD = 2;

export class BodyBands {
  /** each block's level wanted (-1: none), as found */
  private wanted = new WeakMap<Block, number>();
  private warned = false;

  constructor(
    readonly surfaces: BodySurface[],
    private readonly ndim: 1 | 2 | 3,
    /** lower corner of the domain, and the coarsest cell's side, in the box's lengths */
    private readonly lo: readonly number[],
    private readonly dx0: number,
    private readonly levels: number,
  ) {}

  /** The block's lower corner and side. */
  private place(b: Block): { lo: number[]; size: number } {
    const size = (BS * this.dx0) / (1 << b.level);
    return { lo: [0, 1, 2].map((a) => (a < this.ndim ? this.lo[a] + b.c[a] * size : 0)), size };
  }

  /** The level a block's bodies want it at, or -1 if none comes near it. */
  target(b: Block): number {
    let t = this.wanted.get(b);
    if (t !== undefined) return t;
    t = -1;
    const { lo, size } = this.place(b);
    for (const s of this.surfaces) {
      const want = Math.min(this.levels, Math.max(0, s.level ?? this.levels));
      if (want > t && s.near(lo, size)) t = want;
    }
    this.wanted.set(b, t);
    return t;
  }

  /**
   * A block's refinement indicator with its bodies' wants: above `hi` (refine) below the level
   * wanted, at it kept from merging (and above `hi` where it cannot refine further anyway).
   */
  indicator(b: Block, raw: number, hi: number, lo: number, levelLimit: number): number {
    const t = Math.min(this.target(b), levelLimit);
    if (t < 0 || b.level > t) return raw;
    if (b.level < t || t >= levelLimit) return Infinity;
    return Math.max(raw, hi > lo ? 0.5 * (hi + lo) : hi);
  }

  /**
   * What amr/regrid.wgsl paints in the blocks a re-mesh makes by refinement near a body: per
   * slot the place of its marks (0: none), and the marks of its cells, four to a word (0: gas,
   * else the body's mark). Null when no new block is near a body.
   */
  paint(ops: RegridOp[], leaves: Block[], cap: number): Uint32Array | null {
    const cells = BS * BS * (this.ndim === 3 ? BS : 1);
    const words = cells / 4;
    const todo: Block[] = [];
    for (const b of leaves) if (ops[b.slot]?.kind === RG_PROLONG && this.target(b) >= 0) todo.push(b);
    if (!todo.length) return null;
    if (todo.length > PAINT_BLOCKS && !this.warned) {
      this.warned = true;
      console.warn(`bodies: ${todo.length} new blocks near a body in one re-mesh; the cells of those past ${PAINT_BLOCKS} are prolonged, not painted`);
    }
    const n = Math.min(todo.length, PAINT_BLOCKS);
    const base = HEAD + cap;
    const out = new Uint32Array(base + n * words);
    out[0] = base;
    for (const s of this.surfaces) out[1] |= 1 << s.mark;
    const depth = this.ndim === 3 ? BS : 1;
    for (let q = 0; q < n; q++) {
      const b = todo[q];
      out[HEAD + b.slot] = q + 1;
      const { lo, size } = this.place(b);
      const h = size / BS;
      let li = 0;
      for (let k = 0; k < depth; k++)
        for (let j = 0; j < BS; j++)
          for (let i = 0; i < BS; i++, li++) {
            // (A 1D block is its row, on y = 0: amr/regrid.wgsl reads the first row alone.)
            const x = lo[0] + (i + 0.5) * h, y = this.ndim === 1 ? 0 : lo[1] + (j + 0.5) * h, z = this.ndim === 3 ? lo[2] + (k + 0.5) * h : 0;
            let m = 0;
            for (const s of this.surfaces) if (s.inside(x, y, z)) m = s.mark;
            out[base + q * words + (li >> 2)] |= m << (8 * (li & 3));
          }
    }
    return out;
  }

  /** Bytes the paint buffer needs for `cap` slots. */
  static bytes(ndim: 1 | 2 | 3, cap: number): number {
    return 4 * (HEAD + cap + PAINT_BLOCKS * ((BS * BS * (ndim === 3 ? BS : 1)) / 4));
  }
}

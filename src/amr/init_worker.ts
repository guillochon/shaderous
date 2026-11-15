/**
 * A worker of amr/init_pool.ts: computes refinement indicators and leaves' initial states for
 * AmrSim.reset, with the problem looked up by id from the presets (functions do not cross to a
 * worker) and the rest of the configuration as sent.
 */
import { makeEos } from '../eos/eos';
import { loadHelmholtz } from '../eos/helmholtz';
import { problemById } from '../problems/index';
import type { SimConfig } from '../sim';
import { problemCoefficients } from '../species/mixture';
import { speciesGroups } from '../species/species';
import { indicator, leafState, surfaceOf, type FillContext, type Sample } from './init_cpu';
import type { InitRequest, InitReply } from './init_pool';

let ctx: (FillContext & { surface: (x: number, y: number, z: number) => Sample }) | null = null;
let setupKey = '';

const post = (m: InitReply, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

self.onmessage = async (e: MessageEvent<InitRequest>) => {
  const m = e.data;
  try {
    if (m.kind === 'setup') {
      const problem = problemById(m.problemId);
      const cfg = { ...m.cfg, problem } as SimConfig;
      // Its float64 mirror needs the table here too (eos/helmholtz.ts); the pool sends no work
      // before this setup answers.
      if (cfg.eos === 'helmholtz') await loadHelmholtz();
      const eos = makeEos(cfg.eos, cfg.eosParams);
      const fn = problem.init(m.n, eos, cfg.ndim, cfg.params);
      ctx = {
        cfg, dx0: m.dx0, levels: m.levels, fn, eos,
        kc: problemCoefficients(problem, cfg.eosParams.gamma),
        groups: speciesGroups(problem),
        mhd: problem.mhd !== undefined,
        surface: surfaceOf(fn, eos),
      };
      setupKey = m.key;
      post({ kind: 'ready', key: m.key, job: m.job });
      return;
    }
    if (!ctx || m.key !== setupKey) throw new Error('worker not set up for this run');
    if (m.kind === 'indicators') {
      const out = new Float64Array(m.blocks.length);
      for (let i = 0; i < m.blocks.length; i++) {
        const b = m.blocks[i];
        out[i] = indicator(ctx.cfg, ctx.dx0, b, ctx.surface, b.sub);
      }
      post({ kind: 'indicators', key: m.key, job: m.job, values: out }, [out.buffer]);
      return;
    }
    // Leaves' states, each sent back as soon as it is made so the page can upload it and let it go.
    for (const b of m.blocks) {
      const s = leafState(ctx, b);
      const transfer: Transferable[] = [s.u1.buffer, s.u2.buffer];
      for (const t of [s.faces, s.km, s.sp]) if (t) transfer.push(t.buffer);
      post({ kind: 'leaf', key: m.key, job: m.job, index: b.index, state: s }, transfer);
    }
    post({ kind: 'done', key: m.key, job: m.job });
  } catch (err) {
    post({ kind: 'error', key: m.key, job: m.job, message: String((err as Error)?.stack ?? err) });
  }
};

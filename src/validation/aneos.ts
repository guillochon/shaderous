// Validation tests of the ANEOS materials (eos/aneos.wgsl, eos/aneos.ts; plans/aneos-eos.md).
// Registered in tests.ts after the others.
import { assemble } from '../gpu/shaders';
import { bindGroup, computePipeline } from '../gpu/device';
import { configFor, PARAM_BYTES, Sim } from '../sim';
import { AmrSim } from '../amr/amr_sim';
import { problemById } from '../presets/index';
import { impactDebris } from '../debris/impact';
import { ANEOS_MATERIALS, prepareAneos } from '../eos/aneos';
import { aneosKeyOf, defaultEosParams, EOS_MODULES, eosConstants1, eosConstants2, makeEos, physicalEosParams } from '../eos/eos';
import { compSlot } from '../species/mixture';
import { PLANET_UNITS } from '../problems/planet';
import type { TestDef } from './tests';

export const ANEOS_TESTS: TestDef[] = [
  {
    id: 'aneos-eos',
    name: 'ANEOS materials on the GPU',
    criterion:
      'Rock, iron alloy, water and forsterite, and rock mixed with iron at mass fractions 0.1 to 0.9: over densities 1e-4 to 30 g/cm^3 and temperatures 200 K to 1e5 K, the energy from the pressure and the pressure back from that energy agree with the float64 mirror, the round trip to 1e-3 at the 99th percentile and the energy and the sound speed to 1e-2 (single precision holds ln e and asinh p to about 2e-6, and the slopes of the table to about 1e-3)',
    async run(gpu) {
      const d = gpu.device;
      const metrics: Record<string, number> = {};
      let pass = true;
      const base = { ...defaultEosParams(), ...physicalEosParams(PLANET_UNITS) };
      const cases = [
        ...ANEOS_MATERIALS.map((m, i) => ({ label: m, aneosA: i, aneosB: -1, xs: [0] })),
        { label: 'pyrolite_fe85si15', aneosA: 0, aneosB: 1, xs: [0, 0.1, 0.25, 0.4, 0.75, 0.9, 1] },
      ];
      const q = (a: number[], f: number) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];
      for (const c of cases) {
        const ep = { ...base, aneosA: c.aneosA, aneosB: c.aneosB };
        await prepareAneos(d, aneosKeyOf(ep));
        const pair = c.aneosB >= 0;
        // (rho, T) on a grid, as code density and the pressure the mirror gives there.
        const rows: number[][] = [];
        for (const xb of c.xs) {
          const eos = makeEos('aneos', ep, pair ? [1, xb, 0, 1] : undefined);
          for (let i = 0; i < 24; i++)
            for (let j = 0; j < 16; j++) {
              const rho = Math.fround((10 ** (-4 + (5.48 * i) / 23)) / ep.helmRho);
              const T = (200 * 10 ** ((2.7 * j) / 15)) / ep.helmT;
              const p = Math.fround(eos.pressure(rho, eos.eintFromT(rho, T)));
              // Where the pressure tells the temperature: not the cold end of the vapor dome.
              if (p * ep.helmRho * ep.helmE > 1e6) rows.push([rho, p, xb]);
            }
        }
        const inData = new Float32Array(rows.length * 4);
        rows.forEach((r, i) => inData.set(r, i * 4));
        const mk = (usage: number) => d.createBuffer({ size: inData.byteLength, usage });
        const inBuf = mk(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const outBuf = mk(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const readBuf = mk(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const params = d.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        d.queue.writeBuffer(inBuf, 0, inData);
        const pb = new Float32Array(PARAM_BYTES / 4);
        pb.set(eosConstants1('aneos', ep), 8);
        pb.set(eosConstants2('aneos', ep), 64);
        d.queue.writeBuffer(params, 0, pb);
        const slots: Record<string, string> = { eos: EOS_MODULES.aneos };
        if (pair) slots.comp = compSlot([[1, 0, 0, 1.5], [1, 1, 0, 1.5]], 1);
        const pipe = await computePipeline(d, assemble('eos/eos_test.wgsl', slots), `eos test aneos ${c.label}`);
        const enc = d.createCommandEncoder();
        const cp = enc.beginComputePass();
        cp.setPipeline(pipe);
        cp.setBindGroup(0, bindGroup(d, pipe, { 0: params, 1: inBuf, 2: outBuf }));
        cp.dispatchWorkgroups(Math.ceil(rows.length / 64));
        cp.end();
        enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, inData.byteLength);
        d.queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();
        for (const b of [inBuf, outBuf, readBuf, params]) b.destroy();
        const round: number[] = [];
        const de: number[] = [];
        const dc: number[] = [];
        const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);
        rows.forEach(([rho, p, xb], i) => {
          const eos = makeEos('aneos', ep, pair ? [1, xb, 0, 1] : undefined);
          const [eint, pBack, cs] = out.subarray(i * 4, i * 4 + 3);
          round.push(rel(pBack, p));
          de.push(rel(eint, eos.eintFromP(rho, p)));
          dc.push(rel(cs, eos.soundSpeed(rho, p)));
        });
        metrics[`${c.label}_points`] = rows.length;
        metrics[`${c.label}_roundtrip_p99`] = q(round, 0.99);
        metrics[`${c.label}_eint_p99`] = q(de, 0.99);
        metrics[`${c.label}_cs_p99`] = q(dc, 0.99);
        metrics[`${c.label}_roundtrip_max`] = Math.max(...round);
        pass &&= Number.isFinite(Math.max(...round, ...de, ...dc)) && q(round, 0.99) < 1e-3 && q(de, 0.99) < 1e-2 && q(dc, 0.99) < 1e-2;
      }
      return { pass, metrics };
    },
  },
  {
    id: 'planet-rest',
    name: 'A planet at rest (iron-alloy core under a rock mantle, ANEOS)',
    criterion:
      'An Earth-mass planet with a 30% Fe85Si15 core under pyrolite, started on its integrated structure (not settled) at 16 cells to the radius, keeps its central density within 8% over four dynamical times (t <= 4), with its mass on the grid within 1% of an Earth mass and conserved to 1e-4, and total energy conserved to 1e-3',
    async run(gpu) {
      const sim = await Sim.create(gpu, configFor(problemById('planet'), 64, { tEnd: 0 }));
      try {
        const d0 = await sim.diagnostics();
        const rho0 = d0.ranges[0][1];
        let worst = 0;
        let last = d0;
        let steps = 0;
        for (let t = 0.5; t <= 4.01; t += 0.5) {
          sim.cfg.tEnd = t;
          sim.writeParams();
          steps = await sim.runToEnd();
          last = await sim.diagnostics();
          worst = Math.max(worst, Math.abs(last.ranges[0][1] - rho0) / rho0);
        }
        const drift = Math.abs(last.eTot - d0.eTot) / Math.abs(d0.eTot);
        const dm = Math.abs(last.mass - d0.mass) / d0.mass;
        const pass = worst < 0.08 && drift < 1e-3 && dm < 1e-4 && Math.abs(d0.mass - 1) < 0.01;
        // Code density is Earth masses per Earth radius cubed: 23.09 g/cm^3.
        return { pass, metrics: { mass: d0.mass, central_density_gcc: rho0 * 23.09, max_deviation: worst, energy_drift: drift, mass_drift: dm, kinetic_over_total: last.eKin / Math.abs(last.eTot), steps } };
      } finally {
        sim.destroy();
      }
    },
  },
  {
    id: 'giant-impact',
    name: 'The moon-forming impact, to two hours after contact (adaptive mesh, two materials)',
    criterion:
      'Theia (0.12 Earth masses) strikes the proto-Earth (0.9) at 45 degrees and the speed of its fall, both with 30% Fe85Si15 cores under pyrolite, on two levels at 8 finest cells to an Earth radius; at t = 12 (two hours after contact) mass is conserved to 1e-3 and total energy to 1%, and the Kepler sorting of the gas (debris/impact.ts) gives a planet of 0.93 to 1.01 Earth masses with the iron share it started with (0.28 to 0.32), a disk of 0.005 to 0.08 Earth masses that is mostly Theia’s (over half), and under 0.02 escaping',
    async run(gpu) {
      const sim = await AmrSim.create(gpu, configFor(problemById('moon-forming-impact'), 128, { amrLevels: 1, tEnd: 0 }));
      try {
        const d0 = await sim.diagnostics();
        sim.cfg.tEnd = 12;
        sim.writeParams();
        const steps = await sim.runToEnd();
        const d = await sim.diagnostics();
        const c = await sim.readLeaves();
        const X = await sim.readLeafSpecies();
        const iron = new Float32Array(c.count);
        const theia = new Float32Array(c.count);
        let peak = 0;
        for (let q = 0; q < c.count; q++) {
          const rho = c.u1[4 * q];
          peak = Math.max(peak, rho);
          iron[q] = X[1][q] / Math.max(X[0][q] + X[1][q], 1e-30);
          // The dye is 1 in the Earth and 0.5 in Theia.
          theia[q] = Math.min(1, Math.max(0, 2 * (1 - c.u2[4 * q + 2] / rho)));
        }
        const r = impactDebris(c, sim.cfg.G, 5.513 / 23.09, [iron, theia], 1e-7 * peak);
        const drift = Math.abs(d.eTot - d0.eTot) / Math.abs(d0.eTot);
        const dm = Math.abs(d.mass - d0.mass) / d0.mass;
        const metrics = {
          steps, energy_drift: drift, mass_drift: dm, planet: r.planet, disk: r.disk, escaping: r.escaping, moon_lunar_masses: r.moon / 0.0123,
          disk_iron: r.shares[1][0], disk_from_theia: r.shares[1][1], planet_iron: r.shares[0][0],
        };
        const pass = Number.isFinite(d.eTot) && dm < 1e-3 && drift < 1e-2 && r.planet > 0.93 && r.planet < 1.01 && r.shares[0][0] > 0.28 && r.shares[0][0] < 0.32 && r.disk > 0.005 && r.disk < 0.08 && r.shares[1][1] > 0.5 && r.escaping < 0.02;
        return { pass, metrics };
      } finally {
        sim.destroy();
      }
    },
  },
];

// Forces on bodies (plans/wind-tunnel.md 4.2, docs/numerics.md, Forces on bodies).
//
// The update (hydro/walls_on.wgsl, amr/walls_on.wgsl) and, with implicit pressure, the
// correction (pressure/pwalls_on.wgsl, amr/pressure/pwalls_on.wgsl) write each gas cell's
// wall-face momentum fluxes, times the faces' areas, into two textures laid out as the state:
// WP the pressure's part, WV viscosity's (with implicit pressure the update's whole wall flux,
// advection's being nothing). The reduction (hydro/forces.wgsl, amr/forces.wgsl) sums them per
// body, the body read from the solid cell's mark (1 + its index), with the torque about each
// body's point. What is read is the force of the step the state ended (on blocks with per-block
// timesteps, each block's own last step).
import type { Bindable } from '../gpu/device';
import { Reducer, SUM } from '../gpu/reduce';
import { MAX_BODIES, type BodyRef, type ForceRef, type Problem } from '../problems/types';
import type { Slots } from '../gpu/preprocess';

/** Floats per body in the reduction: the pressure's force, viscosity's, the torque. */
export const BODY_FLOATS = 9;
export const FORCE_CHANNELS = MAX_BODIES * BODY_FLOATS;

type Vec3 = [number, number, number];

/** The force the gas exerts on one body, as summed over its wall faces. */
export interface BodyForce {
  id: string;
  /** the whole force, pressure's and viscosity's */
  force: Vec3;
  pressure: Vec3;
  viscous: Vec3;
  /** about the body's point (BodyRef.center) */
  torque: Vec3;
  /**
   * With a reference (Problem.forceRef): the drag, lift and side coefficients F / (rho U^2 A / 2),
   * and the moment coefficients T / (rho U^2 A L / 2) about x, y, z.
   */
  coefficients?: { drag: number; lift: number; side: number; moment: Vec3 };
}

/** Whether a run sums the forces on its bodies: it has some (the setup's obstacles), in Cartesian coordinates. */
export function forcesOn(c: { problem: Problem }): boolean {
  return (c.problem.bodies?.length ?? 0) > 0 && !c.problem.geometry && !c.problem.mhd;
}

/** The shader slots of the passes that write the forces, on or off. */
export function wallSlots(on: boolean): Slots {
  return {
    walls: on ? 'hydro/walls_on.wgsl' : 'hydro/walls_off.wgsl',
    amr_walls: on ? 'amr/walls_on.wgsl' : 'amr/walls_off.wgsl',
    pwalls: on ? 'pressure/pwalls_on.wgsl' : 'pressure/pwalls_off.wgsl',
    ap_walls: on ? 'amr/pressure/pwalls_on.wgsl' : 'amr/pressure/pwalls_off.wgsl',
  };
}

/** The reduction's output function, written out channel by channel (no run-time indexing: gpu/reduce.ts). */
function forcePut(visc: boolean, moving: boolean): string {
  const parts = ['fp.x', 'fp.y', 'fp.z', 'fv.x', 'fv.y', 'fv.z', 'tq.x', 'tq.y', 'tq.z'];
  let s = `const FORCE_VISC = ${visc};\nconst FORCE_MOVING = ${moving};\n`;
  s += 'fn force_put(b: u32, fp: vec3<f32>, fv: vec3<f32>, tq: vec3<f32>) -> Red {\n  var r = red_identity();\n';
  for (let k = 0; k < MAX_BODIES; k++) {
    s += `  if (b == ${k}u) {\n`;
    parts.forEach((p, i) => (s += `    r[${k * BODY_FLOATS + i}] = ${p};\n`));
    s += '  }\n';
  }
  return s + '  return r;\n}\n';
}

export class WallForces {
  /** the pressure's part, (force, mark), per cell */
  readonly wp: GPUTexture;
  /** viscosity's part, or one texel of nothing where no pass writes it */
  readonly wv: GPUTexture;
  private readonly refs: GPUBuffer;
  private red!: Reducer;
  /** whether a pass writes WV: with viscosity, or with implicit pressure (the update's whole wall flux) */
  readonly visc: boolean;
  /**
   * Moving bodies (hydro/moving_exchange.wgsl): the momentum the exchange hands the gas where a
   * cell changes kind, over the step's dt, as a force on the body, and its mark; or one texel of
   * nothing.
   */
  readonly wx: GPUTexture;
  readonly moving: boolean;

  private constructor(d: GPUDevice, make: (label: string) => GPUTexture, visc: boolean, bodies: BodyRef[], moving: boolean) {
    this.visc = visc;
    this.moving = moving;
    this.wx = moving
      ? make('wall forces (exchange)')
      : d.createTexture({ label: 'no exchange forces', size: [1, 1, 1], dimension: '3d', format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.wp = make('wall forces (pressure)');
    this.wv = visc
      ? make('wall forces (viscous)')
      : d.createTexture({ label: 'no viscous wall forces', size: [1, 1, 1], dimension: '3d', format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.refs = d.createBuffer({ label: 'body points', size: MAX_BODIES * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const r = new Float32Array(MAX_BODIES * 4);
    bodies.slice(0, MAX_BODIES).forEach((b, i) => r.set(b.center, 4 * i));
    d.queue.writeBuffer(this.refs, 0, r);
  }

  /**
   * @param make     a texture laid out as the state (rgba32float, storage and sampled)
   * @param dims     the reduction's grid: the state's cells, or blocks along x as the block grid's diagnostics
   * @param source   hydro/forces.wgsl or amr/forces.wgsl
   * @param extra    further bindings of the reduction (the block table on blocks)
   */
  static async create(d: GPUDevice, make: (label: string) => GPUTexture, visc: boolean, bodies: BodyRef[], dims: number[], source: string, slots: Slots, params: GPUBuffer, extra: Record<number, Bindable> = {}, moving = false): Promise<WallForces> {
    const w = new WallForces(d, make, visc, bodies, moving);
    w.red = await Reducer.create(
      d, dims, new Array(FORCE_CHANNELS).fill(SUM), source, { ...slots, force_put: forcePut(visc, moving) },
      [{ 0: params, 1: w.wp.createView(), 2: w.wv.createView(), 3: w.refs, 4: w.wx.createView(), ...extra }], 'wall forces',
    );
    return w;
  }

  /** The update's bindings (with implicit pressure it writes WV only, the correction WP). */
  updateBindings(): Record<number, GPUTextureView> {
    // (Both are bound whatever writes them: a shader's layout holds every texture it names, even
    // behind a constant that is false. WV of nothing is a texel no pass writes to.)
    return { 56: this.wp.createView(), 57: this.wv.createView() };
  }
  /** The implicit correction's binding. */
  correctBindings(): Record<number, GPUTextureView> {
    return { 56: this.wp.createView() };
  }

  encode(pass: GPUComputePassEncoder) {
    this.red.encode(pass);
  }
  get result(): GPUBuffer {
    return this.red.result;
  }

  destroy() {
    this.wp.destroy();
    this.wv.destroy();
    this.wx.destroy();
    this.refs.destroy();
  }
}

/** The bodies' forces from the reduction's channels (FORCE_CHANNELS floats). */
export function bodyForcesFrom(a: Float32Array, bodies: BodyRef[], ref?: ForceRef): BodyForce[] {
  const out: BodyForce[] = [];
  const n = Math.min(bodies.length, MAX_BODIES);
  for (let k = 0; k < n; k++) {
    const o = k * BODY_FLOATS;
    const v = (i: number): Vec3 => [a[o + i], a[o + i + 1], a[o + i + 2]];
    const pressure = v(0);
    const viscous = v(3);
    const force = pressure.map((p, i) => p + viscous[i]) as Vec3;
    // Bodies past the last share its sums.
    const id = k === MAX_BODIES - 1 && bodies.length > MAX_BODIES ? bodies.slice(k).map((b) => b.id).join(' + ') : bodies[k].id;
    const f: BodyForce = { id, force, pressure, viscous, torque: v(6) };
    if (ref && ref.rho > 0 && ref.speed > 0 && ref.area > 0) {
      const q = 0.5 * ref.rho * ref.speed ** 2 * ref.area;
      const side = 3 - ref.drag - ref.lift;
      f.coefficients = {
        drag: force[ref.drag] / q,
        lift: force[ref.lift] / q,
        side: side >= 0 && side < 3 && side !== ref.drag && side !== ref.lift ? force[side] / q : 0,
        moment: f.torque.map((t) => t / (q * (ref.length || 1))) as Vec3,
      };
    }
    out.push(f);
  }
  return out;
}

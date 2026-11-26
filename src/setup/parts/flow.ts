// Flows: named, built-in initial conditions with their parameters, for what shapes cannot say
// (a shear layer with its seeded modes, a ZND detonation profile, an Alfven wave). A setup file
// names one and gives its numbers; the function itself is in this build (setup/parts/flows/).
import type { AnyBurnSpec } from '../../burn/burn';
import type { Eos } from '../../eos/eos';
import type { Prim } from '../../problems/types';
import type { UnitSystem } from '../../units/units';
import type { Rule } from '../options';
import type { FlowPart, SetupSpec } from '../spec';
import type { Bodies } from './bodies';

export type InitFn = (x: number, y: number, z: number) => Prim;
export type PotentialFn = (x: number, y: number, z: number) => [number, number, number];
export type FaceFn = (axis: number, x: number, y: number, z: number) => number;

/** One parameter of a flow, as the dialog shows it and the codec checks it. */
export interface FlowField {
  label: string;
  tip?: string;
  default: number | boolean;
  min?: number;
  max?: number;
  step?: number;
  factor?: number;
  /** what changing it costs a running simulation (default: a restart) */
  apply?: 'live' | 'shaders' | 'restart';
}

/** What a flow is given: the setup (with the run's parameter values in it) and the grid. */
export interface FlowContext {
  spec: SetupSpec;
  box: number;
  ndim: 2 | 3;
  G: number;
  units?: UnitSystem;
  /** the run's species, in order */
  species?: readonly string[];
}

/** With the grid's resolution and the equation of state: what an initial state needs. */
export interface FlowInitContext extends FlowContext {
  n: number;
  eos: Eos;
  bodies: Bodies;
}

export interface FlowDef<P = Record<string, number | boolean>> {
  name: string;
  label: string;
  tip: string;
  fields: Record<string, FlowField>;
  /** It fills the whole box: it excludes other such flows, and the background is not used. */
  wholeBox: boolean;
  /** What the setup must have for it to run (options.ts facts): its init throws without them. */
  requires?: Rule[];
  /**
   * A whole-box flow whose state suits the weakly compressible liquid (densities near one
   * reference). Other whole-box flows are not offered for it: their densities, a liquid
   * compressed or stretched far past its model, tore runs apart (random test seed 10207).
   */
  liquid?: boolean;
  /** The initial state everywhere (a whole-box flow). */
  init?(ctx: FlowInitContext, p: P): InitFn;
  /** Or a change to the state the other parts painted. */
  decorate?(ctx: FlowInitContext, p: P, base: InitFn): InitFn;
  /** A magnetic field: a vector potential, or values on the faces. */
  potential?(ctx: FlowContext, p: P): PotentialFn;
  face?(ctx: FlowContext, p: P): FaceFn;
  /**
   * A damping zone that goes with the flow (Problem.damping, hydro/damping.wgsl): the gas held
   * at, or relaxed toward, a state that may change with time. A flow with an exact solution
   * holds its open faces to it with one. Undefined: none (a disk's or a tunnel's, if any).
   */
  damping?(ctx: FlowContext, p: P): { wgsl: string; radius: number; rate: number; spherical?: boolean } | undefined;
  /** Burning that goes with the flow. */
  burn?(ctx: FlowContext, p: P): AnyBurnSpec;
  /**
   * What is wrong with the flow in this setup that rules cannot say (setup/check.ts asks): a
   * sentence, or null. For values the flow sets in physical units that the setup's units would
   * carry past the GPU's 32-bit numbers.
   */
  issues?(ctx: FlowContext, p: P): string | null;
  /** Heat put in once the stars are settled (Problem.heat). */
  heat?(ctx: FlowInitContext, p: P): (x: number, y: number, z: number, rho: number) => number | undefined;
}

const found = import.meta.glob('./flows/*.ts', { eager: true }) as Record<string, { flows?: FlowDef<never>[] }>;

/** Every flow of this build, by name. */
export const FLOWS: Record<string, FlowDef> = Object.fromEntries(
  Object.values(found).flatMap((m) => (m.flows ?? []).map((f) => [f.name, f as unknown as FlowDef])),
);

/** A flow part's parameters: the file's values over the flow's defaults. */
export function flowValues(part: FlowPart): { def: FlowDef; values: Record<string, number | boolean> } {
  const def = FLOWS[part.name];
  if (!def) throw new Error(`unknown flow: ${part.name}`);
  const values: Record<string, number | boolean> = {};
  for (const [k, f] of Object.entries(def.fields)) {
    const v = part[k];
    values[k] = typeof v === typeof f.default ? (v as number | boolean) : f.default;
  }
  return { def, values };
}

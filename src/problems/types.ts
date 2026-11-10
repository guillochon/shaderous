import type { Eos, EosKind, EosParams } from '../eos/eos';

export interface Prim {
  rho: number;
  vx: number;
  vy: number;
  vz: number;
  p: number;
  s: number; // passive scalar
}

export const PERIODIC = 0;
export const OUTFLOW = 1;
export const REFLECT = 2;
export type Bc3 = [number, number, number];

export type GravityMode = 'off' | 'periodic' | 'isolated';

// Fields the renderer can show; order matches field.wgsl.
export const FIELDS = ['Density', 'Temperature', 'Passive scalar', 'Pressure', 'Speed'] as const;

export interface Problem {
  id: string;
  label: string;
  blurb: string;
  eos: EosKind;
  eosParams: Partial<EosParams>;
  bcLo: Bc3;
  bcHi: Bc3;
  gravity: GravityMode;
  G: number;
  box: number; // side length; the domain is [-box/2, box/2]^3
  tEnd: number;
  rhoFloor?: number;
  pFloor?: number;
  /** Enforce the Truelove pressure floor by default. */
  jeansFloor?: boolean;
  view?: { field?: number; log?: boolean; mode?: 'volume' | 'slice' };
  /** Returns the initial primitive state as a function of position. */
  init(n: number, eos: Eos): (x: number, y: number, z: number) => Prim;
}

export const all = (b: number): Bc3 => [b, b, b];

// What the setup dialog asks of the page that opens it (ui/app.ts), so the dialog itself holds
// no simulation: it edits a copy of a setup document and hands it back.
import type { SetupSpec } from '../../setup/spec';
import type { UnitSystem } from '../../units/units';

export interface PresetEntry {
  id: string;
  /** the name in the list (a variant's: "Sod shock tube: Along x") */
  label: string;
}

export interface SetupHost {
  /** The document the run stands for now: its preset or file, with every setting changed since. */
  running(): SetupSpec;
  /**
   * Take the edited document: the page does the least that gets the run there (a uniform write,
   * new shaders, or a fresh start) and resolves once it has. Rejects with what went wrong.
   */
  apply(spec: SetupSpec): Promise<void>;
  /** The presets, in their groups. */
  presets(): { group: string; entries: PresetEntry[] }[];
  /** A preset's whole document, and the text of its file as it is in the repository. */
  presetSpec(id: string): SetupSpec;
  presetFile(id: string): string;
  /** GPU memory the grid of a document needs, and what the page may take, in bytes; null where it cannot say. */
  memory(spec: SetupSpec): { need: number; budget: number } | null;
  /** The grids on offer: cells per side in 2D and 3D, the largest uniform ones, and the most levels of refinement for a resolution. */
  grids(): { res2d: number[]; res3d: number[]; uniformMax2d: number; uniformMax3d: number; maxLevels(n: number): number };
  /** A link that opens the page with a document, and whether it is short enough to share. */
  link(spec: SetupSpec): Promise<{ url: string; tooLong: boolean }>;
  /** The run's physical units for a document, or undefined if it is scale free. */
  units(spec: SetupSpec): UnitSystem | undefined;
  /** Whether the readouts are in physical units now. */
  physical(): boolean;
}

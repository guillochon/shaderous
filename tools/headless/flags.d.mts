export interface FlagEntry {
  /** 'name' (a switch) or 'name VALUE' */
  flags?: string[];
  /** a line of help alone, in place of flags */
  show?: string;
  help: string[];
  many?: boolean;
}
export const SECTIONS: { title: string; entries: FlagEntry[] }[];
export function helpText(): string;
export const OPTIONS: Record<string, { type: 'string' | 'boolean'; multiple?: boolean; short?: string }>;

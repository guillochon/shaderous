// The setup that is not a preset: one a user has made, opened from a file or a link, or a
// preset since changed in the dialog. The page holds one at a time, under the id 'custom', so
// that everything that finds a problem by its id (the menus, links, state files, the scripting
// interface) finds it as it finds a preset.
import { compiledFrom, type Compiled } from '../setup/compile';
import type { SetupSpec } from '../setup/spec';

export const CUSTOM_ID = 'custom';

let current: Compiled | null = null;
let version = 0;

/** Make a document the custom setup. It is kept as given, but for its id. */
export function setCustomSetup(spec: SetupSpec): Compiled {
  current = compiledFrom({ ...spec, id: CUSTOM_ID });
  version++;
  return current;
}

/** The custom setup, compiled, or null if there is none. */
export const customSetup = (): Compiled | null => current;

/** Changes whenever the custom setup does: what was made of the last one (a link's text) is stale. */
export const customVersion = (): number => version;

// The tables some equations of state need before they can be used: the white dwarf's
// (eos/helmholtz.ts) and hydrogen and helium's (eos/hhe.ts). One call for whichever a run has.
import type { EosKind, EosParams } from './eos';
import { loadHelmholtz, prepareHelmholtz } from './helmholtz';
import { hheTableFor, loadHhe, prepareHhe } from './hhe';

/** Load the table the float64 mirror of `kind` needs, if any (makeEos then works). */
export async function loadEosTable(kind: EosKind, params: Pick<EosParams, 'hheX'>): Promise<void> {
  if (kind === 'helmholtz') await loadHelmholtz();
  if (kind === 'hhe') {
    // No table for the composition: the run is refused (hheUnsupported in eos/eos.ts).
    const name = hheTableFor(params.hheX);
    if (name) await loadHhe(name);
  }
}

/**
 * Ready a device for runs with `kind`: its table loaded, uploaded, and made the one every
 * bind group made from here on gets (eos/table.ts). Call it before the run's passes are built.
 */
export async function prepareEosTable(device: GPUDevice, kind: EosKind, params: Pick<EosParams, 'hheX'>): Promise<void> {
  if (kind === 'helmholtz') await prepareHelmholtz(device);
  if (kind === 'hhe') {
    const name = hheTableFor(params.hheX);
    if (name) await prepareHhe(device, name);
  }
}

// The tables some equations of state need before they can be used: the white dwarf's
// (eos/helmholtz.ts), hydrogen and helium's (eos/hhe.ts) and the ANEOS materials' (eos/aneos.ts).
// One call for whichever a run has.
import { loadAneos, prepareAneos } from './aneos';
import { aneosKeyOf, type EosKind, type EosParams } from './eos';
import { loadHelmholtz, prepareHelmholtz } from './helmholtz';
import { hheTableFor, loadHhe, prepareHhe } from './hhe';

type TableParams = Pick<EosParams, 'hheX' | 'aneosA' | 'aneosB'>;

/** Load the table the float64 mirror of `kind` needs, if any (makeEos then works). */
export async function loadEosTable(kind: EosKind, params: TableParams): Promise<void> {
  if (kind === 'helmholtz') await loadHelmholtz();
  if (kind === 'hhe') {
    // No table for the composition: the run is refused (hheUnsupported in eos/eos.ts).
    const name = hheTableFor(params.hheX);
    if (name) await loadHhe(name);
  }
  if (kind === 'aneos') await loadAneos(aneosKeyOf(params));
}

/**
 * Ready a device for runs with `kind`: its table loaded, uploaded, and made the one every
 * bind group made from here on gets (eos/table.ts; the ANEOS materials' at a binding of their
 * own, eos/aneos.ts). Call it before the run's passes are built.
 */
export async function prepareEosTable(device: GPUDevice, kind: EosKind, params: TableParams): Promise<void> {
  if (kind === 'helmholtz') await prepareHelmholtz(device);
  if (kind === 'hhe') {
    const name = hheTableFor(params.hheX);
    if (name) await prepareHhe(device, name);
  }
  if (kind === 'aneos') await prepareAneos(device, aneosKeyOf(params));
}

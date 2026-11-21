// The checks of the setup dialog's issues list (src/setup/check.ts), the preview's point-mass
// integrator and the browser library of setups.
import { describe, expect, it } from 'vitest';
import { PRESET_IDS, presetSpec } from '../../src/presets/index';
import { buildSetupBodies, checkSetup, diskMass } from '../../src/setup/check';
import { formOf, OPTION_BY_PATH } from '../../src/setup/options';
import { FLOWS } from '../../src/setup/parts/flow';
import { emptySetup, type DiskPart, type Part, type SetupSpec } from '../../src/setup/spec';
import { getSetup, LIBRARY_KEY, listSetups, removeSetup, renameSetup, saveSetup, type SetupStore } from '../../src/ui/setup/library';
import { guessTime, integrateOrbits, orbitEnergy, sceneOf, type NBody } from '../../src/ui/setup/preview';

const setup = (over: Partial<SetupSpec>, parts: Part[] = []): SetupSpec => ({ ...emptySetup(), domain: { ndim: 3, n: 64, amr: 0, box: 4 }, ...over, parts });
/** A 3D box with isolated self-gravity: where stars may be. */
const stellar = (parts: Part[], over: Partial<SetupSpec> = {}): SetupSpec => setup({ physics: { gravity: 'isolated' }, ...over }, parts);
const star = (id: string, pos: [number, number, number], more: Partial<Extract<Part, { kind: 'star' }>> = {}): Part => ({ kind: 'star', id, type: 'polytrope', n: 1.5, mass: 1, radius: 0.5, pos, ...more });
const mass = (id: string, pos: [number, number, number] = [0, 0, 0], more: Partial<Extract<Part, { kind: 'mass' }>> = {}): Part => ({ kind: 'mass', id, gm: 1, soft: 0.2, pos, ...more });
const disk = (about: string[], more: Partial<DiskPart> = {}): Part => ({ kind: 'disk', id: 'disk', about, profile: 'thin', rIn: 0.5, rOut: 1.5, sigma0: 1e-4, cs: 0.1, floor: 1e-4, ...more });

const texts = (spec: SetupSpec, level: 'error' | 'warning', env?: Parameters<typeof checkSetup>[1]) => checkSetup(spec, env).filter((i) => i.level === level).map((i) => i.text);
const has = (list: string[], what: RegExp) => list.some((t) => what.test(t));

describe('checkSetup', () => {
  it('finds no error in any preset', () => {
    expect(PRESET_IDS.length).toBeGreaterThan(10);
    for (const id of PRESET_IDS) expect(texts(presetSpec(id)!, 'error'), id).toEqual([]);
  });

  it('finds a flow without what its init needs (random test seeds 5015, 5019, 5022)', () => {
    const flow = (name: string, over: Partial<SetupSpec> = {}) => texts(setup(over, [{ kind: 'flow', name }]), 'error');
    expect(has(flow('first-core'), /needs physical units/i)).toBe(true);
    expect(has(flow('co-ignition'), /needs physical units/i)).toBe(true);
    expect(has(flow('helium-shell-ignition'), /needs a star/i)).toBe(true);
    expect(has(flow('shearing-noise'), /needs a shearing box/i)).toBe(true);
    expect(has(flow('zeldovich'), /needs comoving coordinates/i)).toBe(true);
    // (Seed 10159: the ZND detonation burns fuel into ash.)
    expect(has(flow('znd-detonation'), /needs the species fuel and ash/i)).toBe(true);
    expect(has(flow('znd-detonation', { physics: { species: ['fuel', 'ash'] } }), /needs the species fuel and ash/i)).toBe(false);
    expect(has(flow('first-core', { domain: { ndim: 3, n: 64, units: { length: { unit: 'au', value: 1 }, mass: { unit: 'Msun', value: 1 } } } }), /needs physical units/i)).toBe(false);
  });

  it('finds a planet whose core is not among the run’s materials (random test seed 10031)', () => {
    const units = { length: { unit: 'Rearth', value: 1 }, mass: { unit: 'Mearth', value: 1 } };
    const planet: Part = { kind: 'star', id: 'p', type: 'planet', mass: 1, material: 'pyrolite', coreMaterial: 'fe85si15' };
    const doc = (species?: string[]) => stellar([planet], { domain: { ndim: 3, n: 64, box: 4, units }, physics: { eos: 'aneos', gravity: 'isolated', G: 'units', ...(species ? { species } : {}) } });
    expect(has(texts(doc(), 'error'), /fe85si15, is not among the run’s materials/)).toBe(true);
    expect(has(texts(doc(['pyrolite', 'fe85si15']), 'error'), /not among the run’s materials/)).toBe(false);
  });

  it('finds a gamma of 1 where the species set the gas (random test seed 10062)', () => {
    const doc = (gamma: number, species: string[] | 'aprox13') => texts(setup({ physics: { eosParams: { gamma }, species } }), 'error');
    expect(has(doc(1, 'aprox13'), /each needs a gamma above 1/)).toBe(true);
    expect(has(doc(1, ['he4', 'c12']), /each needs a gamma above 1/)).toBe(true);
    expect(has(doc(1, ['fuel', 'ash']), /gamma above 1/)).toBe(false);
    expect(has(doc(5 / 3, 'aprox13'), /gamma above 1/)).toBe(false);
  });

  it('finds a star without gravity, a liquid far from its density, and the Helmholtz gas without units (seeds 10028, 10006, 10025)', () => {
    expect(has(texts(stellar([star('a', [0, 0, 0])], { physics: { gravity: 'isolated', G: 0 } }), 'error'), /a star needs gravity/)).toBe(true);
    expect(has(texts(stellar([star('a', [0, 0, 0])]), 'error'), /a star needs gravity/)).toBe(false);
    const blob = (rho: number): Part => ({ kind: 'region', shape: { type: 'sphere', radius: 0.1 }, rho });
    const tunnel = (rho: number) => texts(setup({}, [{ kind: 'tunnel', mach: 0.1 }, { kind: 'obstacle', shape: { type: 'sphere', radius: 0.1 } }, blob(rho)]), 'error');
    expect(has(tunnel(0.16), /must start within 10%/)).toBe(true);
    expect(has(tunnel(1.05), /must start within 10%/)).toBe(false);
    expect(has(texts(setup({ physics: { eos: 'liquid', eosParams: { liqRho: 2 } } }, [blob(1)]), 'error'), /must start within 10%/)).toBe(true);
    expect(has(texts(setup({ physics: { eos: 'helmholtz' } }), 'error'), /needs physical units/i)).toBe(true);
    // Seed 10601: comoving coordinates have units only from their box size in Mpc/h.
    const units = { length: { unit: 'au', value: 1 }, mass: { unit: 'Msun', value: 1 } };
    const cosmo = (boxMpc?: number) => texts(setup({ domain: { ndim: 3, n: 64, units, bc: { lo: [0, 0, 0], hi: [0, 0, 0] } }, physics: { cosmology: { zStart: 10, ...(boxMpc ? { boxMpc } : {}) } } }, [{ kind: 'flow', name: 'co-ignition' }]), 'error');
    expect(has(cosmo(), /needs physical units/i)).toBe(true);
    expect(has(cosmo(10), /needs physical units/i)).toBe(false);
  });

  it('takes a wind tunnel’s gas for the run’s (random test seed 10012)', () => {
    const tunnel = (mach: number) => texts(setup({}, [{ kind: 'tunnel', mach }, { kind: 'obstacle', shape: { type: 'sphere', radius: 0.1 } }, { kind: 'flow', name: 'first-core' }]), 'error');
    expect(has(tunnel(0.1), /equation of state with an energy/i)).toBe(true);
    expect(has(tunnel(2), /equation of state with an energy/i)).toBe(false);
    // Seed 10207: the Rayleigh-Taylor flow's two densities in the tunnel's liquid; the flows made for the liquid stay offered.
    const flowIn = (name: string) => texts(setup({}, [{ kind: 'tunnel', mach: 0.1 }, { kind: 'obstacle', shape: { type: 'sphere', radius: 0.1 } }, { kind: 'flow', name }]), 'error');
    expect(has(flowIn('rayleigh-taylor'), /not for the liquid/i)).toBe(true);
    expect(has(flowIn('taylor-green'), /not for the liquid/i)).toBe(false);
    // Seed 10227: the tunnel's liquid is solved with implicit pressure, which takes no per-block timesteps.
    const binned = (mach: number) => texts(setup({ domain: { ndim: 3, n: 128, amr: 1 }, mesh: { timeBins: 1 } }, [{ kind: 'tunnel', mach }, { kind: 'obstacle', shape: { type: 'sphere', radius: 0.1 } }]), 'error');
    expect(has(binned(0.1), /not with implicit pressure/i)).toBe(true);
    expect(has(binned(2), /not with implicit pressure/i)).toBe(false);
  });

  it('finds nothing wrong with an empty box or a well-made binary', () => {
    expect(checkSetup(setup({}))).toEqual([]);
    const binary = setup({}, [mass('a', [0, 0, 0], { gm: 0.5 }), mass('b', [0, 0, 0], { gm: 0.5 }), { kind: 'orbit', of: ['a', 'b'], type: 'binary', separation: 1 }, disk(['a', 'b'], { rIn: 1, rOut: 1.8 })]);
    expect(checkSetup(binary)).toEqual([]);
  });

  it('finds stars that overlap, and a star that leaves the box', () => {
    const errors = texts(stellar([star('a', [0, 0, 0]), star('b', [0.6, 0, 0])]), 'error');
    expect(has(errors, /"a" and "b" overlap/)).toBe(true);
    const out = checkSetup(stellar([star('a', [1.8, 0, 0])]));
    const leaves = out.find((i) => /leaves the box/.test(i.text));
    expect(leaves?.level).toBe('error');
    expect(leaves?.path).toBe('parts.0');
    expect(texts(stellar([star('a', [-1, 0, 0]), star('b', [1, 0, 0])]), 'error')).toEqual([]);
  });

  it('finds point masses that overlap or start outside the box', () => {
    expect(has(texts(setup({}, [mass('a'), mass('b')]), 'error'), /same place/)).toBe(true);
    expect(has(texts(setup({}, [mass('a'), mass('b', [0.1, 0, 0])]), 'warning'), /overlap/)).toBe(true);
    expect(has(texts(setup({}, [mass('a', [3, 0, 0])]), 'warning'), /outside the box/)).toBe(true);
  });

  it('warns of a softening under two cells', () => {
    const out = checkSetup(setup({}, [mass('a', [0, 0, 0], { soft: 0.01 })]));
    const soft = out.find((i) => /softening/.test(i.text));
    expect(soft?.level).toBe('warning');
    expect(soft?.path).toBe('parts.0.soft');
    // Without a resolution there are no cells to measure against.
    expect(checkSetup(setup({ domain: { box: 4 } }, [mass('a', [0, 0, 0], { soft: 0.01 })]))).toEqual([]);
  });

  it('warns of a disk heavier than a tenth of what it orbits', () => {
    const heavy = setup({}, [mass('a'), disk(['a'], { sigma0: 0.1 })]);
    expect(has(texts(heavy, 'warning'), /of what it orbits/)).toBe(true);
    expect(texts(setup({}, [mass('a'), disk(['a'])]), 'warning')).toEqual([]);
    // A uniform annulus, to check the integral: the tapers take a little off pi (rOut^2 - rIn^2) sigma.
    const m = diskMass(disk(['a'], { rIn: 0.5, rOut: 1.5, sigma0: 1 }) as DiskPart, 1, 2, 4);
    expect(m).toBeGreaterThan(0.75 * Math.PI * 2);
    expect(m).toBeLessThan(1.1 * Math.PI * 2);
  });

  it('warns of a polytrope the gas cannot hold stable', () => {
    const soft = stellar([star('a', [0, 0, 0], { n: 3 })], { physics: { gravity: 'isolated', eosParams: { gamma: 1.2 } } });
    const found = checkSetup(soft).find((i) => /convection/.test(i.text));
    expect(found?.level).toBe('warning');
    expect(found?.path).toBe('parts.0.n');
    // n = 1.5 in gas of gamma 5/3 is marginal, and fine.
    expect(has(texts(stellar([star('a', [0, 0, 0])]), 'warning'), /convection/)).toBe(false);
  });

  it('finds an orbit that names a body that is not there, or of the wrong kind', () => {
    const missing = setup({}, [mass('a'), { kind: 'orbit', of: ['a', 'nobody'], type: 'binary' }]);
    const found = checkSetup(missing).find((i) => /"nobody", which is not there/.test(i.text));
    expect(found?.level).toBe('error');
    expect(found?.path).toBe('parts.1.of');
    const wrong = stellar([star('s', [0, 0, 0]), mass('m', [1.5, 0, 0]), { kind: 'orbit', of: ['s', 'm'], type: 'collision' }]);
    expect(has(texts(wrong, 'error'), /second body must be a star/)).toBe(true);
    const region: Part = { kind: 'region', id: 'r', shape: { type: 'sphere', center: [0, 0, 0], radius: 0.1 }, rho: 1 };
    const notBody = stellar([star('s', [0, 0, 0]), region, { kind: 'orbit', of: ['s', 'r'], type: 'binary' }]);
    expect(has(texts(notBody, 'error'), /must be a point mass or a star/)).toBe(true);
    const unset = setup({}, [mass('a'), mass('b', [1, 0, 0]), { kind: 'orbit', of: ['', ''], type: 'binary' }]);
    expect(has(texts(unset, 'error'), /not chosen/)).toBe(true);
  });

  it('checks a binary with a star: its frame, its tilt and how full its star is of its Roche lobe', () => {
    const orbit = (more: Partial<Extract<Part, { kind: 'orbit' }>> = {}): Part => ({ kind: 'orbit', of: ['s', 'm'], type: 'binary', separation: 1.6, ...more });
    const binary = (more: Partial<Extract<Part, { kind: 'orbit' }>> = {}, over: Partial<SetupSpec> = {}) => stellar([star('s', [0, 0, 0]), mass('m', [0, 0, 0], { gm: 0.5, soft: 0.15 }), orbit(more)], over);
    expect(checkSetup(binary())).toEqual([]);
    expect(checkSetup(binary({ corotating: true }))).toEqual([]);
    // A star of radius 0.5 and mass 1 beside half its mass fills its lobe at a = 0.5 / 0.4406 = 1.135.
    expect(checkSetup(binary({ separation: undefined, fill: 1 }))).toEqual([]);
    const over = checkSetup(binary({ separation: 1.05 })).find((i) => /Roche lobe/.test(i.text));
    expect(over?.level).toBe('warning');
    expect(over?.text).toMatch(/star "s" is 1\.0\d times the size of its Roche lobe/);
    expect(has(texts(binary({ corotating: true, ecc: 0.2 }), 'error'), /only a circular orbit/)).toBe(true);
    expect(has(texts(binary({ corotating: true }, { physics: { gravity: 'isolated', rotation: { omega: [0, 0, 1] } } }), 'error'), /rotating frame of its own/)).toBe(true);
    expect(has(texts(binary({ incl: 30 }), 'error'), /x-y plane/)).toBe(true);
    const fillOfMass = stellar([mass('m', [0, 0, 0], { gm: 0.5 }), star('s', [0, 0, 0]), { kind: 'orbit', of: ['m', 's'], type: 'binary', fill: 1 }]);
    expect(has(texts(fillOfMass, 'error'), /fill is of its first body/)).toBe(true);
  });

  it('finds a disk that orbits nothing', () => {
    const found = checkSetup(setup({}, [mass('a'), disk([])])).find((i) => /orbits nothing/.test(i.text));
    expect(found?.level).toBe('error');
    expect(found?.path).toBe('parts.1.about');
    expect(has(texts(setup({}, [mass('a'), disk(['ghost'])]), 'error'), /"ghost", which is not there/)).toBe(true);
  });

  it('finds more than one flow that fills the box', () => {
    const whole = Object.values(FLOWS).filter((f) => f.wholeBox).map((f) => f.name);
    expect(whole.length).toBeGreaterThan(1);
    const two = setup({}, [{ kind: 'flow', name: whole[0] }, { kind: 'flow', name: whole[1] }]);
    expect(has(texts(two, 'error'), /only one/i)).toBe(true);
    expect(has(texts(setup({}, [{ kind: 'flow', name: whole[0] }]), 'error'), /only one/i)).toBe(false);
  });

  it('finds a shape without what its kind needs', () => {
    const open = setup({}, [{ kind: 'region', shape: { type: 'box' } as never, rho: 2 }, { kind: 'obstacle', shape: { type: 'half', normal: [0, 0, 0] } }]);
    const errors = checkSetup(open).filter((i) => i.level === 'error');
    expect(errors.map((i) => i.path)).toEqual(['parts.0.shape.type', 'parts.1.shape.type']);
    expect(checkSetup(setup({}, [{ kind: 'region', shape: { type: 'box', min: [-1, -1, -1], max: [0, 1, 1] }, rho: 2 }]))).toEqual([]);
  });

  it('holds the grid to the memory budget', () => {
    const found = checkSetup(setup({}), { memory: { need: 3 * 2 ** 30, budget: 2 ** 30 } }).find((i) => /GPU memory/.test(i.text));
    expect(found?.level).toBe('error');
    expect(found?.text).toMatch(/3\.0 GB.*1\.0 GB/);
    expect(checkSetup(setup({}), { memory: { need: 2 ** 20, budget: 2 ** 30 } })).toEqual([]);
    expect(checkSetup(setup({}), { memory: null })).toEqual([]);
  });

  it('reports what the compiler throws, and never throws itself', () => {
    const twins = setup({}, [mass('a'), mass('a', [1, 0, 0])]);
    expect(has(texts(twins, 'error'), /same id/)).toBe(true);
    const hostile = [
      { ...emptySetup(), parts: [{ kind: 'bogus' }] },
      { ...emptySetup(), parts: [{ kind: 'orbit' }, { kind: 'disk' }, { kind: 'star' }, { kind: 'region' }] },
      { ...emptySetup(), domain: { units: { length: { unit: 'furlong', value: 1 } } }, physics: { G: 'units', composition: 'cheese' } },
      { ...emptySetup(), parts: [{ kind: 'flow', name: 'no such flow' }] },
    ] as unknown as SetupSpec[];
    for (const doc of hostile) {
      const out = checkSetup(doc);
      expect(out.some((i) => i.level === 'error'), JSON.stringify(doc)).toBe(true);
    }
  });

  it('builds what it can when a part cannot be built', () => {
    // A white dwarf's structure needs its equation of state's scales; the mass beside it does not.
    const spec = stellar([{ kind: 'star', id: 'wd', type: 'white-dwarf', mass: 0.6 }, mass('m', [1, 0, 0])]);
    const built = buildSetupBodies(spec);
    expect(built.bodies?.massIds).toEqual(['m']);
    const orphan = buildSetupBodies(setup({}, [mass('a'), { kind: 'orbit', of: ['a', 'gone'], type: 'binary' }]));
    expect(orphan.skipped).toEqual([1]);
    expect(orphan.bodies?.masses.length).toBe(1);
  });
});

describe('the preview', () => {
  const pair = (): NBody[] => [
    { gm: 0.5, pos: [-0.5, 0, 0], vel: [0, -0.5, 0] },
    { gm: 0.5, pos: [0.5, 0, 0], vel: [0, 0.5, 0] },
  ];

  it('closes a circular two-body orbit to a percent over one period', () => {
    const start = pair();
    const out = integrateOrbits(start, 2 * Math.PI);
    for (let i = 0; i < 2; i++) expect(Math.hypot(...out.bodies[i].pos.map((x, k) => x - start[i].pos[k]))).toBeLessThan(0.01);
    // The path is the circle: every sample half the separation from the center.
    expect(out.paths[0].length).toBeGreaterThan(100);
    for (const p of out.paths[0]) expect(Math.hypot(...p)).toBeCloseTo(0.5, 3);
    // The bodies given are left where they were.
    expect(start[0].pos).toEqual([-0.5, 0, 0]);
  });

  it('conserves energy, on an eccentric orbit too', () => {
    const e0 = orbitEnergy(pair());
    expect(e0).toBeCloseTo(-0.125, 12);
    expect(Math.abs(orbitEnergy(integrateOrbits(pair(), 2 * Math.PI).bodies) / e0 - 1)).toBeLessThan(1e-4);
    // Eccentricity 0.6 from apocenter: the step shortens toward pericenter.
    const v = 0.5 * Math.sqrt((1 - 0.6) / (1 + 0.6) / 1);
    const ecc: NBody[] = [{ gm: 0.5, pos: [-0.8, 0, 0], vel: [0, -v, 0] }, { gm: 0.5, pos: [0.8, 0, 0], vel: [0, v, 0] }];
    const e1 = orbitEnergy(ecc);
    expect(Math.abs(orbitEnergy(integrateOrbits(ecc, 2 * Math.PI).bodies) / e1 - 1)).toBeLessThan(1e-3);
  });

  it('holds a fixed body still and lets a massless one pull on nothing', () => {
    const out = integrateOrbits([{ gm: 1, pos: [0, 0, 0], vel: [0, 0, 0], fixed: true }, { gm: 1, pos: [1, 0, 0], vel: [0, 1, 0], massless: true }], 2 * Math.PI);
    expect(out.bodies[0].pos).toEqual([0, 0, 0]);
    expect(Math.hypot(...out.bodies[1].pos)).toBeCloseTo(1, 3);
    expect(integrateOrbits([], 1).paths).toEqual([]);
    expect(integrateOrbits(pair(), 0).steps).toBe(0);
  });

  it('draws its scene from the document alone', () => {
    const scene = sceneOf(setup({ domain: { ndim: 3, n: 64, box: 4, bc: { lo: [0, 1, 2], hi: [0, 1, 3] } } }, [
      mass('a', [0, 0, 0], { gm: 0.5 }), mass('b', [0, 0, 0], { gm: 0.5 }), mass('c', [1.5, 0, 0], { gm: 1e-3 }),
      { kind: 'orbit', of: ['a', 'b'], type: 'binary', separation: 1 }, disk(['a', 'b']),
      { kind: 'region', shape: { type: 'sphere', radius: 0.2 }, rho: 2 },
    ]));
    expect(scene.faces).toEqual({ lo: [0, 1, 2], hi: [0, 1, 3] });
    expect(scene.bodies.map((b) => [b.id, b.part, b.draggable])).toEqual([['a', 0, false], ['b', 1, false], ['c', 2, true]]);
    // The orbit placed the pair half a separation either side of the center.
    expect(scene.bodies[0].pos[0]).toBeCloseTo(-0.5, 12);
    expect(scene.bodies[1].soft).toBe(0.2);
    expect(scene.disks[0].gm).toBeCloseTo(1, 12);
    expect(scene.shapes.length).toBe(1);
    // With no dynamical time given: about the period of an orbit as wide as the bodies are apart.
    expect(guessTime(scene)).toBeGreaterThan(2 * Math.PI);
    expect(sceneOf({ ...emptySetup(), parts: [{ kind: 'bogus' }] } as unknown as SetupSpec).bodies).toEqual([]);
  });
});

describe('the library of setups', () => {
  function fake(initial?: string): SetupStore & { data: Map<string, string> } {
    const data = new Map<string, string>();
    if (initial !== undefined) data.set(LIBRARY_KEY, initial);
    return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => void data.set(k, v) };
  }
  const doc = setup({ name: 'Mine', physics: { viscosity: 0.01 } }, [mass('a')]);

  it('saves, lists, reads back, renames and removes', () => {
    const store = fake();
    expect(listSetups(store)).toEqual([]);
    expect(saveSetup('  My   binary ', doc, { store })).toBe('My binary');
    expect(saveSetup('Other', doc, { store })).toBe('Other');
    expect(listSetups(store)).toEqual(['My binary', 'Other']);
    const got = getSetup('My binary', store)!;
    expect(got.notes).toEqual([]);
    expect(got.spec).toEqual({ ...doc, name: 'My binary' });
    // A name that is taken gets a number, unless the entry is to be replaced.
    expect(saveSetup('Other', doc, { store })).toBe('Other (2)');
    expect(saveSetup('Other', setup({}), { store, replace: true })).toBe('Other');
    expect(getSetup('Other', store)!.spec.parts).toEqual([]);
    expect(saveSetup('', doc, { store })).toBe('Untitled setup');
    expect(renameSetup('My binary', 'Renamed', store)).toBe('Renamed');
    expect(listSetups(store)).toEqual(['Renamed', 'Other', 'Other (2)', 'Untitled setup']);
    expect(getSetup('Renamed', store)!.spec.name).toBe('Renamed');
    expect(renameSetup('Renamed', 'Other', store)).toBe('Other (3)');
    expect(renameSetup('nothing', 'x', store)).toBeNull();
    expect(removeSetup('Other', store)).toBe(true);
    expect(removeSetup('Other', store)).toBe(false);
    expect(listSetups(store)).toEqual(['Other (3)', 'Other (2)', 'Untitled setup']);
    expect(getSetup('Other', store)).toBeNull();
  });

  it('reads documents back through the codec', () => {
    const store = fake(JSON.stringify({ odd: { ...doc, surprise: 1, physics: { cfl: 7 } }, broken: { format: 'something else' }, text: 'not a document' }));
    expect(listSetups(store)).toEqual(['odd', 'broken']);
    const got = getSetup('odd', store)!;
    expect(got.spec.physics?.cfl).toBe(1);
    expect(got.notes.length).toBe(2);
    expect(getSetup('broken', store)).toBeNull();
  });

  it('survives a corrupt store, a missing one, and one that throws', () => {
    for (const bad of ['{not json', '[1, 2]', '"text"', 'null', '']) {
      const store = fake(bad);
      expect(listSetups(store), bad).toEqual([]);
      expect(getSetup('x', store)).toBeNull();
      expect(removeSetup('x', store)).toBe(false);
      // Saving starts the library again.
      expect(saveSetup('Mine', doc, { store })).toBe('Mine');
      expect(listSetups(store)).toEqual(['Mine']);
    }
    expect(listSetups(null)).toEqual([]);
    expect(saveSetup('Mine', doc, { store: null })).toBeNull();
    const refuses: SetupStore = {
      getItem: () => {
        throw new Error('storage is off');
      },
      setItem: () => {
        throw new Error('storage is full');
      },
    };
    expect(listSetups(refuses)).toEqual([]);
    expect(getSetup('Mine', refuses)).toBeNull();
    expect(saveSetup('Mine', doc, { store: refuses })).toBeNull();
    expect(renameSetup('a', 'b', refuses)).toBeNull();
    expect(removeSetup('a', refuses)).toBe(false);
    // No browser here at all: the default store is none.
    expect(listSetups()).toEqual([]);
  });
});

describe('shapes of a group', () => {
  const forms = (path: string) => {
    const c = OPTION_BY_PATH[path].control;
    if (c.type !== 'group' || !c.forms) throw new Error(`${path} has no shapes`);
    return c.forms;
  };

  it('tells units by mass from units by density and velocity', () => {
    expect(formOf(forms('domain.units'), presetSpec('detonation-co')!.domain!.units).value).toBe('flow');
    expect(formOf(forms('domain.units'), presetSpec('polytrope')!.domain!.units).value).toBe('mass');
    expect(formOf(forms('domain.units'), undefined).value).toBe('mass');
  });

  it('asks the network, not fuel to ash burning, for its 13 species and physical units', () => {
    const arrhenius = setup({ physics: { eos: 'ideal', species: ['fuel', 'ash'], burn: { fuel: 'fuel', ash: 'ash', q: 50, ea: 25, k: 1 } } });
    expect(texts(arrhenius, 'error')).toEqual([]);
    const network = setup({ physics: { eos: 'ideal', species: ['fuel', 'ash'], burn: { network: 'aprox13' } } });
    expect(has(texts(network, 'error'), /Network \(aprox13\): needs The 13 species/)).toBe(true);
  });

  it('refuses fuel to ash burning where the species are nuclides that set the equation of state', () => {
    const spec = setup({ physics: { eos: 'ideal', species: ['he4', 'c12'], burn: { fuel: 'he4', ash: 'c12', q: 50, ea: 25, k: 1 } } });
    expect(has(texts(spec, 'error'), /Fuel to ash \(Arrhenius\): not available where every species is a nuclide/)).toBe(true);
  });

  it('wants fuel to ash burning to name species the gas carries', () => {
    const spec = setup({ physics: { eos: 'ideal', species: ['he4'], burn: { fuel: 'fuel', ash: 'ash', q: 50, ea: 25, k: 1 } } });
    const errors = texts(spec, 'error');
    expect(has(errors, /its fuel, "fuel", is not a species/)).toBe(true);
    expect(has(errors, /its ash, "ash", is not a species/)).toBe(true);
  });
});

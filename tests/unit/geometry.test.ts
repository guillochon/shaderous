// Mesh files and signed distance fields (plans/wind-tunnel.md 11.1).
import { describe, expect, it } from 'vitest';
import { apply, axisAngle, buildSdf, flatten, identity, measure, mul, orient, parseAirfoil, parseMeshFile, parseSvgPath, sampleSdf, windingNumber, worldMatrix, type Mesh } from '../../src/geometry';
import { ahmedBody } from '../../src/geometry/builtin';

const { readFileSync }: { readFileSync(p: URL): Uint8Array } = await import(/* @vite-ignore */ 'node:fs' as string);
const fixture = (name: string) => new Uint8Array(readFileSync(new URL(`./fixtures/meshes/${name}`, import.meta.url)));

/** Triangles as sorted strings, each rotated to start at its smallest corner (keeping its winding). */
function canonical(m: Mesh): string[] {
  const f = flatten(m);
  const out: string[] = [];
  for (let t = 0; t < f.indices.length; t += 3) {
    const v = [0, 1, 2].map((k) => Array.from(f.positions.subarray(3 * f.indices[t + k], 3 * f.indices[t + k] + 3)).map((x) => x.toFixed(5)).join(','));
    const s = v.indexOf([...v].sort()[0]);
    out.push([v[s], v[(s + 1) % 3], v[(s + 2) % 3]].join(' | '));
  }
  return out.sort();
}

/** A sphere of radius r: an icosahedron subdivided `levels` times, its vertices pushed out to r. */
function icosphere(levels: number, r = 1): Mesh {
  const t = (1 + Math.sqrt(5)) / 2;
  let v: number[][] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const norm = (p: number[]) => {
    const l = Math.hypot(p[0], p[1], p[2]);
    return p.map((x) => (x / l) * r);
  };
  v = v.map(norm);
  for (let l = 0; l < levels; l++) {
    const mid = new Map<string, number>();
    const m = (a: number, b: number) => {
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      if (!mid.has(k)) {
        v.push(norm(v[a].map((x, i) => (x + v[b][i]) / 2)));
        mid.set(k, v.length - 1);
      }
      return mid.get(k)!;
    };
    f = f.flatMap(([a, b, c]) => {
      const [ab, bc, ca] = [m(a, b), m(b, c), m(c, a)];
      return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
    });
  }
  return { positions: new Float32Array(v.flat()), indices: new Uint32Array(f.flat()), parts: [{ name: 'sphere', first: 0, count: f.length, transform: identity(), parent: -1 }] };
}

/** The field's volume with each point's share by its distance: clamp(1/2 - d/h, 0, 1) h^3. */
const fieldVolume = (d: Float32Array, h: number) => d.reduce((s, x) => s + Math.min(1, Math.max(0, 0.5 - x / h)), 0) * h ** 3;

describe('mesh-parsers', () => {
  const cube = canonical(parseMeshFile('cube.stl', fixture('cube.stl')));
  it('binary and ASCII STL, OBJ with quads, ASCII and binary PLY give the same triangles', () => {
    expect(cube.length).toBe(12);
    for (const f of ['cube_ascii.stl', 'cube.obj', 'cube_ascii.ply', 'cube_bin.ply']) expect(canonical(parseMeshFile(f, fixture(f)))).toEqual(cube);
  });
  it('a closed unit cube: volume 1, area 6', () => {
    const m = measure(flatten(parseMeshFile('cube.stl', fixture('cube.stl'))));
    expect(m.volume).toBeCloseTo(1, 6);
    expect(m.area).toBeCloseTo(6, 6);
  });
  it('a binary STL whose header begins with "solid" is read as binary', () => {
    expect(new TextDecoder().decode(fixture('cube.stl').subarray(0, 5))).toBe('solid');
    expect(parseMeshFile('x.stl', fixture('cube.stl')).indices.length).toBe(36);
  });
  it('OBJ groups become parts', () => {
    const m = parseMeshFile('cube.obj', fixture('cube.obj'));
    expect(m.parts.map((p) => [p.name, p.count])).toEqual([['lower', 6], ['upper', 6]]);
  });
  it('glTF and GLB: the same chain of three nodes, each placed by its parents', () => {
    const a = parseMeshFile('chain.gltf', fixture('chain.gltf'));
    const b = parseMeshFile('chain.glb', fixture('chain.glb'));
    expect(canonical(a)).toEqual(canonical(b));
    expect(a.parts.map((p) => p.name)).toEqual(['base', 'arm', 'tip']);
    expect(a.meters).toBe(1);
    const center = (m: Mesh, node: string) => {
      const f = flatten(m, node);
      const c = [0, 0, 0];
      const n = f.indices.length;
      for (const i of f.indices) for (let k = 0; k < 3; k++) c[k] += f.positions[3 * i + k] / n;
      return c;
    };
    // The arm: 2 along x and turned 90 degrees about z; the tip 1 along the arm's x, so 1 along y.
    const tip = flatten(a, 'tip');
    expect(tip.indices.length).toBe(36);
    const c = center(a, 'tip');
    [2, 1, 0].forEach((x, k) => expect(c[k]).toBeCloseTo(x, 5));
    expect(flatten(a, 'arm').indices.length).toBe(72);
    expect(() => flatten(a, 'wing')).toThrow(/no part named "wing"/);
  });
  it('refuses what it does not read, with the reason', () => {
    expect(() => parseMeshFile('part.step', new TextEncoder().encode('ISO-10303-21;'))).toThrow();
    const draco = new TextEncoder().encode(JSON.stringify({ asset: { version: '2.0' }, extensionsRequired: ['KHR_draco_mesh_compression'] }));
    expect(() => parseMeshFile('x.gltf', draco)).toThrow(/compressed/);
  });
});

describe('outlines', () => {
  it('an airfoil in Selig and in Lednicer order is the same loop', () => {
    const selig = parseAirfoil('NACA 0012\n1 0\n0.5 0.06\n0 0\n0.5 -0.06\n1 0\n');
    const led = parseAirfoil('NACA 0012\n3 3\n\n0 0\n0.5 0.06\n1 0\n\n0 0\n0.5 -0.06\n1 0\n');
    expect(windingNumber(selig, 0.5, 0)).not.toBe(0);
    expect(windingNumber(led, 0.5, 0)).not.toBe(0);
    expect(windingNumber(selig, 0.5, 0.1)).toBe(0);
    expect(led[0].length).toBeGreaterThanOrEqual(8);
  });
  it('an SVG path with a curve: inside and outside', () => {
    const o = parseSvgPath('<svg><path d="M 0 0 L 10 0 Q 10 10 0 10 z"/></svg>');
    expect(windingNumber(o, 2, -2)).not.toBe(0); // y is turned up
    expect(windingNumber(o, 2, 2)).toBe(0);
    expect(() => parseSvgPath('M 0 0 A 5 5 0 0 1 10 0 Z')).toThrow(/arcs/);
  });
});

describe('sdf-sphere', () => {
  it('an icosphere: within the band, the distance is r - R to a fraction of a cell', () => {
    const h = 0.05;
    const { field, report } = buildSdf(icosphere(4), { h });
    let worst = 0;
    let n = 0;
    const [nx, ny, nz] = field.dims;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const d = field.data[i + nx * (j + ny * k)];
          if (Math.abs(d) >= field.band * 0.999) continue;
          const r = Math.hypot(field.origin[0] + i * h, field.origin[1] + j * h, field.origin[2] + k * h);
          worst = Math.max(worst, Math.abs(d - (r - 1)));
          n++;
        }
    expect(n).toBeGreaterThan(1000);
    expect(worst / h).toBeLessThan(0.1);
    expect(sampleSdf(field, 0, 0, 0)).toBeLessThan(0);
    expect(sampleSdf(field, 5, 0, 0)).toBeGreaterThan(0);
    expect(report.stray).toBeLessThan(0.01);
    // The shadow along x: a disk of radius 1.
    expect(report.frontal[0] / Math.PI).toBeCloseTo(1, 1);
  });
});

describe('sdf-volume', () => {
  it('the volume the field holds converges to the mesh’s at second order', () => {
    const mesh = icosphere(4);
    const want = measure(mesh).volume;
    const err = [0.2, 0.1, 0.05].map((h) => Math.abs(fieldVolume(buildSdf(mesh, { h }).field.data, h) / want - 1));
    const order = Math.log2(err[0] / err[2]) / 2;
    expect(order).toBeGreaterThan(1.5);
    expect(err[2]).toBeLessThan(0.01);
  });
});

describe('sdf-broken', () => {
  const h = 0.1;
  const mesh = icosphere(3);
  const clean = buildSdf(mesh, { h }).field;
  /** Points whose side differs from the clean sphere's: all within a cell of the surface. */
  const misplaced = (f: typeof clean) => {
    let far = 0;
    for (let q = 0; q < f.data.length; q++) if (f.data[q] < 0 !== clean.data[q] < 0 && Math.abs(clean.data[q]) > h) far++;
    return far;
  };
  it('a tenth of the triangles removed: the same inside, to within a cell', () => {
    const idx = Array.from(mesh.indices).filter((_, k) => Math.floor(k / 3) % 10 !== 3);
    const holed: Mesh = { ...mesh, indices: new Uint32Array(idx), parts: [{ ...mesh.parts[0], count: idx.length / 3 }] };
    const r = buildSdf(holed, { h });
    expect(misplaced(r.field)).toBe(0);
    expect(r.report.stray).toBeGreaterThan(0.001);
  });
  it('half of the triangles flipped: the same inside, to within a cell', () => {
    const idx = Array.from(mesh.indices);
    for (let t = 0; t < idx.length; t += 6) [idx[t + 1], idx[t + 2]] = [idx[t + 2], idx[t + 1]];
    const flipped: Mesh = { ...mesh, indices: new Uint32Array(idx) };
    expect(misplaced(buildSdf(flipped, { h }).field)).toBe(0);
  });
});

describe('sdf-touching', () => {
  it('three cubes, two of them touching face to face: inside near the shared face, on both sides', () => {
    // chain.glb: the arm's cube at (2, 0, 0), the tip's at (2, 1, 0), touching at y = 1/2.
    const { field } = buildSdf(flatten(parseMeshFile('chain.glb', fixture('chain.glb'))), { h: 0.05 });
    for (const y of [0.42, 0.47, 0.53, 0.58]) for (const x of [1.8, 2, 2.2]) expect(sampleSdf(field, x, y, 0.1)).toBeLessThan(0);
    expect(sampleSdf(field, 2.7, 0.5, 0)).toBeGreaterThan(0);
  });
});

describe('builtin bodies', () => {
  it('the Ahmed-like body is one closed piece, wound outward, of the right size', () => {
    const m = ahmedBody(25);
    const o = orient(m);
    expect(o.flipped).toBe(0);
    expect(o.pieces).toBe(1);
    expect(o.open).toBe(0);
    const { volume } = measure(m);
    // The box less the front's rounding and the slant: a little under L W H.
    expect(volume).toBeGreaterThan(0.85 * 1.044 * 0.389 * 0.288);
    expect(volume).toBeLessThan(1.044 * 0.389 * 0.288);
    const { report } = buildSdf(m, { h: 0.02 });
    // Its shadow along x: the section, W x H (counted on the field's points, h = 0.02: to 10%).
    expect(Math.abs(report.frontal[0] / (0.389 * 0.288) - 1)).toBeLessThan(0.1);
  });
});

describe('sdf-thin', () => {
  it('a plate thinner than a cell is reported as such; a thick one is not', () => {
    const plate = (t: number): Mesh => {
      const V = [[-1, -1, -t / 2], [1, -1, -t / 2], [1, 1, -t / 2], [-1, 1, -t / 2], [-1, -1, t / 2], [1, -1, t / 2], [1, 1, t / 2], [-1, 1, t / 2]];
      const Q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]];
      const f = Q.flatMap((q) => [q[0], q[1], q[2], q[0], q[2], q[3]]);
      return { positions: new Float32Array(V.flat()), indices: new Uint32Array(f), parts: [{ name: 'plate', first: 0, count: 12, transform: identity(), parent: -1 }] };
    };
    const h = 0.1;
    expect(buildSdf(plate(0.03), { h }).report.thinnest).toBeLessThan(1);
    expect(buildSdf(plate(0.03), { h }).report.thinnest).toBeCloseTo(0.3, 2);
    expect(buildSdf(plate(0.5), { h }).report.thinnest).toBeCloseTo(5, 2);
  });
});

describe('body-pose', () => {
  it('a chain of three transforms (tower, nacelle yawed, hub turned) places a point as worked by hand', () => {
    // Tower: 10 up z. Nacelle: yawed 90 degrees about z, 1 forward along its x. Hub: turned 90
    // degrees about its own x. A point 2 along the hub's y.
    const T = (x: number, y: number, z: number) => {
      const m = identity();
      m[12] = x;
      m[13] = y;
      m[14] = z;
      return m;
    };
    const parts = [
      { name: 'tower', first: 0, count: 0, transform: T(0, 0, 10), parent: -1 },
      { name: 'nacelle', first: 0, count: 0, transform: mul(axisAngle([0, 0, 1], 90), T(1, 0, 0)), parent: 0 },
      { name: 'hub', first: 0, count: 0, transform: axisAngle([1, 0, 0], 90), parent: 1 },
    ];
    // By hand: hub turns (0, 2, 0) to (0, 0, 2); nacelle: +1 along x, (1, 0, 2), yawed to (0, 1, 2); tower: (0, 1, 12).
    const p = apply(worldMatrix(parts, 2), 0, 2, 0);
    [0, 1, 12].forEach((x, k) => expect(p[k]).toBeCloseTo(x, 10));
  });
});

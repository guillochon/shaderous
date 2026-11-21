// Outlines for 2D bodies (plans/wind-tunnel.md 3.1): airfoil coordinate files (Selig and Lednicer
// order) and SVG paths, as closed polygons in the plane; and what a run asks of one: is a point
// inside (by its winding number, so that a polygon wound either way, or several, work).

/** Closed polygons, each a flat list x0, y0, x1, y1, ... (the last point joins the first). */
export type Outline = number[][];

/**
 * An airfoil's coordinates. Selig: a name line, then x y from the trailing edge over the upper
 * surface to the leading edge and back along the lower. Lednicer: a name line, a line of the
 * two surfaces' point counts, then the upper surface and the lower, each from the leading edge.
 */
export function parseAirfoil(text: string): Outline {
  const rows = text
    .split(/\r?\n/)
    .map((l) => l.trim().split(/[\s,]+/).map(Number))
    .filter((r) => r.length >= 2 && Number.isFinite(r[0]) && Number.isFinite(r[1]));
  if (rows.length < 3) throw new Error('not an airfoil file: fewer than three points');
  // Lednicer: its first numeric row holds the counts (both above 1, so it is not a point in the chord).
  if (rows[0][0] > 1.5 && rows[0][1] > 1.5) {
    const [nu, nl] = [Math.round(rows[0][0]), Math.round(rows[0][1])];
    const upper = rows.slice(1, 1 + nu);
    const lower = rows.slice(1 + nu, 1 + nu + nl);
    if (upper.length !== nu || lower.length !== nl) throw new Error('airfoil (Lednicer order): fewer points than its counts');
    const loop = [...upper.reverse(), ...lower.slice(lower[0][0] === upper[upper.length - 1][0] && lower[0][1] === upper[upper.length - 1][1] ? 1 : 0)];
    return [loop.flatMap((r) => [r[0], r[1]])];
  }
  return [rows.flatMap((r) => [r[0], r[1]])];
}

/**
 * An SVG path's `d` (or the first `<path d="...">` of an SVG file): M, L, H, V, C, S, Q, T and Z,
 * absolute and relative, curves flattened into `segments` pieces each. Arcs are refused. y is
 * turned up (SVG's points down).
 */
export function parseSvgPath(text: string, segments = 16): Outline {
  const tag = text.match(/<path[^>]*\sd\s*=\s*"([^"]*)"/s);
  const d = tag ? tag[1] : text;
  const tokens = d.match(/[a-zA-Z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/g) ?? [];
  const out: Outline = [];
  let cur: number[] = [];
  let x = 0, y = 0, sx = 0, sy = 0;
  let cx = 0, cy = 0; // the last control point, for S and T
  let cmd = '';
  let k = 0;
  const num = () => {
    const v = Number(tokens[k++]);
    if (!Number.isFinite(v)) throw new Error('SVG path: a number was expected');
    return v;
  };
  const line = (nx: number, ny: number) => {
    cur.push(nx, -ny);
    x = nx;
    y = ny;
  };
  const curve = (pts: number[][]) => {
    // Bezier of the given control points (current point first), flattened.
    for (let s = 1; s <= segments; s++) {
      const t = s / segments;
      let p = pts.map((q) => q.slice());
      while (p.length > 1) p = p.slice(1).map((q, i) => [p[i][0] + t * (q[0] - p[i][0]), p[i][1] + t * (q[1] - p[i][1])]);
      cur.push(p[0][0], -p[0][1]);
    }
    x = pts[pts.length - 1][0];
    y = pts[pts.length - 1][1];
  };
  const finish = () => {
    if (cur.length >= 6) out.push(cur);
    cur = [];
  };
  while (k < tokens.length) {
    if (/[a-zA-Z]/.test(tokens[k])) cmd = tokens[k++];
    const rel = cmd === cmd.toLowerCase();
    const ox = rel ? x : 0;
    const oy = rel ? y : 0;
    switch (cmd.toUpperCase()) {
      case 'M': {
        finish();
        const nx = ox + num(), ny = oy + num();
        sx = nx;
        sy = ny;
        line(nx, ny);
        cmd = rel ? 'l' : 'L';
        break;
      }
      case 'L': line(ox + num(), oy + num()); break;
      case 'H': line(ox + num(), y); break;
      case 'V': line(x, oy + num()); break;
      case 'C': {
        const p1 = [ox + num(), oy + num()], p2 = [ox + num(), oy + num()], p3 = [ox + num(), oy + num()];
        curve([[x, y], p1, p2, p3]);
        [cx, cy] = p2;
        break;
      }
      case 'S': {
        const p1 = [2 * x - cx, 2 * y - cy], p2 = [ox + num(), oy + num()], p3 = [ox + num(), oy + num()];
        curve([[x, y], p1, p2, p3]);
        [cx, cy] = p2;
        break;
      }
      case 'Q': {
        const p1 = [ox + num(), oy + num()], p2 = [ox + num(), oy + num()];
        curve([[x, y], p1, p2]);
        [cx, cy] = p1;
        break;
      }
      case 'T': {
        const p1 = [2 * x - cx, 2 * y - cy], p2 = [ox + num(), oy + num()];
        curve([[x, y], p1, p2]);
        [cx, cy] = p1;
        break;
      }
      case 'Z':
        x = sx;
        y = sy;
        finish();
        break;
      case 'A':
        throw new Error('SVG path: arcs (A) are not read yet: convert them to curves (Inkscape: Path > Object to Path)');
      default:
        throw new Error(`SVG path: unknown command "${cmd}"`);
    }
    if (!'CSQT'.includes(cmd.toUpperCase())) [cx, cy] = [x, y];
  }
  finish();
  if (!out.length) throw new Error('SVG: no closed path was found');
  return out;
}

/** Bounding box of an outline: [xmin, ymin, xmax, ymax]. */
export function outlineBounds(o: Outline): [number, number, number, number] {
  const b: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const p of o)
    for (let i = 0; i < p.length; i += 2) {
      b[0] = Math.min(b[0], p[i]);
      b[1] = Math.min(b[1], p[i + 1]);
      b[2] = Math.max(b[2], p[i]);
      b[3] = Math.max(b[3], p[i + 1]);
    }
  return b;
}

/** Winding number of the outline about (x, y): nonzero inside. */
export function windingNumber(o: Outline, x: number, y: number): number {
  let w = 0;
  for (const p of o) {
    const n = p.length / 2;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const ay = p[2 * i + 1], by = p[2 * j + 1];
      const side = (p[2 * j] - p[2 * i]) * (y - ay) - (x - p[2 * i]) * (by - ay);
      if (ay <= y) {
        if (by > y && side > 0) w++;
      } else if (by <= y && side < 0) w--;
    }
  }
  return w;
}

/** Signed area of the outline (positive counterclockwise). */
export function outlineArea(o: Outline): number {
  let a = 0;
  for (const p of o) {
    const n = p.length / 2;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      a += p[2 * i] * p[2 * j + 1] - p[2 * j] * p[2 * i + 1];
    }
  }
  return a / 2;
}

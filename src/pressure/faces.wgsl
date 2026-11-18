// What lies across a face of a cell, for the pressure passes: gas, something nothing crosses (a
// wall of the box, the surface of an obstacle), an inlet (a cell held at its state: the mass
// flux through the face is that cell's), or an open face of the box (zero pressure at the face).
// Requires P and U2S, the state texture that holds the cells' marks.
const FK_GAS = 0;
const FK_WALL = 1;
const FK_INLET = 2;
const FK_OPEN = 3;
// `q` is the cell across (gas, inlet); `only`: an open face that lets gas out and none in.
struct PFace { kind: i32, q: vec3<i32>, only: bool }

fn pface(c: vec3<i32>, a: i32, up: bool) -> PFace {
  let n = vec3<i32>(P.dims.xyz);
  var q = c;
  q[a] += select(-1, 1, up);
  if (q[a] < 0 || q[a] >= n[a]) {
    let k = select(P.bc_lo[a], P.bc_hi[a], up);
    if (k == BC_PERIODIC) {
      q[a] = select(n[a] - 1, 0, up);
    } else if (bc_wall(k)) {
      return PFace(FK_WALL, c, false);
    } else {
      return PFace(FK_OPEN, c, k == BC_OUTFLOW_ONLY);
    }
  }
  let m = textureLoad(U2S, q, 0).w;
  if (m > 0.5) { return PFace(FK_WALL, c, false); }
  if (m < -0.5) { return PFace(FK_INLET, q, false); }
  return PFace(FK_GAS, q, false);
}

// The momentum an open face carries: the cell's own, outward only where the face lets none in.
fn open_mom(m: f32, up: bool, only: bool) -> f32 {
  if (!only) { return m; }
  return select(min(m, 0.0), max(m, 0.0), up);
}

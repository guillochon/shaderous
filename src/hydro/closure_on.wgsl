// The <closure> slot with the turbulence closure on (closure/closure.ts): the coefficients
// hydro/closure.wgsl wrote for the step, per cell.
const CLOSURE = true;
@group(0) @binding(48) var CLOS: texture_3d<f32>;

// A cell's (a, b). A cell beyond a face of the box that is not periodic has none: the closure
// leaves the fluxes through walls and open faces alone.
fn closure_cell(c: vec3<i32>) -> vec2<f32> {
  let n = grid_n();
  if (all(c >= vec3<i32>(0)) && all(c < n)) { return textureLoad(CLOS, c, 0).xy; }
  var m = c;
  for (var a = 0; a < 3; a++) {
    if (c[a] >= 0 && c[a] < n[a]) { continue; }
    if (select(P.bc_lo[a], P.bc_hi[a], c[a] >= n[a]) != 0u) { return vec2<f32>(0.0); }
    m[a] = ((c[a] % n[a]) + n[a]) % n[a];
  }
  return textureLoad(CLOS, m, 0).xy;
}

// The closure on the face between cells a and b: the smaller of the two cells' coefficients,
// so that a cell whose gate is shut (a shock) shuts every one of its faces. Both cells of the
// face read the same two texels, so they scale its jumps alike and compute the same flux.
fn closure_face(a: vec3<i32>, b: vec3<i32>) -> vec2<f32> {
  return min(closure_cell(a), closure_cell(b));
}

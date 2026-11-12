// Shared by the species passes on a uniform grid (hydro/species_predict.wgsl and
// hydro/species.wgsl): the species at the start of the step and their slopes.
//
// Groups of four are stacked along z: group g of cell c is the texel c + (0, 0, g nz).
#include "hydro/common.wgsl"
#include <species_n>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;   // state at the start of the step
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(5) var S: texture_3d<f32>;    // rho X at the start of the step

fn group_at(c: vec3<i32>, g: u32) -> vec3<i32> {
  return c + vec3<i32>(0, 0, i32(g * P.dims.z));
}

fn held(c: vec3<i32>) -> bool {
  return has_solids() && abs(textureLoad(U2, c, 0).w) > 0.5;
}

// Mass fractions of a cell, or of the cell a ghost maps to.
struct Cell { x: array<vec4<f32>, NG>, rho: f32 }
fn load_cell(c: vec3<i32>) -> Cell {
  let m = bc_map(c).c;
  var o: Cell;
  o.rho = max(textureLoad(U1, m, 0).x, P.rho_floor);
  let inv = 1.0 / o.rho;
  for (var g = 0u; g < NG; g++) { o.x[g] = textureLoad(S, group_at(m, g), 0) * inv; }
  return o;
}

const SLOPE_FLOOR: f32 = 1e-6;

// Monotonized-central slope.
fn mc(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  let c = 0.5 * (a + b);
  let s = sign(c) * min(min(2.0 * abs(a), 2.0 * abs(b)), abs(c));
  return select(vec4<f32>(0.0), s, a * b > vec4<f32>(0.0));
}

// Slopes of every species of cell o between its neighbors m and p, with one limiter for all:
// the central slopes, scaled by the smallest ratio of any species' monotonized-central slope to
// its central one. Central slopes sum to zero (the fractions sum to one in every cell), so the
// scaled ones do too, and each species stays within its own limit. (Limiting each species on
// its own leaves face values that do not sum to one; scaling them back, as consistent
// multi-fluid advection does (Plewa & Mueller 1999), changes a species that is the same
// everywhere.)
fn slopes(m: Cell, o: Cell, p: Cell) -> array<vec4<f32>, NG> {
  var phi = 1.0;
  for (var g = 0u; g < NG; g++) {
    let a = o.x[g] - m.x[g];
    let b = p.x[g] - o.x[g];
    let c = 0.5 * (a + b);
    // Differences below SLOPE_FLOOR are rounding (X is rho X / rho in single precision) and
    // change sign from cell to cell: counted, a species that is the same everywhere would stop
    // every slope. A species that varies by less than that may overshoot by as little.
    let real = max(abs(a), abs(b)) > vec4<f32>(SLOPE_FLOOR);
    let r = select(vec4<f32>(1.0), mc(a, b) / c, real & (abs(c) > vec4<f32>(1e-30)));
    phi = min(phi, min(min(r.x, r.y), min(r.z, r.w)));
  }
  var s: array<vec4<f32>, NG>;
  for (var g = 0u; g < NG; g++) { s[g] = 0.5 * phi * (p.x[g] - m.x[g]); }
  return s;
}

fn unit(axis: i32) -> vec3<i32> {
  var e = vec3<i32>(0);
  e[axis] = 1;
  return e;
}

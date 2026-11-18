// A shearing box with a magnetic field (plans/coordinate-systems-plan.md D4; docs/numerics.md
// section 22): what the two x planes of the box compute for the same face or edge, made to
// agree. Each side works out its boundary faces' fluxes and edges' fields from its own ghosts,
// the other side's cells sheared along y; the two evaluations differ. As in the hydrodynamic
// update (hydro/update.wgsl shear_face), each is replaced by the mean of its own and the other
// side's, interpolated along y at the boundary's offset and boosted by the jump of the
// background flow, q Omega L:
//
//   fluxes: mass, momentum, energy and the advected scalars through the x faces at x = 0 and
//           x = n, so that the mass and the x and z momentum leaving one side enter the other;
//   edges:  the impulses (dt/dx) E_y and E_z on the two x planes (Athena's remap of the
//           electric field), so that the faces around them change consistently: the b_x of the
//           two planes stay sheared images of each other, the divergence stays at round-off,
//           and the net vertical flux through the box (the sum of b_z) is conserved. E_z takes
//           the frame change, u B_x, from -v x B with v_y boosted by u; E_y has no v_y in it.
//
// Two dispatches each, as the same buffer is read and written: MODE 0 (fluxes) and 2 (edges)
// put the means in the scratch buffer SH, MODE 1 and 3 copy them back. One invocation per
// (row j, layer, side). Runs after mhd/flux.wgsl and after mhd/emf.wgsl, slab by slab.
#include "hydro/common.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> FX: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> EMF: array<vec4<f32>>;
@group(0) @binding(14) var<uniform> SLAB: Slab;
@group(0) @binding(19) var<storage, read_write> SH: array<vec4<f32>>;

override STAGE: u32 = 1u;
override MODE: u32 = 0u;

#include "mhd/flux_index.wgsl"

// The other plane's row at row j of this one, in rows: J = j + offset on the lower plane.
fn partner_row(j: i32, upper: bool) -> f32 {
  return f32(j) + select(-g_shear, g_shear, upper);
}
fn wrap_y(j: i32) -> i32 {
  let ny = i32(P.dims.y);
  return ((j % ny) + ny) % ny;
}

// A face's (mass, momentum) and (energy, internal energy, scalar, face velocity), seen from a
// frame moving at u along y.
fn boost_a(a: vec4<f32>, u: f32) -> vec4<f32> {
  return vec4<f32>(a.x, a.y, a.z + u * a.x, a.w);
}
fn boost_b(a: vec4<f32>, b: vec4<f32>, u: f32) -> vec4<f32> {
  return vec4<f32>(b.x + u * a.z + 0.5 * u * u * a.x, b.yzw);
}

// Layers this slab has: cells for the fluxes, edge layers (one more in 3D) for the edges.
fn layers(edges: bool) -> i32 {
  if (P.dims.z == 1u) { return 1; }
  return SLAB.nz + select(0, 1, edges);
}
fn layer_z(l: i32) -> i32 {
  return select(SLAB.z0 + l, 0, P.dims.z == 1u);
}

// B_x on the plane at texel x, at the y node j - 1/2 (between rows j - 1 and j), layer z.
fn bx_node(x: i32, j: i32, z: i32) -> f32 {
  let k = min(z, i32(P.dims.z) - 1);
  return 0.5 * (textureLoad(BF, vec3<i32>(x, wrap_y(j - 1), k), 0).x + textureLoad(BF, vec3<i32>(x, wrap_y(j), k), 0).x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = vec3<i32>(P.dims.xyz);
  let edges = MODE >= 2u;
  let rows = n.y + select(0, 1, edges);
  let j = i32(gid.x);
  let l = i32(gid.y);
  let side = i32(gid.z);
  if (j >= rows || l >= layers(edges) || side > 1) { return; }
  g_shear = select(clock.shear, shear_half(clock.shear, clock.dt), STAGE == 2u) / P.dx;
  let upper = side == 1;
  let z = layer_z(l);
  let x = select(0, n.x, upper);
  let xp = select(n.x, 0, upper);
  let u = select(1.0, -1.0, upper) * shear_jump();
  let slot = u32(((side * layers(edges) + l) * rows + j) * 2);

  if (MODE == 0u) {
    let i = 3u * flux_index(vec3<i32>(x, j, z));
    let J = partner_row(j, upper);
    let s = floor(J);
    let e = J - s;
    let i0 = 3u * flux_index(vec3<i32>(xp, wrap_y(i32(s)), z));
    let i1 = 3u * flux_index(vec3<i32>(xp, wrap_y(i32(s) + 1), z));
    let a0 = FX[i0];
    let a1 = FX[i1];
    let w0 = 0.5 * (1.0 - e);
    let w1 = 0.5 * e;
    SH[slot] = 0.5 * FX[i] + w0 * boost_a(a0, u) + w1 * boost_a(a1, u);
    SH[slot + 1u] = 0.5 * FX[i + 1u] + w0 * boost_b(a0, FX[i0 + 1u], u) + w1 * boost_b(a1, FX[i1 + 1u], u);
  } else if (MODE == 1u) {
    let i = 3u * flux_index(vec3<i32>(x, j, z));
    FX[i] = SH[slot];
    FX[i + 1u] = SH[slot + 1u];
  } else if (MODE == 2u) {
    // Rows 0 to n - 1; row n (the same edge as row 0 on the periodic y axis) is copied from row
    // 0 by MODE 3.
    if (j >= n.y) { return; }
    let own = EMF[emf_index(vec3<i32>(x, j, z))];
    let J = partner_row(j, upper);
    let s = floor(J);
    let e = J - s;
    let j0 = wrap_y(i32(s));
    let j1 = wrap_y(i32(s) + 1);
    let p0 = EMF[emf_index(vec3<i32>(xp, j0, z))];
    let p1 = EMF[emf_index(vec3<i32>(xp, j1, z))];
    // The stage's impulse (dt/dx) E_z of the frame change.
    let dt = select(clock.dt, 0.5 * clock.dt, STAGE == 1u);
    let k = (dt / P.dx) * u;
    let z0 = p0.z + k * bx_node(xp, j0, z);
    let z1 = p1.z + k * bx_node(xp, j1, z);
    let w0 = 0.5 * (1.0 - e);
    let w1 = 0.5 * e;
    SH[slot] = vec4<f32>(own.x, 0.5 * own.y + w0 * p0.y + w1 * p1.y, 0.5 * own.z + w0 * z0 + w1 * z1, own.w);
  } else {
    let src = u32(((side * layers(edges) + l) * rows + select(j, 0, j == n.y)) * 2);
    EMF[emf_index(vec3<i32>(x, j, z))] = SH[src];
  }
}

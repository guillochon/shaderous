// One radiation sweep on the uniform grid (radiation/radiation.ts; plan section 4.3): explicit
// transport of (E, F) between cells, then the local implicit exchange of energy and momentum
// with the gas. Run P.rad2.x times per hydro step, each over clock.dt / P.rad2.x. Float64
// mirror: RefRad1D.sweep in reference/radiation.ts.
//
// Each cell computes the fluxes through all of its faces itself, from the four cells across
// each face, so the two cells of a face get the same bits (docs/architecture.md, gathered
// fluxes). The face flux is that of the full speed of light; each cell multiplies its own
// divergence by its own reduction factor r. The HLL dissipation is scaled by 1 / tau of the face
// (Fornax), so in opaque gas the scheme diffuses at c / (3 kappa rho) whatever the cells resolve.
//
// The gas's side collects in G = (rho e, momentum gained), which radiation/commit.wgsl folds into
// the state after the last sweep; the first sweep takes rho e from the state itself.
#include "hydro/common.wgsl"
#include <eos>
#include <opacity>
#include "radiation/common.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;    // the gas: (rho, rho v), fixed through the sweeps
@group(0) @binding(3) var U2: texture_3d<f32>;    // (E, rho e, rho s, mark)
@group(0) @binding(4) var RI: texture_3d<f32>;    // radiation (E, F) before the sweep
@group(0) @binding(5) var GI: texture_3d<f32>;    // the gas's side: (rho e, momentum gained)
@group(0) @binding(6) var RO: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var GO: texture_storage_3d<rgba32float, write>;
struct SweepInfo { first: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(8) var<uniform> SW: SweepInfo;

#include "radiation/face_flux.wgsl"

// The internal energy of the gas of cell m as the sweeps have it.
fn gas_eint(m: vec3<i32>) -> f32 {
  if (SW.first != 0u) { return textureLoad(U2, m, 0).y; }
  return textureLoad(GI, m, 0).x;
}

// A cell, or the cell a ghost maps to: the flux reversed across a wall as the velocity is.
fn rad_load(c: vec3<i32>) -> RCell {
  let m = bc_map(c);
  let R = textureLoad(RI, m.c, 0);
  let rho = max(textureLoad(U1, m.c, 0).x, P.rho_floor);
  comp_use(comp_tex(m.c));
  let T = eos_temperature(rho, gas_eint(m.c));
  return RCell(R.x, R.yzw * m.flip, kappa_rosseland(rho, T) * rho);
}

#include "radiation/exchange.wgsl"

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (any(gid >= P.dims.xyz)) { return; }
  let c = vec3<i32>(gid);
  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);
  let R = textureLoad(RI, c, 0);
  // Solid and fixed cells keep their radiation (a fixed cell is an inflow), and their gas.
  if (has_solids() && abs(u2.w) > 0.5) {
    textureStore(RO, c, R);
    textureStore(GO, c, vec4<f32>(gas_eint(c), select(textureLoad(GI, c, 0).yzw, vec3<f32>(0.0), SW.first != 0u)));
    return;
  }
  g_dx = P.dx;
  let dt = clock.dt / P.rad2.x;
  let flat = P.dims.z == 1u;
  g_inner = stencil_inside(c, 2);
  let q = rad_load(c);
  var divE = 0.0;
  var divF = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    if (a == 2 && flat) { break; }
    var e = vec3<i32>(0);
    e[a] = 1;
    let m2 = rad_load(c - 2 * e);
    let m1 = rad_load(c - e);
    let p1 = rad_load(c + e);
    let p2 = rad_load(c + 2 * e);
    let lo = face_flux(a, P.dx, m2, m1, q, p1);
    let hi = face_flux(a, P.dx, m1, q, p1, p2);
    divE += hi.E - lo.E;
    divF += hi.F - lo.F;
  }
  let rho = max(u1.x, P.rho_floor);
  comp_use(comp_tex(c));
  let eint = gas_eint(c);
  let T = eos_temperature(rho, eint);
  let kR = kappa_rosseland(rho, T);
  let kP = kappa_planck(rho, T);
  let r = rad_reduction(kR * rho * P.dx);
  let Es = max(R.x - r * dt * divE / P.dx, 0.0);
  let Fs = R.yzw - r * dt * divF / P.dx;
  let x = rad_exchange(rho, eint, Es, Fs, kP, kR, r, dt);
  // The work of the radiation's force on the gas, from the radiation, at the gas's velocity half
  // way through this sweep's kick: summed over the sweeps it is the kinetic energy the commit
  // adds exactly, so E_gas + E / r is kept.
  let gained = select(textureLoad(GI, c, 0).yzw, vec3<f32>(0.0), SW.first != 0u);
  let v = (u1.yzw + gained + 0.5 * x.dmom) / rho;
  var E = max(x.E - r * dot(v, x.dmom), 0.0);
  // A flux larger than c E is no state: it is cut back to c E (it was only by round-off).
  var F = x.F;
  let fm = length(F);
  if (fm > rad_c() * E) { F *= rad_c() * E / fm; }
  textureStore(RO, c, vec4<f32>(E, F));
  textureStore(GO, c, vec4<f32>(x.eint, gained + x.dmom));
}

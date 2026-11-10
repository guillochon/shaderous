// Unsplit MUSCL-Hancock update on a uniform grid, gather form: each cell evaluates its own six
// face fluxes from the half-step states written by hydro/predict.wgsl and a reconstruction
// recomputed from the primitives, which are themselves converted from the conserved state on
// load (see hydro/recon.wgsl).
//
// The stencil is the cell and two neighbours each way along each axis (13 cells): face states
// of a neighbour need that neighbour's slope along the face normal only.
//
// Conservation depends on the two cells sharing a face computing identical fluxes: a face is
// the upper face of one cell and the lower face of the next, which are different call sites
// of face_flux(). Its inputs are pure functions of the cell coordinates, so the results should
// match bit for bit, and the validation suite checks that they do (DEBUG_FLUX below).
//
// Performance notes (D3D12, RTX 3080): keep this straight-line code on locals. Caching the
// stencil in a shader-local array was 3x slower than repeating the texture loads.
#include "hydro/common.wgsl"
#include <eos>
#include <limiter>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"
#include "hydro/fluff.wgsl"

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(6) var G: texture_3d<f32>;
@group(0) @binding(7) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(8) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(9) var PA: texture_3d<f32>;
@group(0) @binding(10) var PB: texture_3d<f32>;

#include "hydro/region.wgsl"
#include "hydro/recon.wgsl"
#include "gravity/gfield.wgsl"

// Half-step state of a cell. A cell beyond a boundary takes the state of the cell it maps to
// (mirrored if reflecting).
fn load_pred(c: vec3<i32>) -> Q {
  if (INTERIOR) { return Q(textureLoad(PA, c, 0), textureLoad(PB, c, 0)); }
  let m = bc_map(c);
  let a = textureLoad(PA, m.c, 0);
  return Q(vec4<f32>(a.x, bc_velocity(a.yzw, m)), textureLoad(PB, m.c, 0));
}

#include "hydro/flux.wgsl"
#include "hydro/advance.wgsl"

// Validation hook: when set, the pass writes the six face mass fluxes instead of the new state,
// so the test suite can check that both cells sharing a face computed the same bits.
override DEBUG_FLUX: bool = false;

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rc = region_cell(gid);
  if (!rc.mine) { return; }
  g_dx = P.dx;
  let c = rc.c;
  let ex = vec3<i32>(1, 0, 0);
  let ey = vec3<i32>(0, 1, 0);
  let ez = vec3<i32>(0, 0, 1);

  // In 2D the single layer is its own neighbour along z: the two z fluxes are equal and cancel,
  // so they are not computed (and no load leaves the layer, which keeps g_inner valid).
  let flat = P.dims.z == 1u;
  g_inner = stencil_inside(c, 2);
  let w0 = load_q(c);
  let p0 = load_pred(c);
  let fx = axis_fluxes(c, ex, 0, w0, p0);
  let fy = axis_fluxes(c, ey, 1, w0, p0);
  var fz = Faces(flux_zero(), flux_zero());
  if (!flat) { fz = axis_fluxes(c, ez, 2, w0, p0); }

  if (DEBUG_FLUX) {
    textureStore(O1, c, vec4<f32>(fx.lo.m, fx.hi.m, fy.lo.m, fy.hi.m));
    textureStore(O2, c, vec4<f32>(fz.lo.m, fz.hi.m, 0.0, 0.0));
    return;
  }

  // Flux-form gravitational work (P.flags bit 0): each face hands half of F_mass * dphi to the
  // two cells it separates, so the sum over cells telescopes to -d/dt of the potential energy.
  let flux_work = has_gravity() && (P.flags & 1u) != 0u;
  var dw = 0.0;
  if (flux_work) {
    // (Giving each face's work wholly to the downstream cell was tried; it conserves just as
    // well in principle but heats low-density cells more and was worse on the Evrard test.)
    dw = -0.5 * (fx.hi.m * dphi_half(c, ex) - fx.lo.m * dphi_half(c, -ex)
               + fy.hi.m * dphi_half(c, ey) - fy.lo.m * dphi_half(c, -ey)
               + fz.hi.m * dphi_half(c, ez) - fz.lo.m * dphi_half(c, -ez));
  }

  let u1 = textureLoad(U1, c, 0);
  let u2 = textureLoad(U2, c, 0);

  // Gravity is kick-drift-kick around the flux update: half a kick with the old potential here,
  // the other half in gravity/correct.wgsl once the new potential is known. (Applying the whole
  // kick after the fluxes would drain 1/2 rho g^2 dt^2 of internal energy per step from gas in
  // hydrostatic balance.) The energy source is either the flux-form work accumulated above, or
  // exactly the kinetic energy the kick adds, k . (m + k/2) / rho, which keeps gravity out of
  // the internal energy at the price of total-energy conservation.
  var kick = vec3<f32>(0.0);
  var ekick = 0.0;
  if (has_gravity()) {
    kick = 0.5 * clock.dt * u1.x * g_load(c);
    if (!flux_work) { ekick = dot(kick, u1.yzw + 0.5 * kick) / max(u1.x, P.rho_floor); }
  }

  let nw = advance(u1, u2, net_inflow(fx, fy, fz), dw, p0.b.x, kick, ekick);
  textureStore(O1, c, nw.u1);
  textureStore(O2, c, nw.u2);
}

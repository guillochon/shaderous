// MHD update on the uniform grid, for one stage of the van Leer integrator: the conservative
// update of each cell from the face fluxes of mhd/flux.wgsl, and the constrained-transport
// update of its lower faces from the edge impulses I = (dt/dx) E of mhd/emf.wgsl,
//   b_a += -(I_c(t + e_b) - I_c(t)) + (I_b(t + e_c) - I_b(t)),  b = a + 1, c = a + 2,
// the discrete form of dB/dt = -curl E.
//
// Divergence to round-off, for good. Each impulse is one number, stored once, that the four
// faces around its edge add with opposite signs, so in exact arithmetic the divergence of every
// cell never changes. In float32 each face's sum rounds on its own, and over many steps the
// rounding would walk the divergence away from zero. So each face is kept as a pair: the field
// b and a compensation c holding what b's rounding has dropped (Kahan; the second face texture),
// and the impulses are added to the pair with error-free sums (Knuth's TwoSum). The pair then
// carries the face's exact sum of impulses, to float32 rounding of the tiny c, and the
// divergence of b stays within a few roundings of where it started instead of growing.
//
//   STAGE 1: half a step from the state at the start of the step: the half-step state. Its
//            second texture's fourth channel holds the shock flag that stage 2 reads (HLLD falls
//            back to HLLE next to shocks).
//   STAGE 2: a full step from the state at the start, with fluxes of the half-step state, and
//            the new compensations.
//
// Gravity (self-gravity, the uniform field, point masses and the frame; docs/numerics.md section 21):
// stage 1 adds the whole source over its half step, so the half-step state the stage-2 fluxes
// come from feels it; stage 2 adds the first half of the step's kick, (dt/2) rho^n g^n, and
// gravity/correct.wgsl the second, with the potential of the updated state. The kick's energy
// is its kinetic energy, or with flux-form work (P.flags bit 0) -1/2 sum F_mass dphi over the
// faces, as in hydro/update.wgsl. Cooling acts at the end of stage 2.
//
// One invocation per texel t in [0, n] along x and y and the slab's layers along z (and the
// grid's upper z faces on the last slab; mhd/flux_index.wgsl): the cell t if it is in the grid,
// and the faces texel t holds that exist. The internal energy's p div v takes the pressure at
// the start of the step in stage 1 and at the half step in stage 2.
#include "hydro/common.wgsl"
#include "hydro/drive.wgsl"
#include <eos>
#include "hydro/dual_energy.wgsl"
#include "hydro/pfloor.wgsl"
#include "hydro/prim.wgsl"
#include "hydro/fluff.wgsl"
#include <cooling>
#include <frame>

@group(0) @binding(1) var<storage, read> clock: Clock;
@group(0) @binding(2) var U1: texture_3d<f32>;   // state at the start of the step
@group(0) @binding(3) var U2: texture_3d<f32>;
@group(0) @binding(4) var BF: texture_3d<f32>;
@group(0) @binding(5) var<storage, read> FX: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> FY: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> FZ: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> EMF: array<vec4<f32>>;
@group(0) @binding(9) var O1: texture_storage_3d<rgba32float, write>;
@group(0) @binding(10) var O2: texture_storage_3d<rgba32float, write>;
@group(0) @binding(11) var OB: texture_storage_3d<rgba32float, write>;
@group(0) @binding(12) var H1: texture_3d<f32>;   // half-step state (stage 2)
@group(0) @binding(13) var H2: texture_3d<f32>;
@group(0) @binding(14) var<uniform> SLAB: Slab;
@group(0) @binding(15) var BC: texture_3d<f32>;   // compensations of the faces at the start
@group(0) @binding(16) var OC: texture_storage_3d<rgba32float, write>;   // new ones (stage 2)
@group(0) @binding(17) var G: texture_3d<f32>;   // self-gravity's field: acceleration, potential in w
// With species: stage 2 records each face's mass flux for the species pass (hydro/species.wgsl;
// the layout of hydro/species_on.wgsl's MF: face (axis, c) the lower face of cell c along it, c
// up to dims inclusive). A placeholder without species.
@group(0) @binding(18) var<storage, read_write> MF: array<f32>;
override SPECIES_FLUX: bool = false;
// The redo's round (mhd/fofc.wgsl): round 0 redoes level 0 and lists the neighbors it leaves all but
// emptied; round 1 redoes every level and lists nothing.
override FOFC_LIST_MORE: bool = false;
// The first-order flux correction (mhd/fofc.wgsl): stage 2 lists the cells it has all but emptied
// (main); redo_main updates them and their neighbors again, with the faces of the field as main
// wrote them (NF, the new parity's).
@group(0) @binding(34) var<storage, read_write> FOFC: Fofc;
@group(0) @binding(37) var NF: texture_3d<f32>;

override STAGE: u32 = 1u;

#include "gravity/ext_bind.wgsl"
#include "gravity/gfield.wgsl"
struct NewState { u1: vec4<f32>, u2: vec4<f32> }
#include "hydro/damping.wgsl"
#include "hydro/cooling.wgsl"

#include "mhd/common.wgsl"
#include "mhd/flux_index.wgsl"
#include "mhd/fofc.wgsl"

// Viscous heating (hydro/diffusion_core.wgsl): the internal energy, advected on its own, takes
// the heat the viscous fluxes of mhd/flux.wgsl leave in the cell, from the faces' stresses
// worked out again here from the same state (the stage's: the start in stage 1, where H holds
// it, the half step in stage 2).
fn load_vel(c: vec3<i32>) -> vec3<f32> {
  let m = bc_map(c);
  let u = textureLoad(H1, m.c, 0);
  return bc_velocity(u.yzw / max(u.x, P.rho_floor), m);
}
fn load_comp(c: vec3<i32>) -> vec4<f32> {
  return comp_tex(bc_map(c).c);
}
fn face_rim(a: vec3<i32>, axis: i32, t: i32) -> i32 {
  return 0;
}
#include "hydro/diffusion_core.wgsl"
// The Ohmic heat (mhd/ohm.wgsl), from the field at the start of the step in both stages.
fn load_b(c: vec3<i32>) -> vec3<f32> {
  let m = bc_map(c);
  return bcell(m.c) * m.bflip;
}
#include "mhd/ohm.wgsl"
fn visc_side(c: vec3<i32>) -> DSide {
  let m = bc_map(c);
  let u = textureLoad(H1, m.c, 0);
  let rho = max(u.x, P.rho_floor);
  return DSide(rho, bc_velocity(u.yzw / rho, m), 0.0, false);
}
// The heat of cell t, per unit dt/dx.
fn visc_heating(t: vec3<i32>, axes: i32) -> f32 {
  let c = visc_side(t);
  var heat = 0.0;
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    heat += visc_heat_d(visc_flux(t - e, e, a, visc_side(t - e), c), c.v);
    heat -= visc_heat_d(visc_flux(t, e, a, c, visc_side(t + e)), c.v);
  }
  return heat;
}

fn impulse(t: vec3<i32>) -> vec3<f32> {
  return EMF[emf_index(t)].xyz;
}

// Error-free sum (Knuth): s + e == a + b exactly, s the rounded sum. Through opaque() so that
// a compiler that reassociates (Metal) cannot reduce e to zero.
fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = opaque(a + b);
  let bb = opaque(s - a);
  // Each error term through opaque() too: (a - x) + (b - bb) is otherwise free to be regrouped
  // as (a + b) - (x + bb), which loses the very rounding this recovers. Firefox's shader compiler
  // did (div B grew as an uncompensated sum's, 4e-6 after 751 steps of Orszag-Tang).
  return vec2<f32>(s, opaque(a - opaque(s - bb)) + opaque(b - bb));
}

// The face along a of texel t after the stage, as (b, c): the pair at the start plus the
// impulses of the four edges around the face.
fn new_face(a: i32, t: vec3<i32>) -> vec2<f32> {
  let b = (a + 1) % 3;
  let c = (a + 2) % 3;
  var eb = vec3<i32>(0);
  eb[b] = 1;
  var ec = vec3<i32>(0);
  ec[c] = 1;
  let ft = face_texel(a, t);
  var s = textureLoad(BF, ft, 0)[a];
  var comp = textureLoad(BC, ft, 0)[a];
  let i0 = impulse(t);
  // In 2D the edges along z offsets are the same edge: those two terms cancel and are skipped.
  let flat = P.dims.z == 1u;
  var r: vec2<f32>;
  if (!(flat && b == 2)) {
    r = two_sum(s, -impulse(t + eb)[c]);
    s = r.x;
    comp += r.y;
    r = two_sum(s, i0[c]);
    s = r.x;
    comp += r.y;
  }
  if (!(flat && c == 2)) {
    // An axisymmetric run's B_z faces (code y) lie at the cell's radius, and the E_phi edges
    // beside them at the radii of its radial faces: d b_z / dt = (1/R) d(R E_phi) / dR, each
    // edge weighed by its radius over the face's. With these the divergence weighed likewise
    // (mhd/diagnostics.wgsl) is still unchanged by every edge.
    var wl = 1.0;
    var wh = 1.0;
    if (a == 1 && cylindrical()) {
      let w = cyl_weights(t.x);
      wl = w.x;
      wh = w.y;
    }
    r = two_sum(s, wh * impulse(t + ec)[b]);
    s = r.x;
    comp += r.y;
    r = two_sum(s, -wl * i0[b]);
    s = r.x;
    comp += r.y;
  }
  // Fold the compensation in: b is the pair rounded, c what that rounding left out.
  let hi = opaque(s + comp);
  return vec2<f32>(hi, comp - opaque(hi - s));
}

struct Fl { a: vec4<f32>, b: vec4<f32> }

fn flux_at(axis: i32, t: vec3<i32>) -> Fl {
  let i = 3u * flux_index(t);
  if (axis == 0) { return Fl(FX[i], FX[i + 1u]); }
  if (axis == 1) { return Fl(FY[i], FY[i + 1u]); }
  return Fl(FZ[i], FZ[i + 1u]);
}

// Shock flag of cell c at the start of the step, as hydro/predictor.wgsl sets it: the flow
// converges and the pressure (here the total, gas and magnetic) jumps by more than the
// threshold between neighbors along some axis.
fn shock_flag(c: vec3<i32>, axes: i32) -> f32 {
  var conv = 0.0;
  var jump = 0.0;
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let m = mq_load(c - e);
    let p = mq_load(c + e);
    conv += p.a[a + 1] - m.a[a + 1];
    let pm = m.b.x + 0.5 * dot(m.B, m.B);
    let pp = p.b.x + 0.5 * dot(p.B, p.B);
    jump = max(jump, abs(pp - pm) / min(pp, pm));
  }
  return select(0.0, 1.0, conv < 0.0 && jump > P.shock_thresh);
}

// What the stage makes of cell t (its texel), with its fluxes read at tf (its place in the flux
// buffers: t, or for a redo across a periodic z boundary, the copy in this slab) and bc its field
// at the end of the stage; with what the check of mhd/fofc.wgsl reads.
struct CellOut { st: NewState, rho_start: f32, rho_raw: f32, eint_raw: f32 }
var<private> g_redo = false;
const FOFC_HEAT = 4.0;
fn cell_step(t: vec3<i32>, tf: vec3<i32>, bc: vec3<f32>, axes: i32, dt: f32, dtdx: f32) -> CellOut {
  var out: CellOut;
  var net_a = vec4<f32>(0.0);   // (mass, momentum)
  var net_b = vec4<f32>(0.0);   // (energy, internal energy, scalar, face velocity)
  // An axisymmetric run: the radial faces weighed by their areas (angular momentum, R rho v_phi,
  // by the square: hydro/flux.wgsl cyl_face), and the radial momentum's geometric source,
  //   (p + B^2 / 2 + rho v_phi^2 - B_phi^2) / R,
  // with the total pressure the mean of the two radial faces' (their normal momentum flux less
  // what the mass carries, plus B_R^2: the flux pass leaves each face's B_R beside it), so that
  // uniform total pressure is in balance to rounding, as in hydro/update.wgsl.
  let cyl = cylindrical();
  let wc = select(vec2<f32>(1.0), cyl_weights(t.x), cyl);
  var src_r = 0.0;
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    if (a < axes) {
      var lo = flux_at(a, tf);
      var hi = flux_at(a, tf + e);
      if (cyl && a == 0) {
        let blo = FX[3u * flux_index(tf) + 2u].w;
        let bhi = FX[3u * flux_index(tf + e) + 2u].w;
        let pt = 0.5 * (lo.a.y - lo.a.x * lo.b.w + blo * blo + hi.a.y - hi.a.x * hi.b.w + bhi * bhi);
        let hs = select(textureLoad(H1, t, 0), textureLoad(U1, t, 0), STAGE == 1u);
        let bphi = FY[3u * flux_index(tf) + 2u].w;
        src_r = (pt + hs.w * hs.w / max(hs.x, P.rho_floor) - bphi * bphi) / (f32(t.x) + 0.5);
        lo = Fl(vec4<f32>(lo.a.xyz * wc.x, lo.a.w * wc.x * wc.x), lo.b * wc.x);
        hi = Fl(vec4<f32>(hi.a.xyz * wc.y, hi.a.w * wc.y * wc.y), hi.b * wc.y);
      }
      net_a += lo.a - hi.a;
      net_b += lo.b - hi.b;
    }
  }
  net_a.y += src_r;
  var u1 = textureLoad(U1, t, 0);
  var u2 = textureLoad(U2, t, 0);
  // A non-inertial frame's velocity-dependent forces (gravity/frame_off.wgsl): stage 1 the
  // half step's whole share, stage 2 the first half of the step's; gravity/correct.wgsl adds
  // the second after the second kick.
  if (FRAME_ON) {
    let fs = frame_open(u1, u2, select(0.5 * dt, dt, STAGE == 1u));
    u1 = fs.u1;
    u2 = fs.u2;
  }
  comp_use(comp_tex(t));
  // Pressure for p div v.
  var ph: f32;
  var flag = 0.0;
  if (STAGE == 1u) {
    ph = prim_from_cons(u1, u2).w2.x;
    flag = shock_flag(t, axes);
    comp_use(comp_tex(t));
  } else {
    ph = prim_from_cons(textureLoad(H1, t, 0), textureLoad(H2, t, 0)).w2.x;
  }
  // Gravity: the half step's whole source in stage 1, the first half kick in stage 2.
  var kick = vec3<f32>(0.0);
  var ekick = 0.0;
  var dw = 0.0;
  if (has_gravity() || ext_on()) {
    let w = select(0.5 * dt, dt, STAGE == 1u);
    let kx = w * u1.x * ext_accel_wb(cell_pos(t), P.dx);
    kick = w * u1.x * g_self(t) + kx;
    let flux_work = STAGE == 2u && has_gravity() && (P.flags & 1u) != 0u;
    if (flux_work) {
      // Each face hands half of F_mass dphi to the two cells it separates (hydro/update.wgsl).
      for (var a = 0; a < axes; a++) {
        var e = vec3<i32>(0);
        e[a] = 1;
        dw -= 0.5 * (flux_at(a, tf + e).a.x * dphi_half(t, e) - flux_at(a, tf).a.x * dphi_half(t, -e));
      }
    }
    let ke = select(kick, kx, flux_work);
    ekick = dot(ke, u1.yzw + 0.5 * kick) / max(u1.x, P.rho_floor);
  }
  // Turbulence driving (hydro/drive.wgsl): the stage's whole kick (dt is the stage's).
  if (drive_on()) {
    let kd = drive_kick(kick, ekick, cell_pos(t), dt, u1.x, u1.yzw);
    kick = kd.xyz;
    ekick = kd.w;
  }
  let eb = 0.5 * dot(bc, bc);
  let rho_new = max(u1.x + dtdx * net_a.x, P.rho_floor);
  let rho = alfven_floor(rho_new, 2.0 * eb);
  let mom = u1.yzw + kick + dtdx * net_a.yzw;
  // Mass the Alfven ceiling adds comes at rest: the kinetic energy it takes stays out of the heat.
  let E = u2.x + ekick + dtdx * (net_b.x + dw) - 0.5 * dot(mom, mom) * (1.0 / rho_new - 1.0 / rho);
  var heat = 0.0;
  if (VISCOSITY) { heat = visc_heating(t, axes); }
  if (RESISTIVITY) { heat += ohm_heat(t); }
  let eint = u2.y + dtdx * (net_b.y + max(ph, P.p_floor) * net_b.w + heat);
  out.rho_raw = u1.x + dtdx * net_a.x;
  out.rho_start = u1.x;
  out.eint_raw = eint;
  let sc = u2.z + dtdx * net_b.z;
  g_kphi = swirl_ke(rho, mom);
  // A redo (mhd/fofc.wgsl) keeps the advected internal energy, and E is rebuilt from it, where
  // E - KE - B^2 / 2 would make more than FOFC_HEAT times that energy: its faces carry first-order
  // energy fluxes, and the field second-order constrained transport, and in gas of low beta the
  // difference between the two in the field's energy is many times the heat itself (cells near a
  // sink in 1e-7 gas at 1e7 times their temperature). Elsewhere E stays as the fluxes made it.
  let ke = 0.5 * dot(mom, mom) / rho + eb;
  let heated = g_redo && eint > 0.0 && E - ke > FOFC_HEAT * eint;
  var en = dual_energy(rho, ke, select(E, ke, heated), eint);
  var mout = mom;
  let tame = tame_fluff(rho, mom, en.y);
  if (tame.changed) {
    mout = tame.mom;
    en = vec2<f32>(0.5 * dot(mout, mout) / rho + eb + tame.eint, tame.eint);
  }
  var st = NewState(vec4<f32>(rho, mout), vec4<f32>(en.x, en.y, sc, flag));
  // The damping zone and start-up damping over the whole step, on the gas alone: B^2 / 2 stays out of it.
  if (STAGE == 2u && damping_on()) {
    let d = apply_damping(NewState(st.u1, vec4<f32>(st.u2.x - eb, st.u2.yz, 0.0)), (vec3<f32>(t) + 0.5) / vec3<f32>(P.dims.xyz), clock.dt, clock.t);
    st = NewState(d.u1, vec4<f32>(d.u2.x + eb, d.u2.yz, flag));
  }
  // Cooling over the whole step (it changes E by exactly the change of e_int).
  if (STAGE == 2u) { st = apply_cooling(st, clock.dt); }
  out.st = st;
  return out;
}

// The cell has lost all but FOFC_KEEP of its density, or its internal energy (mhd/fofc.wgsl).
fn fofc_bad(c: CellOut) -> bool {
  if (!(c.rho_raw >= FOFC_KEEP * c.rho_start)) { return true; }
  return EOS_HAS_ENERGY && !(c.eint_raw > 0.0);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ed = emf_dims();
  if (any(vec2<i32>(gid.xy) >= ed.xy) || i32(gid.z) >= update_layers()) { return; }
  g_dx = P.dx;
  let dt = select(clock.dt, 0.5 * clock.dt, STAGE == 1u);
  g_dt = dt;
  let dtdx = dt / P.dx;
  let t = vec3<i32>(gid) + vec3<i32>(0, 0, emf_first());
  let n = mhd_dims();
  let axes = select(3, 2, P.dims.z == 1u);

  // Faces held by this texel: along a, those with an index along a up to n (up to n - 1 on a
  // periodic axis, whose upper face is the lower face of cell 0), and in the grid along the
  // other two.
  var bout = vec3<f32>(0.0);
  var cout = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    var ok = t[a] < n[a] || (t[a] == n[a] && !periodic_axis(a));
    for (var b = 0; b < 3; b++) {
      if (b != a && t[b] >= n[b]) { ok = false; }
    }
    if (ok) {
      let f = new_face(a, t);
      bout[a] = f.x;
      cout[a] = f.y;
    }
  }
  textureStore(OB, t, vec4<f32>(bout, 0.0));
  if (STAGE == 2u) { textureStore(OC, t, vec4<f32>(cout, 0.0)); }
  if (SPECIES_FLUX && STAGE == 2u) {
    // The mass flux through each face this texel's lower faces are, along the axes in use.
    let nm = vec3<u32>(P.dims.xyz) + 1u;
    for (var a = 0; a < axes; a++) {
      var ok = t[a] <= n[a];
      for (var b = 0; b < 3; b++) {
        if (b != a && t[b] >= n[b]) { ok = false; }
      }
      if (ok) {
        let u = vec3<u32>(t);
        MF[((u.z * nm.y + u.y) * nm.x + u.x) * 3u + u32(a)] = flux_at(a, t).a.x;
      }
    }
  }

  if (any(t >= n)) { return; }
  // The cell, with its field at the end of the stage: the mean of its two faces along each axis.
  var bc = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let bh = new_face(a, t + e).x;
    bc[a] = select(0.5 * (bout[a] + bh), cyl_bcell_r(t.x, bout[a], bh), cylindrical() && a == 0);
  }
  let cs = cell_step(t, t, bc, axes, dt, dtdx);
  textureStore(O1, t, cs.st.u1);
  textureStore(O2, t, cs.st.u2);
  // Stage 2: a cell the second-order fluxes have all but emptied (mhd/fofc.wgsl).
  if (STAGE == 2u && fofc_bad(cs)) { fofc_add(t, false); }
}

// The redo of mhd/fofc.wgsl: entry j / 7 of the list is a listed cell; k = j % 7 picks it (0) or a
// face neighbor (1 to 6), which is updated again from the fluxes as the fix left them if it is in
// the grid and this slab's turn: the slab's cells but its top layer (whose upper faces of the field
// the next slab writes), and the layer below them, the last slab's top layer. A cell in a later
// slab is redone there, from the list as it then stands.
@compute @workgroup_size(256)
fn redo_main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let cells = 7u * fofc_count();
  for (var j = wg.x * 256u + li; j < cells; j += FOFC_THREADS) {
    if (fofc_in_round(j / 7u, !FOFC_LIST_MORE)) { redo(j); }
  }
}

fn redo(j: u32) {
  g_redo = true;
  g_dx = P.dx;
  let dt = clock.dt;
  g_dt = dt;
  let dtdx = dt / P.dx;
  let n = mhd_dims();
  let axes = select(3, 2, P.dims.z == 1u);
  var t = fofc_listed(j / 7u);
  let k = i32(j % 7u);
  if (k > 0) {
    let a = (k - 1) / 2;
    if (a >= axes) { return; }
    t[a] += select(-1, 1, (k - 1) % 2 == 1);
    if (t[a] < 0 || t[a] >= n[a]) {
      if (!periodic_axis(a)) { return; }
      t[a] = (t[a] + n[a]) % n[a];
    }
  }
  // Its place in this slab's flux buffers (mhd/flux_index.wgsl): the layers from z0 - 2; across a
  // periodic z boundary, cell 0 of the grid is layer nz of the last slab.
  var tf = t;
  if (P.dims.z != 1u) {
    let z0 = SLAB.z0;
    let z1 = SLAB.z0 + SLAB.nz;
    let top = select(z1 - 1, z1, SLAB.last != 0u);
    if (t.z >= top) { return; }
    if (t.z < z0 - 2) {
      if (!periodic_axis(2) || t.z + n.z > z1) { return; }
      tf.z = t.z + n.z;
    }
  }
  var bc = vec3<f32>(0.0);
  for (var a = 0; a < 3; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    let lo = textureLoad(NF, face_texel(a, t), 0)[a];
    let hi = textureLoad(NF, face_texel(a, t + e), 0)[a];
    bc[a] = select(0.5 * (lo + hi), cyl_bcell_r(t.x, lo, hi), cylindrical() && a == 0);
  }
  let cs = cell_step(t, tf, bc, axes, dt, dtdx);
  textureStore(O1, t, cs.st.u1);
  textureStore(O2, t, cs.st.u2);
  // A neighbor this has all but emptied: listed for the next round (none after the last).
  if (FOFC_LIST_MORE && k > 0 && fofc_bad(cs)) { fofc_add(t, true); }
  if (SPECIES_FLUX) {
    // The mass fluxes of the cell's lower faces, and of its upper ones on the grid's boundary.
    let nm = vec3<u32>(P.dims.xyz) + 1u;
    for (var a = 0; a < axes; a++) {
      var e = vec3<i32>(0);
      e[a] = 1;
      let u = vec3<u32>(t);
      MF[((u.z * nm.y + u.y) * nm.x + u.x) * 3u + u32(a)] = flux_at(a, tf).a.x;
      if (t[a] == n[a] - 1 && !periodic_axis(a)) {
        let v = vec3<u32>(t + e);
        MF[((v.z * nm.y + v.y) * nm.x + v.x) * 3u + u32(a)] = flux_at(a, tf + e).a.x;
      }
    }
  }
}

// The turbulence closure's network (closure/closure.ts; trained in tools/closure): the features
// of a cell's resolved flow, the gates that switch the closure off at shocks and in flow that
// is not sheared, and the small network between them. Two-dimensional: the cell and its four
// neighbors. The float64 twin is closureCell() in closure/closure.ts, and the one the network
// was trained through is tools/closure/closure.py.
//
// The <closure_net> slot holds the weights W and the constants that were exported with them (CL_IN
// features, CL_H units in each of CL_LAYERS hidden layers, the gates' thresholds, the outputs'
// limits; CL_K0 and CL_KH, the vec4s a row of the first and of the later layers takes).
#include <closure_net>


// What the closure reads of a cell.
struct ClCell { rho: f32, v: vec2<f32>, p: f32, s: f32 }

fn cl_smoothstep(e: vec2<f32>, x: f32) -> f32 {
  let t = clamp((x - e.x) / (e.y - e.x), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// How far from monotone and straight a quantity is along one axis, from its differences below
// and above the cell: 0 for a straight line, 1 at an extremum or the foot of a step.
fn cl_kink(dm: f32, dp: f32, floor: f32) -> f32 {
  return abs(dp - dm) / (abs(dp) + abs(dm) + floor);
}
fn cl_kink2(dm: vec2<f32>, dp: vec2<f32>, floor: f32) -> f32 {
  return sqrt(dot(dp - dm, dp - dm) + 1.0e-30) / (sqrt(dot(dp, dp) + 1.0e-30) + sqrt(dot(dm, dm) + 1.0e-30) + floor);
}

struct ClFeatures { f: array<f32, CL_IN>, gate: f32 }

// The features and the gate of cell c, whose sound speed is cs, from its neighbors below and
// above along x and y.
fn cl_features(c: ClCell, xm: ClCell, xp: ClCell, ym: ClCell, yp: ClCell, cs: f32) -> ClFeatures {
  // dx times the velocity's gradient: its divergence D, rotation W and strain A (the
  // deviatoric strain's magnitude, with the trace taken out as in three dimensions, so that
  // any one-dimensional compression has D^2 / (D^2 + A^2) = 3/7).
  let gx = 0.5 * (xp.v - xm.v);
  let gy = 0.5 * (yp.v - ym.v);
  let D = gx.x + gy.y;
  let Wz = gx.y - gy.x;
  let sxx = gx.x - D / 3.0;
  let syy = gy.y - D / 3.0;
  let szz = -D / 3.0;
  let sxy = 0.5 * (gy.x + gx.y);
  let A2 = 2.0 * (sxx * sxx + syy * syy + szz * szz + 2.0 * sxy * sxy);
  let A = sqrt(A2 + 1.0e-30);
  let e2 = (CL_EPS_V * cs) * (CL_EPS_V * cs);
  let tot = D * D + Wz * Wz + A2 + e2;
  let kv = max(cl_kink2(c.v - xm.v, xp.v - c.v, 1.0e-4 * cs), cl_kink2(c.v - ym.v, yp.v - c.v, 1.0e-4 * cs));
  let kr = max(cl_kink(c.rho - xm.rho, xp.rho - c.rho, 1.0e-5 * c.rho), cl_kink(c.rho - ym.rho, yp.rho - c.rho, 1.0e-5 * c.rho));
  // (The scalar has no scale of its own: differences below 1e-4 of it, or below 1e-6 for a dye of
  // order one, are rounding, and count as none.)
  let sf = 1.0e-4 * (abs(c.s) + abs(xp.s) + abs(xm.s)) + 1.0e-6;
  let ks = max(cl_kink(c.s - xm.s, xp.s - c.s, sf), cl_kink(c.s - ym.s, yp.s - c.s, sf));
  let speed = sqrt(dot(c.v, c.v) + 1.0e-30);
  let M = speed / cs;
  let drho = 0.5 * vec2<f32>(xp.rho - xm.rho, yp.rho - ym.rho);
  let dp = 0.5 * vec2<f32>(xp.p - xm.p, yp.p - ym.p);
  let grho = sqrt(dot(drho, drho) + 1.0e-30) / c.rho;
  let gp = sqrt(dot(dp, dp) + 1.0e-30) / c.p;
  var o: ClFeatures;
  o.f[0] = 0.5 * log(A / cs + 1.0e-4) * 0.43429448 + 1.0;
  o.f[1] = (Wz * Wz - A2) / (Wz * Wz + A2 + e2);
  o.f[2] = D / sqrt(tot);
  o.f[3] = kv;
  o.f[4] = M / (1.0 + M);
  o.f[5] = grho / (grho + 0.05);
  o.f[6] = kr;
  o.f[7] = ks;
  o.f[8] = gp / (gp + 0.05);
  o.f[9] = A / (A + speed + 1.0e-4 * cs);
  // Gates: closed in compression (a shock, or any flow in one dimension), across a pressure
  // jump, and where the gas is not sheared or turning.
  let jump = max(abs(xp.p - xm.p) / min(xp.p, xm.p), abs(yp.p - ym.p) / min(yp.p, ym.p));
  let shear = sqrt(Wz * Wz + A2 + 1.0e-30) / cs;
  o.gate = (1.0 - cl_smoothstep(CL_DUCROS, D * D / tot)) * (1.0 - cl_smoothstep(CL_JUMP, jump)) * cl_smoothstep(CL_ACTIVE, shear);
  return o;
}

// The network on features f: its four outputs, each through a sigmoid. The weights W are
// constants of the shader (the <closure_net> slot), packed four to a vec4 (closure/closure.ts
// packedWeights): each row of a layer's matrix as vec4s, padded with zeros, then the layer's
// biases as vec4s, so that a unit is a few dot products. (Read one by one from a buffer, the
// pass took a third longer; the network is nearly all of its cost.)
fn cl_net(f: array<f32, CL_IN>) -> vec4<f32> {
  var fin: array<vec4<f32>, CL_K0>;
  for (var i = 0u; i < CL_IN; i++) { fin[i / 4u][i % 4u] = f[i]; }
  var x: array<vec4<f32>, CL_KH>;
  var y: array<vec4<f32>, CL_KH>;
  var o = 0u;
  for (var g = 0u; g < CL_KH; g++) {
    var acc = W[o + CL_H * CL_K0 + g];
    for (var c = 0u; c < 4u; c++) {
      let row = o + (4u * g + c) * CL_K0;
      var a = 0.0;
      for (var k = 0u; k < CL_K0; k++) { a += dot(W[row + k], fin[k]); }
      acc[c] += a;
    }
    x[g] = tanh(acc);
  }
  o += CL_H * CL_K0 + CL_KH;
  for (var l = 1u; l < CL_LAYERS; l++) {
    for (var g = 0u; g < CL_KH; g++) {
      var acc = W[o + CL_H * CL_KH + g];
      for (var c = 0u; c < 4u; c++) {
        let row = o + (4u * g + c) * CL_KH;
        var a = 0.0;
        for (var k = 0u; k < CL_KH; k++) { a += dot(W[row + k], x[k]); }
        acc[c] += a;
      }
      y[g] = tanh(acc);
    }
    x = y;
    o += CL_H * CL_KH + CL_KH;
  }
  var out = W[o + 4u * CL_KH];
  for (var c = 0u; c < 4u; c++) {
    var a = 0.0;
    for (var k = 0u; k < CL_KH; k++) { a += dot(W[o + c * CL_KH + k], x[k]); }
    out[c] += a;
  }
  return 1.0 / (1.0 + exp(-out));
}

// A cell's coefficients (a, b): the shares of the velocity's and the passive scalar's jumps
// that the Riemann solver is not shown on the cell's faces (hydro/closure_on.wgsl), and the
// gate in z.
fn cl_cell(c: ClCell, xm: ClCell, xp: ClCell, ym: ClCell, yp: ClCell, cs: f32) -> vec4<f32> {
  let ft = cl_features(c, xm, xp, ym, yp, cs);
  // (Where the gate is shut the network is not run: most of a shock tube.)
  if (ft.gate <= 0.0) { return vec4<f32>(0.0); }
  let o = cl_net(ft.f);
  return vec4<f32>(CL_A_MAX * o.x * ft.gate * CL_HEADS.x, CL_B_MAX * o.y * ft.gate * CL_HEADS.y, ft.gate, 0.0);
}

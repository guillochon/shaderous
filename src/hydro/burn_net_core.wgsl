// The trained network of the 13-isotope alpha chain (burn/network.ts): its inputs (the
// <burn_net> slot, generated with every rate fit written in), the screening they use, the
// network itself (weights W, laid out as NET_OFFSETS says) and one step of it, net_step().
// Shared by hydro/burn_net.wgsl and burn/net_test.wgsl (the validation's check against the
// float64 mirror).
#include <burn_net>

@group(0) @binding(10) var<storage, read> W: array<f32>;

#include "hydro/burn_screen.wgsl"

// ---- Screening of the 13 species ----

fn screen_common(T: f32, rho: f32, Y: array<f32, 13>) -> ScreenCommon {
  var ytot = 0.0;
  var zy = 0.0;
  var z2y = 0.0;
  for (var i = 0u; i < 13u; i++) {
    let y = max(Y[i], 1e-30);
    ytot += y;
    zy += NET_Z[i] * y;
    z2y += NET_Z[i] * NET_Z[i] * y;
  }
  return screen_moments(T, rho, ytot, zy / ytot, z2y / ytot);
}

// ---- The network ----

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

// One hidden layer from NET_HIDDEN inputs to NET_HIDDEN outputs, weights from o.
fn dense_hidden(o: u32, x: ptr<function, array<f32, NET_HIDDEN>>, y: ptr<function, array<f32, NET_HIDDEN>>) {
  let bias = o + NET_HIDDEN * NET_HIDDEN;
  for (var j = 0u; j < NET_HIDDEN; j++) {
    var s = W[bias + j];
    let r = o + j * NET_HIDDEN;
    for (var i = 0u; i < NET_HIDDEN; i++) { s += W[r + i] * (*x)[i]; }
    (*y)[j] = silu(s);
  }
}

fn net_eval(f: array<f32, NET_IN>) -> array<f32, NET_OUT> {
  var a: array<f32, NET_HIDDEN>;
  var b: array<f32, NET_HIDDEN>;
  // First layer, from the inputs.
  let o0 = NET_OFFSETS[0];
  for (var j = 0u; j < NET_HIDDEN; j++) {
    var s = W[o0 + NET_IN * NET_HIDDEN + j];
    let r = o0 + j * NET_IN;
    for (var i = 0u; i < NET_IN; i++) { s += W[r + i] * f[i]; }
    a[j] = silu(s);
  }
  // The other hidden layers, back and forth between a and b.
  for (var l = 1u; l < NET_DEPTH; l++) {
    if ((l & 1u) == 1u) { dense_hidden(NET_OFFSETS[l], &a, &b); } else { dense_hidden(NET_OFFSETS[l], &b, &a); }
  }
  if ((NET_DEPTH & 1u) == 0u) { a = b; }
  // Outputs, linear.
  var z: array<f32, NET_OUT>;
  let ol = NET_OFFSETS[NET_DEPTH];
  for (var j = 0u; j < NET_OUT; j++) {
    var s = W[ol + NET_HIDDEN * NET_OUT + j];
    let r = ol + j * NET_HIDDEN;
    for (var i = 0u; i < NET_HIDDEN; i++) { s += W[r + i] * a[i]; }
    z[j] = s;
  }
  return z;
}

// One stage's change in composition from its 27 outputs starting at o: each species loses a
// share of itself, sigmoid(u_i), and what is lost goes to the species by a softmax, all of it
// times 1 - exp(-k dt).
fn flow_stage(z: array<f32, NET_OUT>, o: u32, X: array<f32, 13>, log_dt: f32) -> array<f32, 13> {
  var loss: array<f32, 13>;
  var g: array<f32, 13>;
  var M = 0.0;
  var gm = -1e30;
  for (var i = 0u; i < 13u; i++) {
    let u = 40.0 * tanh(z[o + 1u + i] / 40.0);
    loss[i] = X[i] / (1.0 + exp(-u));
    M += loss[i];
    g[i] = 40.0 * tanh(z[o + 14u + i] / 40.0);
    gm = max(gm, g[i]);
  }
  var es = 0.0;
  for (var i = 0u; i < 13u; i++) {
    g[i] = exp(g[i] - gm);
    es += g[i];
  }
  let kdt = exp(LN10 * min(5.0 + 5.0 * z[o] + log_dt, 6.0));
  let c = select(1.0 - exp(-kdt), kdt * (1.0 - 0.5 * kdt), kdt < 1e-3);
  var D: array<f32, 13>;
  for (var i = 0u; i < 13u; i++) { D[i] = c * (M * g[i] / es - loss[i]); }
  return D;
}

// One step of the network: the change in mass fractions over 10^log_dt seconds.
fn net_step(log_rho: f32, log_T: f32, log_dt: f32, X: array<f32, 13>) -> array<f32, 13> {
  let z = net_eval(net_features(log_rho, log_T, log_dt, X));
  let D1 = flow_stage(z, 0u, X, log_dt);
  var X1: array<f32, 13>;
  for (var i = 0u; i < 13u; i++) { X1[i] = max(X[i] + D1[i], 0.0); }
  let D2 = flow_stage(z, 27u, X1, log_dt);
  var D: array<f32, 13>;
  for (var i = 0u; i < 13u; i++) { D[i] = D1[i] + D2[i]; }
  return D;
}


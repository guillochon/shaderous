// Time-centering of the implicit step (docs/numerics.md §9). The step's pressure force is that
// of p_theta = theta p' + (1 - theta) p^n, and the divergence its solve sees that of
// theta v' + (1 - theta) v^n: backward Euler at theta = 1, Crank-Nicolson at 1/2, one solve
// either way. Each cell takes
//   theta = max(chi, min(1, 1/2 + K C)),   C = c_s dt / dx,
// its own acoustic Courant number: sound the step resolves is carried without backward Euler's
// damping (a wave resolved at C = 0.5 lost 14% of its amplitude a period, 1.6% with this), and
// sound it does not resolve (C >= 5, every low Mach number flow) is damped as before.
// (Crank-Nicolson itself, theta = 1/2 everywhere, leaves grid-scale sound ringing undamped.)
// chi is the cell's compression switch: 1/2 + MU dt max(0, -div v), from the advected face
// velocities. A shock takes backward Euler (with theta near 1/2 the Sod tube's post-shock
// velocity overshot by 40% and rang behind the shock); smooth flow is all but untouched.
const PS_THETA_K = 0.1;
const PS_MU = 4.0;

fn ps_theta(cs: f32, dt: f32, dx: f32) -> f32 {
  return clamp(0.5 + PS_THETA_K * cs * dt / dx, 0.5, 1.0);
}

// A gas: the face velocity that carries mass, momentum and energy through the step, and that the
// pressure and the fields do their work on, is v_c = v^n + chi_f (v' - v^n), chi_f the larger of
// its two cells' switches: the mean of the two in smooth flow, whatever theta. (The corrected
// velocity alone, backward Euler's choice, made the step first order even where the flow is all
// but incompressible.) With chi_f = 1/2 the kinetic energy a face's kick and pressure difference
// give its gas is the work they do on it.
fn ps_chi(dt: f32, div: f32) -> f32 {
  return clamp(0.5 + PS_MU * dt * max(0.0, -div), 0.5, 1.0);
}

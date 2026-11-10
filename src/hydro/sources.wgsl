// Sources and sinks of gas: soft-edged balls (discs in 2D) that add gas, or take it away, at a
// steady rate for as long as they exist. Applied to a cell's state at the end of its update.
// Requires P, the equation of state, hydro/dual_energy.wgsl and hydro/advance.wgsl (NewState).
// Layout matches packSources() in src/sim.ts.
struct Source {
  pos: vec4<f32>,    // centre in fractions of the box, radius (fraction of the box) in w
  vel: vec4<f32>,    // velocity of the gas a source adds (zero: it arrives at rest and spreads
                     //   evenly); softness of the edge in w
  rate: vec4<f32>,   // x: density added per unit time at full weight (a source)
                     // y: fraction of the gas removed per unit time at full weight (a sink)
}
struct Sources { count: u32, list: array<Source, 8> }
@group(0) @binding(11) var<uniform> SRC: Sources;

fn source_weight(x: vec3<f32>, s: Source) -> f32 {
  var d = x - s.pos.xyz;
  // Across a periodic face the ball wraps around with the gas.
  if (P.bc_lo.x == 0u) { d.x -= round(d.x); }
  if (P.bc_lo.y == 0u) { d.y -= round(d.y); }
  if (P.dims.z == 1u) { d.z = 0.0; } else if (P.bc_lo.z == 0u) { d.z -= round(d.z); }
  let r = length(d) / s.pos.w;
  if (r >= 1.0) { return 0.0; }
  return 1.0 - smoothstep(min(1.0 - s.vel.w, 0.999), 1.0, r);
}

// The state of the cell at `x` (in fractions of the box) after a step of length dt.
fn apply_sources(st: NewState, x: vec3<f32>, dt: f32) -> NewState {
  var rho = st.u1.x;
  var mom = st.u1.yzw;
  var eint = st.u2.y;
  var sc = st.u2.z;
  var hit = false;
  for (var i = 0u; i < SRC.count; i++) {
    let s = SRC.list[i];
    let w = source_weight(x, s);
    if (w <= 0.0) { continue; }
    hit = true;
    if (s.rate.y > 0.0) {
      // Sink: the gas that goes takes its momentum and energy with it.
      let rn = max(rho * exp(-s.rate.y * w * dt), P.rho_floor);
      let f = rn / rho;
      mom *= f;
      eint *= f;
      sc *= f;
      rho = rn;
    } else {
      // Source: new gas with the specific internal energy and scalar of what is there, moving
      // at the source's velocity.
      let dr = s.rate.x * w * dt;
      let f = (rho + dr) / rho;
      eint *= f;
      sc *= f;
      mom += dr * s.vel.xyz;
      rho += dr;
    }
  }
  if (!hit) { return st; }
  let ke = 0.5 * dot(mom, mom) / rho;
  var E = ke + eint;
  if (!EOS_HAS_ENERGY) {
    let en = dual_energy(rho, ke, E, eint);
    E = en.x;
    eint = en.y;
  }
  return NewState(vec4<f32>(rho, mom), vec4<f32>(E, eint, sc, st.u2.w));
}

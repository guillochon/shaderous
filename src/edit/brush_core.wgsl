// Editing brush: a soft-edged ball (a disc in 2D) that adds or removes gas, sets the velocity
// or scales the temperature of the cells inside it. Shared by the uniform and block grids.
// Requires P, the equation of state and hydro/dual_energy.wgsl. Layout matches packBrush() in
// src/sim.ts.
struct Brush {
  pos: vec4<f32>,   // centre in fractions of the box, radius (fraction of the box) in w
  vel: vec4<f32>,   // velocity tool: the velocity to set; softness in w (0 hard edge, 1 soft to the centre)
  kind: u32,        // 0 mass, 1 velocity, 2 temperature
  amount: f32,      // mass: density added at full weight (negative removes);
                    // temperature: log of the factor applied at full weight
}
@group(0) @binding(1) var<uniform> B: Brush;

// Weight of the brush at a point given in fractions of the box: 1 inside the hard core,
// falling smoothly to 0 at the radius.
fn brush_weight(x: vec3<f32>) -> f32 {
  var d = x - B.pos.xyz;
  // Across a periodic face the brush wraps around with the gas.
  if (P.bc_lo.x == 0u) { d.x -= round(d.x); }
  if (P.bc_lo.y == 0u) { d.y -= round(d.y); }
  if (P.dims.z == 1u) { d.z = 0.0; } else if (P.bc_lo.z == 0u) { d.z -= round(d.z); }
  let r = length(d) / B.pos.w;
  if (r >= 1.0) { return 0.0; }
  return 1.0 - smoothstep(min(1.0 - B.vel.w, 0.999), 1.0, r);
}

struct Cons { u1: vec4<f32>, u2: vec4<f32> }

fn brush_apply(u1: vec4<f32>, u2: vec4<f32>, w: f32) -> Cons {
  if (w <= 0.0) { return Cons(u1, u2); }
  let rho = max(u1.x, P.rho_floor);
  var rn = rho;
  var mom = u1.yzw;
  var eint = u2.y;
  var s = u2.z;
  if (B.kind == 0u) {
    // New gas arrives with the velocity, specific internal energy and scalar of what is there.
    rn = max(rho + B.amount * w, P.rho_floor);
    let f = rn / rho;
    mom *= f;
    eint *= f;
    s *= f;
  } else if (B.kind == 1u) {
    mom = rho * mix(mom / rho, B.vel.xyz, w);
  } else if (EOS_HAS_ENERGY) {
    eint = eos_eint_from_T(rho, eos_temperature(rho, eint) * exp(B.amount * w));
  }
  let ke = 0.5 * dot(mom, mom) / rn;
  var E = ke + eint;
  if (!EOS_HAS_ENERGY) {
    // Barotropic: the internal energy follows the density.
    let en = dual_energy(rn, ke, E, eint);
    E = en.x;
    eint = en.y;
  }
  return Cons(vec4<f32>(rn, mom), vec4<f32>(E, eint, s, u2.w));
}

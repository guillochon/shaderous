// Editing brush: a soft-edged ball (a disc in 2D) that adds or removes gas, sets the velocity
// or scales the temperature of the cells inside it. Shared by the uniform and block grids.
// Requires P, the equation of state and hydro/dual_energy.wgsl. Layout matches packBrush() in
// src/sim.ts.
struct Brush {
  pos: vec4<f32>,   // centre in fractions of the box, radius (fraction of the box) in w
  vel: vec4<f32>,   // velocity tool: the velocity to set; softness in w (0 hard edge, 1 soft to the centre)
  kind: u32,        // 0 mass, 1 velocity, 2 temperature, 3 obstacle
  amount: f32,      // mass: density added at full weight (negative removes);
                    // temperature: log of the factor applied at full weight;
                    // obstacle: positive makes cells solid, negative turns them back into gas
  at_rest: u32,     // mass: 1 if new gas arrives with no momentum, 0 if with the local velocity
  nvert: u32,       // obstacle: vertices of the outline, or 0 for an ellipse
  // Obstacle: the outline is drawn on the screen and cut through the box along the line of
  // sight. `m` takes a point of the box (in fractions) to homogeneous screen coordinates.
  m: mat4x4<f32>,
  shape: vec4<f32>,                // ellipse: centre and half-axes
  verts: array<vec4<f32>, 16>,     // polygon: two vertices per element
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

fn brush_vert(i: u32) -> vec2<f32> {
  let v = B.verts[i / 2u];
  return select(v.xy, v.zw, (i & 1u) == 1u);
}

// Whether a point of the box lies behind the obstacle's outline.
fn in_outline(x: vec3<f32>) -> bool {
  let q = B.m * vec4<f32>(x, 1.0);
  if (q.w <= 0.0) { return false; }
  let s = q.xy / q.w;
  if (B.nvert == 0u) {
    let d = (s - B.shape.xy) / B.shape.zw;
    return dot(d, d) <= 1.0;
  }
  // Even-odd rule: count the edges a ray towards +x crosses.
  var inside = false;
  var j = B.nvert - 1u;
  for (var i = 0u; i < B.nvert; i++) {
    let a = brush_vert(i);
    let b = brush_vert(j);
    if ((a.y > s.y) != (b.y > s.y) && s.x < (b.x - a.x) * (s.y - a.y) / (b.y - a.y) + a.x) { inside = !inside; }
    j = i;
  }
  return inside;
}

struct Cons { u1: vec4<f32>, u2: vec4<f32> }

// The state of a cell at `x` (in fractions of the box) after the brush.
fn brush_apply(u1: vec4<f32>, u2: vec4<f32>, x: vec3<f32>) -> Cons {
  if (B.kind == 3u) {
    if (!in_outline(x)) { return Cons(u1, u2); }
    if (B.amount > 0.0) {
      // The gas in a new obstacle is kept, at rest, for when the obstacle is removed.
      return Cons(vec4<f32>(u1.x, 0.0, 0.0, 0.0), vec4<f32>(u2.y, u2.y, u2.z, 1.0));
    }
    return Cons(u1, vec4<f32>(u2.xyz, select(u2.w, 0.0, u2.w > 0.5)));
  }
  let w = brush_weight(x);
  // The other tools work on gas only.
  if (w <= 0.0 || u2.w > 0.5) { return Cons(u1, u2); }
  let rho = max(u1.x, P.rho_floor);
  var rn = rho;
  var mom = u1.yzw;
  var eint = u2.y;
  var s = u2.z;
  if (B.kind == 0u) {
    // New gas arrives with the specific internal energy and scalar of what is there, and
    // either with the local velocity or at rest: then the momentum is unchanged and is shared
    // with the new gas, which slows the flow. Gas that is removed takes its own momentum with it.
    rn = max(rho + B.amount * w, P.rho_floor);
    let f = rn / rho;
    mom *= select(f, min(f, 1.0), B.at_rest != 0u);
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

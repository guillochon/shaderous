// The edge field of constrained transport from the four faces around an edge and the four
// cells around it, shared by both grids (mhd/emf.wgsl, mhd/amr_emf.wgsl). Requires fc(a, t), the
// (F_a(B_t1), F_a(B_t2), upwind weight) of the lower face along a of cell t.

// What the edges need of a cell: its reference field -v x B, its velocity and sound speed.
struct Ref { e: vec3<f32>, v: vec3<f32>, cs: f32 }

// Edge fields (mhd/mhd.ts, EdgeScheme): 0 the mean of Gardiner & Stone's four estimates, 1 the
// upwind ones among them (Lee 2013's selection), 2 the mean of Lee & Deane's (2009) estimates,
// 3 the upwind ones among those (Lee 2013's upwind-MEC). Lee & Deane carry each face's field to
// the edge by a Taylor series, E + (D/2) E' + (D^2/8) E'', with central differences of the face
// values along their line (their eqs. 54-59): the parabola through three faces in a row,
// (6 E_j + 3 E_toward - E_away) / 8. It needs the faces two along each line, which the uniform
// grid's flux buffers hold (mhd/flux_index.wgsl) and the block grid's do not: modes 2 and 3 are
// the uniform grid's only.
override EDGE_MODE: u32 = 0u;

// E along axis c at the edge on the lower a and b faces of cell t (a = c + 1, b = c + 2),
// given the four cells around it: r00 at t - ea - eb, r10 at t - eb, r01 at t - ea, r11 at t.
//
// Each of the four faces around the edge gives an estimate of the edge's field: its own value
// carried half a cell along the face to the edge, with the gradient taken between the face and
// the reference field of the cell on that side, from whichever cell is upwind across the face
// (the mass flux's direction). Gardiner & Stone take the mean of the four. Lee (2013) points
// out that for flow mostly along one axis the faces downstream of the edge carry no information
// about it, and the mean then lets oscillations grow (the field loop at small angles); his
// upwind-MEC uses only the estimates from the faces upwind of the edge: for flow toward +b, the
// a face below the edge, toward +a, the b face to its left, both (weighted a half each) when
// the flow has both components. A component under a tenth of the in-plane speed counts as none,
// so noise does not flip the choice, and where the flow is slower than 1e-4 of the sound speed
// the mean is used. (Lee also adds second-derivative terms to each estimate, from Lee & Deane
// 2009; here the estimates are Gardiner & Stone's.)
fn corner(c: i32, t: vec3<i32>, r00: Ref, r10: Ref, r01: Ref, r11: Ref) -> f32 {
  let a = (c + 1) % 3;
  let b = (c + 2) % 3;
  var ea = vec3<i32>(0);
  ea[a] = 1;
  var eb = vec3<i32>(0);
  eb[b] = 1;
  let e00 = r00.e[c];
  let e10 = r10.e[c];
  let e01 = r01.e[c];
  let e11 = r11.e[c];
  // Faces normal to a, at b positions j (t) and j - 1 (t - eb): E_c = -F_a(B_b).
  let fa1 = fc(a, t);
  let fa0 = fc(a, t - eb);
  // Faces normal to b, at a positions i (t) and i - 1 (t - ea): E_c = F_b(B_a).
  let fb1 = fc(b, t);
  let fb0 = fc(b, t - ea);
  let ea1 = -fa1.x;
  let ea0 = -fa0.x;
  let eb1 = fb1.y;
  let eb0 = fb0.y;
  // The four estimates: from the a face below the edge (along b), the a face above it, the b
  // face left of it (along a) and the b face right of it.
  let below = ea0 + ((1.0 - fa0.z) * (eb1 - e10) + fa0.z * (eb0 - e00));
  let above = ea1 + ((1.0 - fa1.z) * (eb1 - e11) + fa1.z * (eb0 - e01));
  let left = eb0 + ((1.0 - fb0.z) * (ea1 - e01) + fb0.z * (ea0 - e00));
  let right = eb1 + ((1.0 - fb1.z) * (ea1 - e11) + fb1.z * (ea0 - e10));
  var below_e = below;
  var above_e = above;
  var left_e = left;
  var right_e = right;
  if (EDGE_MODE >= 2u) {
    // Lee & Deane: the next faces out along each line.
    let ea2 = -fc(a, t - 2 * eb).x;   // a face below the lower one
    let eap = -fc(a, t + eb).x;       // above the upper one
    let eb2 = fc(b, t - 2 * ea).y;    // b face left of the left one
    let ebp = fc(b, t + ea).y;        // right of the right one
    below_e = (6.0 * ea0 + 3.0 * ea1 - ea2) * 0.125;
    above_e = (6.0 * ea1 + 3.0 * ea0 - eap) * 0.125;
    left_e = (6.0 * eb0 + 3.0 * eb1 - eb2) * 0.125;
    right_e = (6.0 * eb1 + 3.0 * eb0 - ebp) * 0.125;
  }
  let mean = 0.25 * ((left_e + right_e) + (below_e + above_e));
  if (EDGE_MODE == 0u || EDGE_MODE == 2u) { return mean; }
  let v = 0.25 * ((r00.v + r11.v) + (r10.v + r01.v));
  let cs = 0.25 * ((r00.cs + r11.cs) + (r10.cs + r01.cs));
  var u = v[a];
  var w = v[b];
  let speed = sqrt(u * u + w * w);
  if (!(speed > 1.0e-4 * cs)) { return mean; }
  if (abs(u) <= 0.1 * speed) { u = 0.0; }
  if (abs(w) <= 0.1 * speed) { w = 0.0; }
  let along_a = select(right_e, left_e, u > 0.0);
  let along_b = select(above_e, below_e, w > 0.0);
  if (u == 0.0) { return along_b; }
  if (w == 0.0) { return along_a; }
  return 0.5 * (along_a + along_b);
}


// Where the boundary value of cell c across its face o (a unit step out of the grid along one
// axis) is kept in gravity/boundary.wgsl's BF: face 2 a + (1 for the upper side), then the
// cell's place along axes a + 1 and a + 2.
fn bf_index(c: vec3<i32>, o: vec3<i32>, n: i32) -> u32 {
  var a = 0;
  if (o.y != 0) { a = 1; }
  if (o.z != 0) { a = 2; }
  let f = 2 * a + select(0, 1, o[a] > 0);
  return u32((f * n + c[(a + 1) % 3]) * n + c[(a + 2) % 3]);
}

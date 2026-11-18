// The operator on a level. Requires PRS (a storage texture with read access) and COEF.
// op_sum returns (sum, diag) such that  h^2 (A p)_c = diag p_c - sum.
fn op_sum(c: vec3<i32>) -> vec2<f32> {
  let d = vec3<i32>(L.dims);
  let k = textureLoad(COEF, c, 0);
  var sum = 0.0;
  var diag = k.w * L.h * L.h;
  let axes = plvl_axes();
  for (var a = 0; a < axes; a++) {
    var e = vec3<i32>(0);
    e[a] = 1;
    // The upper face: this cell's own coefficient.
    let bu = k[a];
    if (c[a] + 1 < d[a]) {
      sum += bu * textureLoad(PRS, c + e).x;
      diag += bu;
    } else if (L.hi[a] == PF_PERIODIC) {
      var q = c;
      q[a] = 0;
      sum += bu * textureLoad(PRS, q).x;
      diag += bu;
    } else if (L.hi[a] == PF_OPEN) {
      diag += 2.0 * bu;
    }
    // The lower face: the coefficient of the cell below.
    if (c[a] > 0) {
      let bl = textureLoad(COEF, c - e, 0)[a];
      sum += bl * textureLoad(PRS, c - e).x;
      diag += bl;
    } else if (L.lo[a] == PF_PERIODIC) {
      var q = c;
      q[a] = d[a] - 1;
      let bl = textureLoad(COEF, q, 0)[a];
      sum += bl * textureLoad(PRS, q).x;
      diag += bl;
    } else if (L.lo[a] == PF_OPEN) {
      // (No cell below holds this face's coefficient: the cell's own stands for it.)
      diag += 2.0 * select(bu, 1.0, bu == 0.0);
    }
  }
  return vec2<f32>(sum, diag);
}

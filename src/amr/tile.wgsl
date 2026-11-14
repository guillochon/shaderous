// Face-flux tile for the block-grid update (amr/update.wgsl), as hydro/tile.wgsl is for the
// uniform grid: the workgroup computes each face of its cells once, with flux_between, into
// workgroup memory, and each cell reads its six. WGSL lets a compiler fuse and reassociate
// floating-point operations (WGSL 15.7.5), so the gather's two call sites for a face (a cell's
// lower face and its neighbor's upper face) need not agree to the bit; here every face two
// same-level cells share is computed by the same code from the same values. Within a block a
// face is computed once; on a block's face, once by each of the two blocks, from its own copy
// of the other's cells in its ghosts, which hold the same stored values.
//
// A workgroup is 8 x 8 x 4 invocations (amr/list.wgsl): in 3D a box of cells of one block, in
// 2D four boxes of 8 x 8 cells, one in each of four blocks of the list. Faces of a block that
// is only depositing this pass, or beyond the end of the list, are skipped.
//
// Faces on the domain's edge have one cell each and nothing to agree with. Their potential
// steps (well-balanced reconstruction) are taken as axis_fluxes takes them, from dphi_blk,
// which knows the boundary's rows: dphi_face and wall_step in amr/update.wgsl look at
// g_face_wall.
// Requires amr/grid.wgsl, amr/list.wgsl, hydro/flux.wgsl, and the bins slot.

// Set by amr_sim.ts when the tile fits in the device's workgroup memory.
override TILE: bool = false;
override TILE_2D: bool = false;

// Faces of one box along each axis, then of all the workgroup's boxes.
override TF_NX: u32 = select(9u * 8u * 4u, 9u * 8u, TILE_2D);
override TF_NY: u32 = select(8u * 9u * 4u, 8u * 9u, TILE_2D);
override TF_NZ: u32 = select(8u * 8u * 5u, 0u, TILE_2D);
override TF_BOX: u32 = TF_NX + TF_NY + TF_NZ;
override TF_N: u32 = select(1u, TF_BOX * select(1u, 4u, TILE_2D), TILE);
override TF_ND: u32 = select(1u, TF_N, VISCOSITY);

var<workgroup> tf_a: array<vec4<f32>, TF_N>;
var<workgroup> tf_b: array<vec4<f32>, TF_N>;
var<workgroup> tf_d: array<vec4<f32>, TF_ND>;

// The face being computed lies on the domain's edge: block `g_face_slot`, below its cell
// `g_face_l` (which may be the ghost at BS).
var<private> g_face_wall: bool = false;
var<private> g_face_slot: i32 = 0;
var<private> g_face_l: vec3<i32> = vec3<i32>(0);

fn tile_box() -> vec3<i32> {
  return select(vec3<i32>(8, 8, 4), vec3<i32>(8, 8, 1), TILE_2D);
}

// Entry of the face along `axis` below the cell at offset f from the first cell of box `b`.
fn face_index(b: u32, axis: i32, f: vec3<i32>) -> i32 {
  let w = tile_box();
  var i = i32(b * TF_BOX);
  if (axis == 0) { return i + f.x + (w.x + 1) * (f.y + w.y * f.z); }
  i += i32(TF_NX);
  if (axis == 1) { return i + f.x + w.x * (f.y + (w.y + 1) * f.z); }
  i += i32(TF_NY);
  return i + f.x + w.x * (f.y + w.y * f.z);
}

// Every invocation of the workgroup must call this (it ends with a barrier); the loop runs the
// same number of times in each.
fn tile_fluxes(wid: vec3<u32>, li: u32) {
  let w = tile_box();
  for (var k = 0u; k < (TF_N + 255u) / 256u; k++) {
    let ii = k * 256u + li;
    if (ii < TF_N) {
      let b = ii / TF_BOX;
      let i = i32(ii % TF_BOX);
      var axis = 0;
      var f: vec3<i32>;
      if (i < i32(TF_NX)) {
        f = vec3<i32>(i % (w.x + 1), (i / (w.x + 1)) % w.y, i / ((w.x + 1) * w.y));
      } else if (i < i32(TF_NX + TF_NY)) {
        let j = i - i32(TF_NX);
        axis = 1;
        f = vec3<i32>(j % w.x, (j / w.x) % (w.y + 1), j / (w.x * (w.y + 1)));
      } else {
        let j = i - i32(TF_NX + TF_NY);
        axis = 2;
        f = vec3<i32>(j % w.x, (j / w.x) % w.y, j / (w.x * w.y));
      }
      // The box's first cell, as the update's own invocations decode it.
      let first = inv_interior(wid * vec3<u32>(8u, 8u, 4u) + vec3<u32>(0u, 0u, b));
      if (first.ok && !(BINS_ON && !blk_active(first.slot))) {
        let slot = first.slot;
        let lu = first.l + f;
        var e = vec3<i32>(0);
        e[axis] = 1;
        var nb = 0;
        if (lu[axis] == 0) { nb = bt(slot, 6u + u32(2 * axis)); }
        if (lu[axis] == BS) { nb = bt(slot, 7u + u32(2 * axis)); }
        g_face_wall = nb <= -2;
        g_face_slot = slot;
        g_face_l = lu;
        g_dx = blk_dx(slot);
        let F = flux_between(atl(blk_origin(slot), lu - e), e, axis);
        tf_a[ii] = vec4<f32>(F.m, F.mom);
        tf_b[ii] = vec4<f32>(F.e, F.ei, F.s, F.vn);
        if (VISCOSITY) { tf_d[ii] = F.d; }
      }
    }
  }
  g_face_wall = false;
  workgroupBarrier();
}

fn tile_flux(b: u32, axis: i32, f: vec3<i32>) -> Flux {
  let i = face_index(b, axis, f);
  let a = tf_a[i];
  let c = tf_b[i];
  var d = vec4<f32>(0.0);
  if (VISCOSITY) { d = tf_d[i]; }
  return Flux(a.x, a.yzw, c.x, c.y, c.z, c.w, d);
}

// Both faces along `axis` of the cell at offset f in box b.
fn tile_faces(b: u32, axis: i32, f: vec3<i32>, e: vec3<i32>) -> Faces {
  return Faces(tile_flux(b, axis, f), tile_flux(b, axis, f + e));
}

// Where a block's recorded face mass flux lives (amr/species_on.wgsl): face (axis, l) is the
// lower face of local cell l along that axis, l from 0 to BS inclusive (BS: the block's upper
// face). Requires amr/grid.wgsl.
//
// The records are one buffer. Past what one binding may hold (maxStorageBufferBindingSize, 2 GiB
// on NVIDIA, some 36,000 blocks' records in 3D) it is bound as two windows, MF and MF_HI, and the
// shaders' mf_load / mf_store go to the one an index lies in (AmrSim.massFluxWindows); a mesh
// whose records fit in one binding has MF alone, and binds nothing more than it ever did.

fn mf_index(slot: i32, l: vec3<i32>, axis: i32) -> u32 {
  let n = u32(BS + 1);
  let dz = select(n, 1u, is2d());
  let u = vec3<u32>(l);
  return (((u32(slot) * dz + u.z) * n + u.y) * n + u.x) * 3u + u32(axis);
}

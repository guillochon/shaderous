// A stencil pass on a uniform grid runs as two pipelines over the same workgroup grid
// (hydro/region_pass.ts). The INTERIOR one handles the cells of the box [lo, hi), whose whole
// stencil lies inside the grid, and its loaders skip the boundary mapping at compile time; the
// general one handles every other cell. A dispatch starts at cell REGION.origin.
override INTERIOR: bool = false;
// Workgroup size: 8 x 8 x 4 normally; thin along the normal for the slabs of the shell.
override WGX: u32 = 8u;
override WGY: u32 = 8u;
override WGZ: u32 = 4u;

// `end` is where this dispatch's own cells stop: its workgroups are rounded up, and the cells
// they reach beyond it belong to another slab, which must be the only one to handle them (a
// pass that adds to a total with atomics, such as the debris tally, would count them twice).
struct Region { origin: vec3<u32>, lo: vec3<u32>, hi: vec3<u32>, end: vec3<u32> }
@group(1) @binding(0) var<uniform> REGION: Region;

// The cell of this invocation, and whether this pipeline is the one that handles it.
struct RegionCell { c: vec3<i32>, mine: bool }

fn region_cell(gid: vec3<u32>) -> RegionCell {
  let cell = gid + REGION.origin;
  let inside = all(cell >= REGION.lo) && all(cell < REGION.hi);
  return RegionCell(vec3<i32>(cell), inside == INTERIOR && all(cell < REGION.end) && all(cell < P.dims.xyz));
}

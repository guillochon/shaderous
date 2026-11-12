// Shared by the species passes on the block grid. Species live in atlases laid out like the
// state's, with the groups of four stacked along z, SP_DEPTH texels apart (species/species.ts):
//   S   rho X of every block's cells (two of them, ping-pong)
//   SX  the mass fractions of every leaf's cells and ghosts (amr/species_assemble.wgsl)
//   SH  the same half a step ahead, for cells and the first ghost layer (amr/species_predict.wgsl)
#include "hydro/common.wgsl"
#include "amr/grid.wgsl"
#include "amr/list.wgsl"
#include <species_n>
#include "hydro/species_slopes.wgsl"

fn sp(c: vec3<i32>, g: u32) -> vec3<i32> {
  return c + vec3<i32>(0, 0, i32(g) * SP_DEPTH);
}

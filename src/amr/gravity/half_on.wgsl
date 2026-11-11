// The rate of change of the potential, with per-block timesteps (amr/bins_on.wgsl).
//
// A block's potential is solved each time the block finishes a step. DPHI is the rate at
// which it changed over the block's last step (amr/gravity/sync.wgsl): for a block that
// steps every substep, the same difference of the last two potentials that a single timestep
// extrapolates with. It carries the potential forward in time while the block is not being
// solved (amr/gravity/extrap.wgsl), and half a step ahead for the update's time-centred
// energy source. It is zero where there is no history on this mesh.
@group(0) @binding(6) var DPHI: texture_3d<f32>;

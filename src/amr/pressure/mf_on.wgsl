// Implicit pressure on blocks with species (amr/species.wgsl): the correction leaves in each
// block's record of face mass fluxes (amr/species_on.wgsl) the mass its faces carried over the
// whole step, advection and correction together, so that the species, moved after it, move as
// the density did and each one's mass is conserved (pressure/mf_on.wgsl on the uniform grid).
// A cell writes the faces the update's record gives it: its lower ones, and an upper one at the
// block's edge or beside a held cell. <ap_mf_op> is mf_store (the liquid: the record becomes the
// face's whole mass flux) or mf_add (a gas: the record gains what the correction carried).
// Requires amr/pressure/common.wgsl.
@group(0) @binding(12) var<storage, read_write> MF: array<f32>;
#include <mf_store>
#include <mf_add>
#include <ap_mf_op>
#include "amr/species_index.wgsl"

fn ap_mf_face(slot: i32, l: vec3<i32>, a: i32, up: bool, F: f32) {
  var q = l;
  if (up) {
    q[a] += 1;
    if (q[a] < BS && !ap_held(slot, q)) { return; }
  }
  ap_mf_op(mf_index(slot, q, a), F);
}

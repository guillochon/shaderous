// Per-block timesteps with a magnetic field (mhd/amr_mhd.ts, encodeSubstep()): the dispatch sizes
// of the flux, edge and update passes over the leaves starting a step (LL), chunk by chunk of
// the flux scratch. Chunk i holds the list's rows [i PER_CHUNK, (i + 1) PER_CHUNK); a chunk past
// the end of the list gets no rows. Two triples per chunk: boxes of BS + 2 texels per side (the
// fluxes), then of BS + 1 (edges and update).
@group(0) @binding(1) var<storage, read> LL: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<u32>;

override PER_CHUNK: u32 = 1u;
override CHUNKS: u32 = 1u;
override X_FLUX: u32 = 1u;
override X_BOX: u32 = 1u;

@compute @workgroup_size(1)
fn main() {
  let n = LL[0];
  for (var i = 0u; i < CHUNKS; i++) {
    let first = i * PER_CHUNK;
    let rows = select(0u, min(PER_CHUNK, n - first), n > first);
    OUT[6u * i] = X_FLUX;
    OUT[6u * i + 1u] = rows;
    OUT[6u * i + 2u] = 1u;
    OUT[6u * i + 3u] = X_BOX;
    OUT[6u * i + 4u] = rows;
    OUT[6u * i + 5u] = 1u;
  }
}

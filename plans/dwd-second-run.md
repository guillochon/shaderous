# Plan: a second magnetized double white dwarf run

The first 1024³ run (`examples/setups/double-wd-random-field.shaderous.json`, Unity job 2079786,
started 2026-11-21) showed what to change. This plan covers the second run: where to start it, a
sweep of exploratory runs to pick that start, and what it waits on. Nothing here starts until the
sweep's results and the optimizations are in.

## What the first run taught

| Issue | What happened | Fix in the second run |
| --- | --- | --- |
| Start | At `fill` 0.97 the pair ran about 3 orbits (t ≈ 60) before transfer grew, and 4.5 (t ≈ 88) before the merger | Start where transfer is just beginning: pick `fill` from the sweep (§1) |
| Frame after the merger | The followed frame turned at random once the donor broke up (Ω from 0.36 to 0.07 and back to 0.13 between t = 83 and 106) | `release: true`, in the example since commit 0145dbc (§2) |
| Block budget | Full at t = 60 (27,000 blocks with parents, 74 GB of an A100). The finest level then lost blocks to coarser ones as the debris spread | The coming optimizations (§3); a larger card or fewer levels if they are not enough |
| Box | 8 units (40,000 km from center to face). The tidal tail left through the faces: 3.5% of the mass by t = 104 | Box 16 at 2048³, the same finest cell (§4) |
| Center of mass | Drifted 0.003 per code unit in the inertial frame (0.13 by t = 80), from grid self-gravity's momentum error and uneven outflow | Watch it. Once the frame is released and at rest, the drift is seen as it is |
| Cost | 4.2 GB checkpoints: about 15 min to write and 20 min to reload. 2.8 to 4.2 code units per wall hour | The optimizations (§3) |

## 1. Exploratory sweep: at which fill does transfer begin?

**What "beginning" means.** The donor loses mass through L1 at a rate Ṁ. For a donor close to
an n = 1.5 polytrope, Ṁ ∝ (ΔR / R)³, where ΔR is how far the donor overfills its lobe. So
measure Ṁ at several fills, fit Ṁ = A (f − f₀)³ above the floor, and take f₀ as the onset. Below
f₀, Ṁ is a numerical floor: the atmosphere settling, and diffusion through L1. That floor depends
on resolution, so the onset has to be found at the resolution the run will use.

**How to measure Ṁ.** The stars carry dye, spread over (0, 1] by their order (check which value
is the donor's). Ṁ is the rate at which donor-dyed mass appears on the accretor's side of the L1 plane,
which stays put because the frame follows the pair. Write `--athdf` outputs every eighth of an
orbit (about 2.5 code units) and a script, like `quadfrac.mjs`, that sums that mass in each. The
stars are settled before t = 0, but leave out the first quarter orbit in case anything is still
adjusting. Also record ρ at L1 over the donor's central density, and look at slices for a stream.

**Runs.**
- **Fills:** 0.935 to 0.965 in steps of 0.005, which is 7 runs. 0.97 is known to overflow, so
  nothing at or above it.
- **Length:** 1 orbit each (t = 20), enough to see whether the donor overflows.
- **Grid:** the run's own: box 16, 2048³ with 5 levels, so the finest cell (0.0078) is the
  first run's. Five levels is the most the code allows; at 2048³ it leaves 64³ root cells. On Unity's `umd-cscdr-gpu` (the first run holds one of its
  12 A100s), all 7 at once, with `--gpu-memory 74`. The first run took about 12 minutes per
  code unit on an A100 at this stage, on the same finest cell, so each should finish in 4 to 5 hours.
  Eight checkpoints each, at 2 to 3 GB, come to about 150 GB of `/work`, which has 580 GB
  free. Delete them once Ṁ is measured.

**Result.** A table of Ṁ against fill, the fit's f₀, and the fill chosen for the run, with slices
at t = 10 and 20 for each fill.

## 2. Frame: release after the merger

Already done and pushed (0145dbc). The example turns it on. Check that the run's diagnostics
report `frame.release` and, after the merger, `atRest`.

## 3. Wait for the optimizations

Another session is working on optimizations. Before submitting:
- Confirm their commits are on main, and pull them on Unity.
- Rerun `gpu-check.sbatch`, plus a short timing run (2 code units of the chosen setup on an
  A100). Compare code units per hour, the block budget at 74 GB, and checkpoint write and reload
  times against the first run.
- If the sweep's mesh growth says the budget will fill before the merger, decide between the H200 (141 GB, preemptible partitions only), one level fewer, or a looser
  `refineDense`.

## 4. Decisions

1. **Box size: 16 at 2048³ with 5 levels** (80,000 km from center to face; chosen 2026-11-22).
   The finest cell stays 0.0078, the first run's, so the stars and the stream are resolved as
   before. The extra volume only adds coarse blocks where little gas goes. A box of 12 would have
   needed a cell 1.5 times coarser at 1024³, since grids come in powers of two.
2. **GPU.** The dedicated A100s (never preempted, 5 days) or the H200 (twice the memory,
   preempted twice before).
3. **Length.** 20 orbits after the start, as in the first run, or a set time after the merger.

## 5. Running it

As for the first run:
- `dwd.sbatch` on `umd-cscdr-gpu`, chained, with `KEEP=2`, `KEEP_EVERY=20` and checkpoints every
  6 code units (or fewer if writes are faster after §3).
- Screenshots (slice and volume) every code unit, fetched hourly by `sync.sh`, which also redraws
  `energies.png`.
- Movies from `make_movies.sh`.

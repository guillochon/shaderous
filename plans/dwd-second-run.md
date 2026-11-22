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
| Box | 8 units (40,000 km from center to face). The tidal tail left through the faces: 3.5% of the mass by t = 104 | A larger box (§4) |
| Center of mass | Drifted 0.003 per code unit in the inertial frame (0.13 by t = 80), from grid self-gravity's momentum error and uneven outflow | Watch it. Once the frame is released and at rest, the drift is seen as it is |
| Cost | 4.2 GB checkpoints: about 15 min to write and 20 min to reload. 2.8 to 4.2 code units per wall hour | The optimizations (§3) |

## 1. Exploratory sweep: at which fill does transfer begin?

**What "beginning" means.** The donor loses mass through L1 at a rate Ṁ. For a donor close to
an n = 1.5 polytrope, Ṁ ∝ (ΔR / R)³, where ΔR is how far the donor overfills its lobe. So
measure Ṁ at several fills, fit Ṁ = A (f − f₀)³ above the floor, and take f₀ as the onset. Below
f₀, Ṁ is a numerical floor: the atmosphere settling, and diffusion through L1. That floor depends
on resolution, so the onset has to be found at the resolution the run will use.

**How to measure Ṁ.** The stars carry dye, spread over (0, 1] by their order (the donor's is the
lower). Ṁ is the rate at which donor-dyed mass appears on the accretor's side of the L1 plane,
which stays put because the frame follows the pair. Write `--athdf` outputs every quarter orbit
(about 5 code units) and a script, like `quadfrac.mjs`, that sums that mass in each. Leave out
the first orbit, which is the stars settling. Also record ρ at L1 over the donor's central
density, and look at slices for a stream.

**Runs.**
- **Fills:** 0.92, 0.94, 0.95, 0.96, 0.97, 0.98 and 1.00. That's 7 runs, each 2 orbits (t = 40).
- **Grid:** 512³ with 4 levels, on Unity's `umd-cscdr-gpu`, at the same time on separate A100s.
  At about an eighth of 1024³'s blocks, each should take about 1 to 2 hours.
- **Then 1024³ at two fills** either side of the onset found at 512³, to see how far the onset
  moves with resolution (about 8 hours each). If it moves by more than 0.01, add one 1024³ run
  at a third fill.
- **Box:** the box chosen in §4, so the start is measured on the grid the run will use.

**Result.** A table of Ṁ against fill at both resolutions, the fit's f₀, and the fill chosen
for the run, with slices at t = 20 and 40 for each fill.

## 2. Frame: release after the merger

Already done and pushed (0145dbc). The example turns it on. Check that the run's diagnostics
report `frame.release` and, after the merger, `atRest`.

## 3. Wait for the optimizations

Another session is working on optimizations. Before submitting:
- Confirm their commits are on main, and pull them on Unity.
- Rerun `gpu-check.sbatch`, plus a short timing run (2 code units of the chosen setup on an
  A100). Compare code units per hour, the block budget at 74 GB, and checkpoint write and reload
  times against the first run.
- If the budget still fills before the merger in the 512³ sweep's mesh growth scaled up, decide
  between the H200 (141 GB, preemptible partitions only), one level fewer, or a looser
  `refineDense`.

## 4. Decisions

1. **Box size.** 12 (60,000 km) or 16. On the same finest cell (0.0078) this means n = 1536 or
   2048, with more root blocks and more memory. A larger box at the same n coarsens the finest
   cell to 0.0117 or 0.0156, which changes the onset (§1). The tail's mass loss argues for 16. The
   block budget may argue for 12.
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

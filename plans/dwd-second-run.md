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
| Orbit | Started at the point-mass Omega, the stars fell together on an eccentric orbit (e ~ 0.013: 2.7% closer at pericenter, which raises the fill there by more than a step of the sweep) | The orbit's Omega found as the stars settle (0f14845); the sweep's runs held their separation to 0.5% over an orbit |
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

**Result** (2026-11-23; Unity jobs 2128002 to 2128008, commit 0f14845). A first sweep (jobs
2119983 to 2119989) was stopped part way: its stars started at the point-mass Omega and fell together
on an eccentric orbit (above), which moved each run's fill by more than the sweep's step. The orbit's
Omega is now found as the stars settle, 0.74% (f 0.935) to 0.9% (f 0.965) above the point masses'.
Over the orbit the separation then held to within 0.5% at every fill (+0.02% at 0.935, -0.46% at
0.965), and the frame's axis stayed put.

Ṁ from `tools/headless/slurm/dwd-mdot.mjs`, the slope of the donor's gas past the L1 plane from
t = 5 to 20 (7 outputs), in M_sun per code time unit (2.75 s; an orbit is 20). m75 counts gas whose
dye is above 0.75; mdye weighs each cell by the donor's share of a donor-accretor mix (2 s - 1). The
two agree within 10% above 0.940.

| fill | Ṁ (m75) | Ṁ (mdye) | Ṁ, t = 5 to 12.5 / 12.5 to 20 (m75) | ρ(L1)/ρ_c, mean (last) | separation over the orbit |
| --- | --- | --- | --- | --- | --- |
| 0.935 | 1.7e-6 ± 0.4e-6 | 5.1e-6 | 0.9e-6 / 3.0e-6 | 1.5e-3 (1.9e-3) | +0.02% |
| 0.940 | 8.6e-6 ± 0.8e-6 | 1.2e-5 | 6.8e-6 / 1.1e-5 | 2.1e-3 (2.7e-3) | -0.01% |
| 0.945 | 1.8e-5 ± 0.1e-5 | 2.3e-5 | 1.6e-5 / 2.1e-5 | 2.7e-3 (3.5e-3) | -0.05% |
| 0.950 | 3.2e-5 ± 0.2e-5 | 3.8e-5 | 2.9e-5 / 3.7e-5 | 3.4e-3 (4.4e-3) | -0.10% |
| 0.955 | 5.2e-5 ± 0.3e-5 | 5.7e-5 | 4.6e-5 / 5.9e-5 | 4.2e-3 (5.2e-3) | -0.17% |
| 0.960 | 8.6e-5 ± 0.3e-5 | 9.3e-5 | 8.1e-5 / 9.0e-5 | 5.2e-3 (6.4e-3) | -0.34% |
| 0.965 | 1.2e-4 ± 0.04e-4 | 1.3e-4 | 1.1e-4 / 1.2e-4 | 6.2e-3 (7.5e-3) | -0.46% |

- **Fit.** Ṁ = A (f - f₀)³ over 0.940 to 0.965 (0.935 is the floor): A = 1.55, f₀ = 0.922 (m75); with
  the floor taken off, A = 1.74, f₀ = 0.924; mdye gives f₀ = 0.920 and 0.923. So f₀ = 0.922 ± 0.002.
  The law holds across the sweep: at 0.940 and 0.965 the fit is within 10% of what was measured.
- **No sharp onset in the range.** f₀ lies below every fill swept: at this resolution even 0.935
  carries a thin stream through L1 (slices), at about the floor's rate. Ṁ grows within the orbit
  by 10 to 35% from 0.940 up, and threefold at 0.935.
- **Slices** at t = 10 and 20 for each fill: `dwd-sweep-slices.png` and `f<fill>/00[24]0-t[12]0-slice.png`
  in `/work/pi_rfisher1_umassd_edu/guilloch_access-ci_org/dwd-sweep/`. A stream at every fill, and
  a halo of transferred gas around the accretor that grows with the fill.
- **Chosen fill: 0.940.** The lowest fill clearly above the floor by both measures, on the cube law,
  with Ṁ about 2e-4 M_sun an orbit: transfer just begun, and resolved. 0.935 would start nearer f₀
  but at the floor, where what is measured is partly numerical. How fast Ṁ runs away from there (the
  first run took 3 orbits from 0.97, with the eccentric start) is not measured by a one-orbit sweep.
- The checkpoints (88 GB, plus 25 GB of the stopped first sweep in `dwd-sweep-eccentric/`) are to
  be deleted once the run's start is settled.

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

**Started 2026-11-23** (Unity job 2165017 and the chain after it; Shaderous 4dca9bc):
`SETUP=tools/headless/slurm/dwd-run2.shaderous.json ZOOM=2 EVERY=0.4 ATHDF_EVERY=15 UNTIL=400
KEEP=2 KEEP_EVERY=20` into `/work/pi_rfisher1_umassd_edu/guilloch_access-ci_org/dwd-2048-f0.940/`.
Fill 0.940, box 16 at 2048³ with 5 levels, on the dedicated A100s; shots every 0.4 (about every 5
minutes at the start), a checkpoint every 6. Started without the optimizations of §3. The frame
turns at 0.31029, the settled Omega, as in the sweep.

As for the first run:
- `dwd.sbatch` on `umd-cscdr-gpu`, chained, with `KEEP=2`, `KEEP_EVERY=20` and checkpoints every
  6 code units (or fewer if writes are faster after §3).
- Screenshots (slice and volume) every code unit, fetched hourly by `sync.sh`, which also redraws
  `energies.png`.
- Movies from `make_movies.sh`.

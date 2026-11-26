# Plan: up to seven levels of refinement

Status (2026-11-26): done. Stages 1 to 5 are in; the numbers are under each stage. Two limits
came out of it: local timesteps stop at 5 bins (cycles of 32 substeps), and at 4 with
self-gravity (stage 4).

Why 7 and not more: with the resolutions offered (2D to 16384^2, 3D to 4096^3) and a whole
16-cell root block per side, 3D at 2048^3 stops at 7 (one root block) and 2D could reach 10.
Nothing in the data structures stops sooner (tree keys exact in a double far past 10;
`sph_monopole.wgsl`'s `1 << 3 * level` overflows an i32 only at level 11), but past 7 the coarse
levels take far more steps than they need (local timesteps reach only 32 times the finest, and
longer cycles need the finer levels re-meshed within a cycle), so 7 is where the plan stops.

Goal: let the adaptive mesh go to 7 levels below the root grid (128 times finer than the
roots), for problems whose interesting part is small against the box: a stellar core in a wide
orbit, a disruption's returning stream, a blast in a large cloud. Five levels is a constant in
a handful of places, not a limit of the data structures; the work is lifting it, finding what a
deeper mesh breaks or slows, and proving it on the validation suite.

## What limits the depth today

| Where | What | Kind |
|---|---|---|
| `src/create.ts` `maxAmrLevels` | `while (l < 5 && ...)` | hard cap |
| `src/setup/options.ts` `domain.amr` | `num(0, 5)` | hard cap (setup documents, Setup dialog) |
| `src/ui/link.ts` `amr` | "Levels of refinement, 0 to 5" | doc string |
| `src/ui/setup/dialog.ts` | `const most = n ? maxLevels(n) : 5` | default |
| `tools/headless/mcp.mjs` | `SETUP_KEYS`: "amr 0 to 5" | doc string |
| `tools/random-test/generate.ts` | `'domain.amr': (g) => g.int(1, 5)` | test range |
| `src/ui/app.ts` `RES_3D` | 3D menu stops at 2048 | practical cap (below) |
| `src/ui/link.ts` `decodeLink` | `Math.min(5, ...)` on `amr` | hard cap (every link, the runner and MCP server): missed by the first reading |
| `src/setup/options.ts` obstacle `refineLevel` | `num(0, 5)` | hard cap: missed by the first reading |
| `src/ui/app.ts` loading a snapshot | `Math.min(maxLevel(snap), maxAmrLevels(n))` | a 6- or 7-level `.athdf` is clamped to 5 on load |

What already handles any depth: the tree's numeric keys (`(level * span + c) ...`, at most
8 x 1024^3 in 2D at 16384^2, exact in a double), the per-level dispatch lists
(`listLevel`, sized from `maxLevel`), the leaf index, the flux correction, the regrid, the
multigrid on blocks. No WGSL array is sized by the number of levels (checked: the `array<_, 7>`
in `amr/assemble.wgsl` and `mhd/amr_common.wgsl` are the 7 face-neighbour reads).

## Consequences of a deeper mesh

1. **Root grid.** For a fixed finest resolution n, each level halves the roots. `maxAmrLevels`
   needs one whole 16-cell block per side at the root (n >> L >= 16). At 7 levels: n >= 2048.
   In 3D the menu stops at 2048, so 7 levels there is a single root block: legal, but the
   coarse level then resolves nothing. Useful 3D depth needs finest 4096^3 or 8192^3 (roots of
   2 or 4 blocks per side). 2D is fine as it is (16384^2 at 7 levels: 8 x 8 root blocks).
2. **Leaf index** (`BlockTree.indexGrid`, the `leaf index` texture in `amr/amr_sim.ts`): one
   r32uint per finest block, (n/16)^3, held on the CPU as well. It grows with the finest
   resolution, not the depth: 8 MB at 2048^3, 64 MB at 4096^3, **512 MB at 8192^3** (and a
   512^3 `Uint32Array` filled at the first re-mesh). 4096^3 is acceptable; 8192^3 needs a
   coarser index (below).
3. **Block budget.** A hydro block in 3D is about 1.2 MB (`AmrSim.blockBytes`, PAD = 20), so
   the default 2.5 GB holds about 2,000 blocks. A deep mesh only pays where the refined region
   is compact. `levelLimit` already lowers the reachable depth when the budget cannot hold it
   and says so in the status line; check that its message and the Setup dialog's estimate
   (`amrFits`, `gridBytes`) stay truthful at 7 levels.
4. **Time steps.** With global steps the root level takes the finest level's step: 128 times
   more than it needs at 7 levels. Local timesteps (docs/numerics.md section 18) cap a cycle at
   8 substeps (3 bins), so they recover at most a factor 8. Raising the cap to 32 or 128
   substeps (5 or 7 bins) is the main performance lever; it touches the bin assignment, the
   one-bin-apart sweeps and the flux bookkeeping between bins, which are written for any bin
   count but tested only up to 3.
5. **Elliptic solves.** The self-gravity and implicit-pressure multigrids on blocks cut the
   residual by only 0.7 to 0.85 per cycle across levels (comment in
   `amr/pressure/source_gas.wgsl`). Two more levels may need more V-cycles per step; measure
   the residual against depth.
6. **Initial mesh.** `samplesPerCell` (`amr/init_cpu.ts`) samples a root block up to
   `1 << (levels - level)` per cell under a per-pass budget, so a feature a few finest cells
   across must still be seen from the root at 7 levels: check with a small blast.
7. **Precision.** Positions and the clock are f32 in the shaders. At 7 levels the finest cell
   is 1/(16 x 2^7 x roots) of the box: 1/16384 at 8 root blocks, ~6e-5, well above f32's 6e-8
   relative step. The run's time over many more steps (dt / t near 1e-6 late in a long run)
   should be checked where the clock is summed in f32.

## Stages

**1. Lift the caps** (small). Replace the 5 with one exported constant (`MAX_AMR_LEVELS = 7`
in `create.ts`), used by `maxAmrLevels`, `options.ts`, the dialog's default and the doc
strings in `link.ts` and `mcp.mjs`. Keep the random tester at 1 to 5 until stage 4 passes.
Unit tests: `maxAmrLevels(2048) === 7`, `maxAmrLevels(1024) === 6`, a 7-level option
accepted by the setup compiler, an `.athdf` of 7 levels not clamped on load.

**2. First deep runs** (measure, fix what breaks). On a static build (memory:
gpu-tests-static-build):
- 2D Sedov blast at 16384^2 finest, 7 levels: mass and energy to round-off through the
  regrids, the blast radius against the similarity solution, steps per second against 5 levels
  at the same finest resolution.
- 3D Sedov at 2048^3, 7 levels (one root block): it runs, the mesh nests 2:1, conservation.
- 3D self-gravitating sphere (the Evrard collapse) at 2048^3, 6 and 7 levels: V-cycles per
  step and residual against depth.
- A magnetized blast at 6 levels: div B stays at round-off through the regrids.
Record the numbers here.

Numbers (2026-11-26, Ampere GPU, headless runner, to t = 0.02):

| 2D Sedov, 16384^2 finest | blocks at t = 0, 0.02 | steps | wall | mass | eTot 0 -> 0.02 |
|---|---|---|---|---|---|
| 5 levels (32 x 32 roots) | 1084, 6784 | 46,424 | 89 s | 1.000000 | 1.000025 -> 1.000032 |
| 7 levels (8 x 8 roots) | 148, 6052 | 46,424 | 98 s | 1.000000 | 1.000025 -> 1.000032 |

Same steps and the same energy to the digits printed; the 7-level mesh holds 11% fewer blocks
but runs 10% slower (more levels to dispatch per step). Global steps: the gain waits on stage 4.

The rest (static build, `--low-priority`, the default 2.5 GB budget):

| Run | levels | steps | wall | mass | eTot | note |
|---|---|---|---|---|---|---|
| 3D Sedov 2048^3, to t = 0.01 | 7 (one root block) | 35,434 | 442 s | 1 -> 0.9999999 | 1.000025 -> 1.000029 | budget full from t = 0.0025 (1919 leaves) |
| 3D Sedov 2048^3, to t = 0.01 | 5 (4^3 roots) | 35,438 | 444 s | 1 -> 0.9999999 | 1.000025 -> 1.000029 | budget full (1926 leaves) |
| Evrard 2048^3, 1500 steps (t = 2.66) | 6 | 1500 | 49 s | | | "block budget limits refinement to 3 of 6 levels" |
| Evrard 2048^3, 1500 steps (t = 2.66) | 7 | 1500 | 39 s | | | "4 of 7 levels"; mass leaves the open box after the bounce |
| MHD blast 2D 4096^2, to t = 0.05 | 6 | 2180 | 32 s | 1.000000 | 1.116576 -> 1.116563 | div B 2.7e-7 |
| MHD blast 2D 4096^2, to t = 0.05 | 4 | 2180 | 30 s | 1.000000 | 1.116576 -> 1.116563 | div B 2.7e-7 |

The 3D Sedov on one root block nests and conserves as on 5 levels, at the same speed. An Evrard
sphere fills too much of the box for 2.5 GB to refine it 6 or 7 levels deep: `levelLimit` lowers
the depth and the status line says so, truthfully. The multigrid's convergence against depth is
measured instead by the validation test `amr-gravity-depth` (Evrard, 256 finest, 2 to 4 levels
below roots of 64, 32 and 16 cells, to t = 0.3): the last relative residual 6.8e-5, 6.5e-5,
6.8e-5, two V-cycles a step beyond the first at every depth, energy lost 5.0e-5 at each. A
deeper tree costs the multigrid nothing in convergence. (`AmrSim.gravityCycles()` reads the
counts the gate pass kept, which nothing read before.)

**3. 3D beyond 2048** (only if stage 2 shows 3D depth is worth it). Add 4096 to `RES_3D` as
an adaptive-only resolution (never uniform). The index at 4096^3 is 64 MB: acceptable, but
count it in `gridBytes`. For 8192^3, make the index one entry per block at level
`levels - 1` with the deepest level resolved by the parent's child mask (an 8x smaller
texture), or a hash of leaves; decide only if 8192^3 is wanted.

Done: 4096 is in `RES_3D` (adaptive only, as every 3D grid past 256 is), and `gridBytes` and
`amrFits` count the leaf index (`leafIndexBytes`). 8192^3 not done.

**4. Local timesteps past 8 substeps.** Make the cycle length a setting (8, 32, 128), test the
bin sweeps and the conservation checks of section 18 at 5 and 7 bins, then measure the gain on
the stage 2 runs. Then widen the random tester to `g.int(1, 7)` with 2D at 7 levels allowed.

Done, to 5 bins: `tbins` (`mesh.timeBins`) takes 4 and 5, cycles of 16 and 32 substeps; the bin
passes were written for any count and needed no change. 7 bins (128 substeps) is not offered:
the mesh is re-checked only between cycles, and 128 substeps carry a feature some 43 cells, out
of the 16-cell buffer block around its refinement; it needs the finer levels re-meshed within a
cycle. With self-gravity the cap is 4 bins (`MAX_TIME_BINS_SELF_GRAVITY`): at 5 an Evrard
collapse (128 finest, 3 levels, to t = 0.8) lost 2.3e-3 of its energy, against 5.7e-4 with one
timestep and 6.0e-4 at 4. Validation `amr-time-bins-32`: Sod in 2D (1024 finest, 4 levels) and
Sedov in 3D (128, 3 levels) reach bin 5, with Sod's L1 error 0.0012, Sedov's radius within 1.6%,
mass and energy to 2e-7, and 76% of the block updates of one timestep. On the 7-level 2D Sedov
of `amr-seven-levels`, 5 bins make 29% of the block updates. The random tester draws 1 to 7
levels, a 2D run of 6 or 7 levels on the finest grid that holds them (1024 or 2048).

**5. Validation and docs.** Add one 7-level case to the validation suite (2D Sedov or a cloud
crushing in 2D, which runs in seconds), update docs/numerics.md (the depth, the index's size,
local-timestep bins), README's feature list, the CHANGELOG.

Done: `amr-seven-levels` (29 s): a 2D Sedov at 2048^2 finest on one 16-cell root block, to
t = 0.02, reaches level 7; its shock radius is within 0.3% of the cylindrical similarity
solution (xi0 = 0.984^-1/4) and the same as on 4 levels; mass and energy conserved to round-off;
fluxes consistent across levels; again with 5 bins. A 7-level `.athdf` (2D Sedov, 2048^2,
50 steps) saved by the runner and continued with `--from` keeps 7 levels and its 76 blocks. Docs: numerics.md section 17 (depth, the
index) and 18 (cycles of 32, the self-gravity cap), README, CHANGELOG (Unreleased).

Also from this work: in the Setup dialog every resolution and every number of levels can be
chosen; a choice the other cannot hold changes the other (more levels than the resolution has
root for raise the resolution; a grid too large for the GPU memory lowers the resolution, or
raises the levels when the resolution was the choice), and such choices carry an asterisk with
a tip saying what will change.

## Done when

A 2D run at 7 levels and a 3D run at 6 or more keep mass and energy to the same tolerances as
at 5 levels through their regrids, the validation suite passes with the 7-level case in it, a
7-level `.athdf` saves and loads unclamped, and the Setup dialog, the link, the runner and the
MCP server all offer 0 to 7.

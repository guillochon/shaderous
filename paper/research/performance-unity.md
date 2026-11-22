# Throughput on Unity's data-center GPUs, and what limits it there

The performance figure adds three data-center GPUs of the UMass Unity cluster to the RTX 3080 and the M4 Pro. This note records how they were measured, and the experiments that show why their throughput grows much less than their memory bandwidth. The figure's data are in `paper/figures/data/performance/` (`throughput_a100.json`, `throughput_h100.json`, `throughput_h200.json`); the experiments' raw outputs stayed on the cluster and in the code repository's ignored `runs/` folder, and their numbers are copied here.

## How the figure's points were measured

- The same set as on the desktop GPUs (`make_performance.py`, `GRIDS`), run by `tools/headless/slurm/throughput.sbatch`: three fresh pages in a row, each running `collect.mjs throughput` at normal priority; `make_performance.py --merge` keeps the median of each point and the range of the three.
- Code: commit 73423db, in a checkout of its own on the cluster (`~/shaderous-bench`), so that other runs' updates did not reach it.
- Browser: Playwright 1.65.0's Chromium headless shell (`chromium_headless_shell-1254`, user agent Chrome/158.0.8127.18) in the Playwright Docker image under Apptainer, on Vulkan, with `--disable-gpu-compositing` (needed on the H200). The desktop points ran in Chrome 156.
- One GPU per Slurm job, not shared with other jobs. Other GPUs of the same node may run other users' jobs.

| Device | nvidia-smi name, memory | Driver | Node | Slurm job | Peak bandwidth (datasheet) |
|---|---|---|---|---|---|
| A100 | NVIDIA A100-SXM4-80GB, 81920 MiB | 580.178.04 | umd-cscdr-gpu001 | 2116251 | 2039 GB/s |
| H100 | NVIDIA H100 80GB HBM3, 81559 MiB | 580.178.04 | uri-gpu012 | 2116252 | 3350 GB/s (SXM) |
| H200 | NVIDIA H200 NVL, 143771 MiB | 615.71.09 | gpu057 | 2116169 | 4800 GB/s (NVL, marked preliminary) |

The bandwidths and their sources are in `reference.json`. Stone2026 (Table 1) also gives AthenaK's hydrodynamic throughput on an H100, 677 x 10^6 cells/s, without saying which H100; its bar in panel (b) spans the PCIe card's 2000 GB/s (product brief PB-11133-001_v02, Table 2) to the NVL's 3900.

## Throughput against the RTX 3080 (Chrome)

At the largest uniform grids (medians from `make_performance.py`):

| Device | 2048^2 (10^9 /s) | 256^3 (10^9 /s) | Bandwidth / RTX 3080's |
|---|---|---|---|
| RTX 3080 | 1.85 | 1.39 | 1 |
| A100 | 1.74 (0.94x) | 1.20 (0.86x) | 2.2 |
| H100 | 2.60 (1.41x) | 1.57 (1.13x) | 3.7 |
| H200 | 2.74 (1.48x) | 1.67 (1.20x) | 5.3 |

The same code on AthenaK's devices: on an A100, 1.20 / 0.614 = 2.0 times AthenaK's updates per second; on an H100, 1.57 / 0.677 = 2.3 times. Per GB/s of bandwidth at 256^3, Shaderous takes 5.9 x 10^5 on the A100 against AthenaK's 3.0 to 4.0 x 10^5; times four (double precision and two stages, `like_for_like` in `reference.json`), AthenaK's 1.2 to 1.6 x 10^6 is 2.0 to 2.7 times Shaderous's.

## What limits the throughput on these GPUs

All on uniform grids, a Kelvin-Helmholtz instability on 2048^2 cells and a Sedov blast on 256^3, with diagnostic scripts that are not part of the code (`runs/probe/` in the code repository, run from a scratch checkout with small patches). Each value is the median of three timings of about 1.5 s each, which agreed to better than 1 percent.

**Clocks.** Logged every 0.2 s by nvidia-smi during the runs. The A100's SM clock stayed at 1382 to 1410 MHz (its maximum is 1410) at 218 to 249 W, with no throttling reason other than idle (jobs 2116787, 2119199).

**Where the time goes** (timestamp queries per pass, src/gpu/timing.ts; A100, job 2119199): on 256^3, the update pass 10.2 ms of a 14.1 ms step (72 percent), the predictor 3.2 ms (23 percent); on 2048^2, 61 and 27 percent.

**Memory traffic.** A step reads and writes roughly 210 bytes per cell (seven rgba32float textures, `Sim.cellBytes`, each read or written about twice). At 1.2 x 10^9 updates per second that is roughly 250 GB/s, an eighth of the A100's peak: an estimate, not a measurement, but far from the limit.

**Fetches against arithmetic** (A100, umd-cscdr-gpu001, job 2119343; ms per step):

| Variant | 2048^2 | 256^3 |
|---|---|---|
| As committed (Koren reconstruction, HLLC) | 2.42 | 14.08 |
| The update's loads of the predicted state doubled (EXTRA_FETCH: each load also loads a neighbor, times a factor that is zero at run time) | 2.51 (+3.5%) | 14.71 (+4.4%) |
| HLL instead of HLLC | 2.40 (-1%) | 13.82 (-2%) |
| minmod limiter | 2.02 (-17%) | 11.65 (-17%) |
| van Leer limiter | 2.01 (-17%) | 11.69 (-17%) |
| MC limiter | 2.07 (-14%) | 12.04 (-15%) |

Doubling the texture loads costs 4 percent; a cheaper reconstruction saves 17. On the A100 the update is limited by arithmetic, and the third-order reconstruction is the largest part of it. On the RTX 3080 (1024^2 and 128^3, desktop) the pattern is the same: 3 percent for doubled loads, 14 to 16 percent saved by a symmetric limiter.

**The Koren reconstruction's arithmetic** (A100, umd-cscdr-gpu002, job 2119619; ms per step):

| Variant | 2048^2 | 256^3 |
|---|---|---|
| As committed | 2.26 | 13.23 |
| The division by 3 as a multiplication | 2.26 | 13.23 |
| Also the smoothness weight's division as inverseSqrt squared | 2.27 | 13.27 |
| Also the unlimited value returned at once where every component is smooth | 2.10 (-7%) | 11.88 (-10%) |
| Only that early return | 2.10 (-7%) | 12.08 (-9%) |
| van Leer, for comparison | 1.88 | 10.78 |

The divisions cost nothing measurable. Returning early where the flow is smooth saves 7 to 9 percent and changes the result only by the round-off of the blend; it was added to the code (src/hydro/limiters/koren.wgsl) after the figure's points were measured. The rest of the reconstruction's cost is its design: two limited increments per cell along each axis, where a symmetric limiter has one slope for both faces.

**Not helpful.**
- Workgroup shapes (A100, job 2117056): 22 shapes of 64 to 256 invocations for the update and the predictor, all within 3 percent of the defaults (best 1.89 against 1.85 x 10^9 on 2048^2, 1.30 against 1.28 x 10^9 on 256^3).
- Dawn's robustness and validation off (`--enable-dawn-features=disable_robustness,skip_validation`): on the A100 with driver 595.91.07 a step became 20 percent slower on 2048^2 and 57 percent slower on 256^3 (job 2119026, off and on twice, repeating to 0.1 percent); on the H100 with driver 580.178.04, 10 and 15 percent faster (job 2117052). Not used.
- Larger uniform grids: 512^3 lost the device on every GPU tried; a two-dimensional grid is held in 3D textures, at most 2048 per side.

Node and driver matter at the level of a few percent: the same A100 model gave 1.20 x 10^9 on 256^3 on umd-cscdr-gpu001 (driver 580.178.04) and 1.28 x 10^9 on umd-cscdr-gpu002 (595.91.07).

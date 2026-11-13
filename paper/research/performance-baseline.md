# Throughput of the GPU versions of Athena, for the performance figure

The performance section compares Shaderous with a published throughput of an established code on similar hardware. This note collects the single-device numbers that the two GPU versions of Athena report, with what each measurement ran, from the papers themselves:

- `Stone2026`: AthenaK, ApJS 283, 27. Read from the arXiv source, `arXiv:2409.16053` (the September 2024 version), in `research/stone2026/fulltext.tex`. The published full text in SciX contains the same A100 hydrodynamics figure (614), but the published tables were not compared line by line.
- `Grete2021`: K-Athena, IEEE TPDS 32, 85. Read from the arXiv source, `arXiv:1905.04341`, in `research/grete2021/fulltext.tex`.

All throughputs are cell updates per second on one device, counting one update per cell per full timestep.

## AthenaK (`Stone2026`, Section 5.1, Tables 1 and 2)

| Device | Hydrodynamics (10^6 cells/s) | MHD (10^6 cells/s) |
|---|---|---|
| Apple M1 Pro (8 cores) | 34 | 11 |
| Intel Xeon Gold 6326 (32 cores) | 63 | 33 |
| NVIDIA V100 | 298 | 158 |
| NVIDIA A100 | 614 | 298 |
| AMD MI250 | 405 | 190 |
| NVIDIA H100 | 677 | 290 |
| NVIDIA Grace Hopper | 1077 | 565 |

- Test: a linear wave convergence test on a uniform mesh. On GPUs the grid is 512 x 256 x 256 cells in MeshBlocks of 128^3; on CPUs, a single MeshBlock of 128^3 on all cores.
- Block size matters. On the A100, MeshBlocks of 64^3, 32^3 and 16^3 on the same grid give 413, 242 and 126 x 10^6 cells/s instead of 614.
- Adaptive mesh: a 2D MHD blast on one A100 runs at 181 x 10^6 cells/s on a uniform 4096^2 grid with 64^2 MeshBlocks, and at 141 x 10^6 with a 1024^2 root grid and two levels. The paper calls this about a 20% overhead.
- Not stated for the benchmark: the reconstruction, Riemann solver and time integrator (the code offers PLM, PPM and WENOZ; LLF, HLLE, HLLC, HLLD and Roe; SSP RK1 to RK4), and the floating-point precision. The only mention of `double` in the paper is for particle data. The variant of each device (for example the A100's memory size) is not given either.

## K-Athena (`Grete2021`, Sections 3 and 3.4)

- Algorithm for all results unless noted: double precision, unsplit adiabatic MHD with van Leer integration, piecewise linear reconstruction, the Roe Riemann solver and constrained transport, on a 3D linear fast magnetosonic wave.
- One NVIDIA V100: more than 10^8 cells/s at the largest grids that fit in memory. The throughput depends strongly on the grid size: on small grids it is more than ten times lower. The plateau is set by DRAM bandwidth.
- Roofline on Summit's V100s: 0.82 TFLOPS achieved against 1.13 TFLOPS allowed by DRAM bandwidth, 72.5% of the bandwidth limit.
- Weak scaling on Summit: 1.94 x 10^12 cells/s in total on 24,576 V100s (the conclusions print 24,567), about 7.9 x 10^7 per GPU at 76% parallel efficiency.
- CPUs, for scale: about 8 x 10^6 cells/s on a 14-core Broadwell (Xeon E5-2680) and 1.4 x 10^7 on a 20-core Skylake (Xeon Gold 6148).
- There is no hydrodynamics-only number; the paper measures MHD throughout.

## What makes a comparison with Shaderous fair

Shaderous's own measurement (`paper/figures/data/performance/throughput.json`, 2026-11-13, the GPU otherwise idle, normal priority, median of three runs): 1.45 x 10^9 cells/s for a 3D Sedov blast on a uniform 256^3 grid, and 1.88 x 10^9 in 2D at 2048^2, on an NVIDIA RTX 3080 12 GB (Ampere) in Chrome 156. These are the largest uniform grids the app offers (UNIFORM_MAX in src/ui/app.ts), and the largest that run here. Every uniform grid, 2D included, is stored in 3D textures, whose side WebGPU caps at 2048 on this adapter (maxTextureDimension3D), so 4096^2 cannot be allocated; Sim.create now says so instead of running steps that do nothing. A 512^3 grid needs seven rgba32float textures of 2 GiB each plus the multigrid levels, about 17 GB on a 12 GB card, and Chrome drops the GPU instance while allocating it, before WebGPU reports running out of memory. The earlier 9.3 x 10^8, and the 5.9 x 10^7 with four levels, were measured while the GPU had other work.

- Physics: compare with AthenaK's hydrodynamics column. K-Athena reports only MHD, which AthenaK finds to be about two times slower than hydrodynamics on GPUs.
- Precision: Shaderous runs in single precision. K-Athena runs in double; AthenaK does not say. Single precision moves half the bytes per cell, so on a bandwidth-limited code it is worth up to about a factor of two by itself.
- Stages: Shaderous takes one MUSCL-Hancock step per update. K-Athena names van Leer integration without saying here how many stages it takes, and AthenaK does not state its integrator for the benchmark. A two-stage integrator does about twice the work per update.
- Grid size: GPU throughput rises with the grid until the device is full. Compare plateau values: AthenaK's 128^3 MeshBlocks on 512 x 256 x 256, K-Athena's largest grids, and Shaderous's 256^3.
- Hardware: the codes are bandwidth-limited, so dividing by each device's peak memory bandwidth puts the V100 and A100 numbers beside the RTX 3080's. The bandwidths should come from the vendors' datasheets, for the device variants that the papers used, which the papers do not name.
- Test problem: linear waves (both Athena papers) against a Sedov blast or Kelvin-Helmholtz (Shaderous). A smooth wave and a blast do the same work per cell in a code without adaptive physics, but Shaderous's blast also exercises its shock detector and first-order fallback.

## Bandwidths and the comparison

Peak memory bandwidth from the vendors' datasheets (sources in `paper/figures/data/performance/reference.json`): V100 900 GB/s (SXM2 and PCIe alike), A100 1555 GB/s (40 GB) or 2039 GB/s (80 GB SXM; 1935 for the 80 GB PCIe), RTX 3080 12 GB 912 GB/s (the 10 GB card has 760). Cell updates per second per GB/s:

| | per GB/s |
|---|---|
| Shaderous, 3D uniform 256^3 | 1.59 x 10^6 |
| Shaderous, 2D uniform 2048^2 | 2.06 x 10^6 |
| AthenaK, V100 | 3.3 x 10^5 |
| AthenaK, A100 | 3.0 to 3.9 x 10^5 |

Shaderous takes 4 to 5 times as many updates per unit of bandwidth in 3D. Single precision (up to 2) and one stage against two (up to 2) could explain a factor of up to 4.

## Next

1. Measure an integrated GPU, and Firefox.

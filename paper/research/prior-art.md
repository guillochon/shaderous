# Prior art for hydrodynamics in graphics shaders

Searched 2026-11-10 with the Valency corpus (semantic and keyword search over arXiv, PubMed, EarthArXiv and others), SciX/ADS (abstract and full-text queries for "WebGPU", "WebGL", "compute shader", "fragment shader", "graphics hardware" with hydrodynamics, conservation laws, Euler equations), and OpenAlex (for graphics venues such as ACM TOG and Eurographics, which ADS indexes poorly).

## What we found

The literature splits into four groups.

1. **Shader-era GPGPU (before CUDA, 2003 to 2007).** Numerical PDE solvers written as fragment programs on graphics textures.
   - Krüger & Westermann (2003, `Kruger2003`) and Bolz et al. (2003, `Bolz2003`): linear algebra, conjugate gradients and multigrid on the GPU.
   - Goodnight et al. (2003, `Goodnight2003`): a multigrid solver for boundary value problems in fragment programs.
   - Harris et al. (2003, `Harris2003`): incompressible cloud dynamics on graphics hardware, with 3D data tiled into 2D textures.
   - Hagen, Lie & Natvig (2006, `Hagen2006`): the compressible Euler equations in 2D and 3D with high-resolution finite-volume schemes on graphics cards. This is the closest earlier work to the hydro core of Shaderous. It is a uniform grid with no gravity and no adaptive mesh.
   - Lefohn et al. (2006, `Lefohn2006`, Glift): adaptive, random-access GPU data structures (page tables, octrees) in textures. Closest earlier work to a texture atlas of blocks.
   - Owens et al. (2007, `Owens2007`): survey of the field.
2. **CUDA-era astrophysical codes and portability layers.** Wang, Abel & Kaehler (2010, `Wang2010`) and GAMER (`Schive2010`, `Schive2018`) for AMR on GPUs; Kestener et al. (2010, `Kestener2010`); Cholla (`Schneider2015`); FARGO3D (`BenitezLlambay2016`); Astaroth (`Pekkila2017`); K-Athena (`Grete2021`); Quokka (`Wibking2022`); Idefix (`Lesur2023`); Kratos (`Wang2025`); AthenaK (`Stone2026`); AGILE (`Porth2026`); RAMSES-GPU (`Teyssier2026`). All are native applications built with a vendor toolchain (CUDA, HIP) or a portability layer (Kokkos, OpenACC). AMR codes among them correct coarse-fine fluxes with flux registers (Berger & Colella 1989) or an equivalent stored-flux fix-up.
3. **Simulators that run in a browser on the GPU.** Kaboudian et al. (2019, `Kaboudian2019`, WebGL 2, reaction-diffusion and cardiac models); Baggaley (2026, `Baggaley2026`, WebGPU, split-step Fourier solver of the Gross-Pitaevskii equation, uniform 2D grid); a WebGL lattice Boltzmann solver (Dovgiy et al. 2021); WebGPU cloth (Sung et al. 2025); browser comet dust tails (Balodis et al. 2027) and a WebGPU cosmic-web visualizer (Schuster et al. 2026). None solves the compressible Euler equations with shocks, self-gravity or adaptive refinement.
4. **Graphics fluid solvers** (Stam-style stable fluids, SPH, lattice Boltzmann) for visual effects. Incompressible or particle-based; not conservative shock-capturing schemes.

## What appears to be new in Shaderous

We found no earlier code that does any of the following. The manuscript states these with the closest earlier work cited, as `writing-style.mdc` requires.

- A shock-capturing, self-gravitating, block-AMR astrophysical code written entirely in a portable graphics API (WebGPU/WGSL) and running in an unmodified browser.
- Conservation to round-off on an adaptive mesh **without storing any face flux and without flux registers**: fluxes are gathered (each cell computes its own faces), the two cells sharing a face compute bit-identical values, and at a refinement boundary the coarse cell recomputes the fine fluxes. With per-block timesteps, a slower block recomputes its faster neighbor's fluxes from stored states (half at the start of its step, half at the midpoint).
- Ghost cells that are never exchanged: the pass that builds primitives computes each ghost texel from wherever its value lives (neighbor block, boundary image, restriction of finer cells, interpolation from a coarser block). This follows from WebGPU forbidding read-write access to `rgba32float` storage textures.
- A fully GPU-resident step: timestep reduction and the assignment of blocks to time bins happen on the GPU; the CPU reads back only regridding indicators and diagnostics, without blocking.
- Single precision throughout (WGSL has no `f64`) combined with a well-balanced reconstruction and grid-consistent settling of stars, so that hydrostatic stars and stratified atmospheres stay in balance.

## Searches that returned nothing relevant

- "WebGPU" AND (hydrodynamics OR "Riemann solver") in SciX full text: 9 hits, none a compressible solver.
- Valency semantic search for browser-based compressible astrophysical hydrodynamics: top hits were cloth, LLM inference, visualization.

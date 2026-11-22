# Citation check, batch 1

Sources saved under `paper/research/<key>/`: harris2003 (PDF + fulltext.txt, TU Wien course copy of the Graphics Hardware 2003 paper), lefohn2006 (eScholarship PDF + fulltext.txt), koren1993 (CWI PDF + fulltext.txt), stone2026, stone2020, wibking2022, baggaley2026 (arXiv LaTeX). Metadata checked against ADS, Crossref and the Eurographics Digital Library.

## Harten1983
- Metadata: OK (Crossref 10.1137/1025002: Harten, Lax, van Leer; SIAM Review 25(1), 35-61, 1983)
- Claim: "the HLL solver of Harten1983" -> OK
  Evidence: Full text not read (SIAM, closed). The abstract covers Godunov-type schemes "that result from using an approximate solution of the Riemann problem". Davis (1988), in his abstract, credits the one-intermediate-state solver to Harten, Lax and van Leer.

## Davis1988
- Metadata: OK (Crossref 10.1137/0909030: S. F. Davis; SIAM J. Sci. Stat. Comput. 9(3), 445-473, 1988)
- Claim: "Wave speeds are the estimates of Davis1988, S_L = min(u_L - c_L, u_R - c_R), S_R = max(u_L + c_L, u_R + c_R)" -> UNVERIFIED (exact formula)
  Evidence: Full text not available (SIAM, closed). The abstract confirms that the paper proposes signal-speed bounds for the HLL solver: "we propose a number of algorithms for obtaining these bounds". The specific min/max formula is the one commonly attributed to Davis (1988), but I could not see it in the source. Davis gives several alternatives, so if the formula is checked, consider naming which one it is.
- Claim: "HLL with the fast magnetosonic speeds in the estimates of Davis1988" -> OK as worded
  Evidence: Davis wrote for gas dynamics. "In the estimates of" correctly presents the use of fast speeds as an adaptation, not something Davis did.

## Toro2009
- Metadata: OK (Crossref 10.1007/b79761: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction*, Springer Berlin Heidelberg 2009; ISBN 978-3-540-25202-3 is the 3rd edition)
- Claim: "unsplit MUSCL-Hancock method in the form given by Toro2009" -> OK (book not read)
  Evidence: Crossref chapter list: ch. 14 "High-Order and TVD Schemes for Non-Linear Systems" (pp. 493-529, where MUSCL-Hancock is presented) and ch. 16 "Methods for Multi-Dimensional PDEs". The book's full text was not read.
- Claim: "a strong shock tube from Toro2009" -> OK (book not read)
  Evidence: Ch. 4, "The Riemann Problem for the Euler Equations" (pp. 115-162), holds the standard test set. The exact test was not seen.
- Claim: "texts of LeVeque2002 and Toro2009" (teaching) -> OK
- Claim: "an exact Riemann solver (Toro2009)" -> OK
  Evidence: Ch. 4 gives the exact Riemann solver for the Euler equations (Crossref chapter metadata).

## Koren1993
- Metadata: OK. CWI repository (ir.cwi.nl/pub/2269) and the chapter PDF: Koren, "A robust upwind discretization method for advection, diffusion and source terms", Chapter 5, pp. 117-138. Book: *Numerical Methods for Advection-Diffusion Problems*, eds. Vreugdenhil & Koren, Vieweg (Braunschweig) 1993. The series number 45 of Notes on Numerical Fluid Mechanics is the usual citation, but I did not see it in a primary record.
- Claim: "default limiter of Koren1993" / "The limiter is that of Koren1993" -> OK
  Evidence: Sec. 5.2, Fig. 5.3: the "novel κ=1/3-limiter", φ(r) = max(0, min(2r, min(1/3 + 2r/3, 2))) (eq. 5.17).
  Note: the 2r and 2 bounds are already part of Koren's limiter, so "with the bounds of the MC limiter" restates Koren's own form. That is not wrong; the vanLeer1977 part is outside this batch.

## Harris2003
- Metadata: OK (Eurographics DL: Harris, Baxter III, Scheuermann, Lastra; Graphics Hardware 2003, pp. 92-101; DOI 10.2312/EGGH03/092-101; DOIs are case-insensitive and the bib form resolves)
- Claim: "incompressible cloud dynamics", "solvers were written as fragment programs that read and wrote textures" -> OK
  Evidence: Sec. 2: "described by the incompressible Euler equations of fluid motion". Sec. 5: "We implement computations using fragment programs written in the Cg shading language".

## Goodnight2003
- Metadata: OK (Eurographics DL: Goodnight, Woolley, Lewin, Luebke, Humphreys; Graphics Hardware 2003, pp. 102-111; DOI 10.2312/EGGH03/102-111)
- Claim: "multigrid ... in fragment programs" -> OK
  Evidence: Abstract: "a system, built on programmable graphics hardware ... implements the multigrid method". Full text not read.

## Bolz2003
- Metadata: OK (Crossref 10.1145/882262.882364: Bolz, Farmer, Grinspun, Schröder; ACM TOG 22(3), 917-924, 2003). The batch summary showed "?" for the authors, but the bib entry has them.
- Claim: "linear algebra and multigrid"; "ran multigrid in the same way" -> OK
  Evidence: Abstract: "a sparse matrix conjugate gradient solver and a regular-grid multigrid solver" on the GeForce FX. Full text not read.

## Kruger2003
- Metadata: OK (Crossref 10.1145/882262.882363: Krüger, Westermann; ACM TOG 22(3), 908-916, 2003)
- Claim: "linear algebra" on GPUs before CUDA -> OK
  Evidence: Abstract: "a framework for the implementation of linear algebra operators on programmable graphics processors". Full text not read.

## Owens2007
- Metadata: OK (Crossref: Owens, Luebke, Govindaraju, Harris, Krüger, Lefohn, Purcell; Comput. Graph. Forum 26(1), 80-113, 2007)
- Claim: "surveyed that period" -> OK
  Evidence: Abstract: "we describe, summarize, and analyze the latest research in mapping general-purpose computation to graphics hardware".

## Lefohn2006
- Metadata: OK (Crossref 10.1145/1122501.1122505: Lefohn, Sengupta, Kniss, Strzodka, Owens; ACM TOG 25(1), 60-99, 2006). The eScholarship preprint lists the authors in a different order (Lefohn, Kniss, Strzodka, Sengupta, Owens). The bib follows the published order, which is correct.
- Claim: "built page tables and octrees in textures for random access on the GPU" -> OK
  Evidence: Sec. 5.2: "Many of the GPU-based sparse or adaptive structures in Table II are based on a page table design". Abstract: "a stack, quadtree, and octree"; data "retrieved from the physical memory texture".

## Baggaley2026
- Metadata: OK (2026arXiv261104060B, A. W. Baggaley, arXiv:2611.04060)
- Claim: browser GPU simulator "for the Gross-Pitaevskii equation" -> OK
  Evidence: Abstract: runs "in a web browser ... It solves eight models built on the Gross-Pitaevskii equation". GPU engine "in WGSL compute shaders".
- Claim: shows "interactive computing on the GPU through WebGL and WebGPU is practical" -> OK
  Evidence: Condensate uses WebGPU ("a WebGPU solver fast enough for a phone to run a 256^2 grid interactively"). Kaboudian2019 supplies the WebGL half.

## Teyssier2026
- Metadata: OK (2026arXiv260720704T; all 12 authors match ADS)
- Claim: RAMSES-GPU is a GPU astrophysics code -> OK
  Evidence: Abstract: "implementation and optimization of ... RAMSES on Graphics Processing Units (GPUs) using CUDA Fortran".

## Porth2026
- Metadata: OK (2026arXiv260719277P; all 13 authors match ADS). The batch summary showed "?" for the authors, but the bib has them.
- Claim: AGILE is a GPU astrophysics code -> OK
  Evidence: Abstract: "AGILE, a GPU-enabled adaptive mesh refinement (AMR) framework".

## Stone2026
- Metadata: OK (2026ApJS..283...27S; ApJS 283, 27; doi 10.3847/1538-4365/ae3717; 9 authors match)
- Claim: a GPU code in the list -> OK
- Claim: "AthenaK, the GPU version of Athena++ written with Kokkos" -> OK
  Evidence: Abstract: "a new implementation of the Athena++ block-based adaptive mesh refinement (AMR) framework using the Kokkos programming model".
- Claim: "3.0e8 hydrodynamic cell updates per second on one NVIDIA V100 and 6.1e8 on one A100" -> OK
  Evidence: Table "Performance of AthenaK on various devices", hydro row: V100 298, A100 614 (Mzcps).
- Claim: "on a uniform grid of 3.4e7 cells" -> OK
  Evidence: Sec. "Single-Device Performance": "On GPUs, we use a 512x256x256 grid tiled with 128^3 MeshBlocks" (3.36e7 cells). The benchmark is a linear wave convergence test.
- Claim: "Stone2026 do not state the precision or the integrator of their benchmark" -> OK
  Evidence: The performance section names neither. In the whole arXiv source, "precision" appears only as "machine precision", and "double" only for particle data. Integrators are named for other tests (RK2, RK3), not for the throughput benchmark. Checked against the arXiv LaTeX (2409.16053), not the ApJS typeset version.

## Wang2025
- Metadata: OK (2025ApJS..277...63W; ApJS 277, 63)
- Claim: Kratos is a GPU astrophysics code -> OK
  Evidence: Abstract: "a novel GPU-based simulation system".

## Lesur2023
- Metadata: OK (2023A&A...677A...9L; A&A 677, A9; 6 authors match)
- Claim: Idefix is a GPU code -> OK
  Evidence: Abstract: tested on "NVidia and AMD GPUs (up to 1024 GPUs on Adastra)".

## Wibking2022
- Metadata: OK (2022MNRAS.512.1430W; MNRAS 512, 1430-1449)
- Claim: Quokka is a GPU code -> OK
- Claim: "The method follows the two-moment modules of ... QUOKKA" -> OK
  Evidence: Sec. 1 / 3: two-moment radiation moment equations with a local closure; the default is the Levermore (1984) M1 closure.
- Claim: "The usual remedy replaces c by a reduced speed ĉ everywhere (Skinner2013, Wibking2022)" -> OK
  Evidence: Sec. 2: "we replace this 1/c factor with a factor of 1/ĉ, where ĉ is the reduced speed of light". It is a single global ĉ, for example "We use a reduced speed of light ĉ = 0.1c".

## Grete2021
- Metadata: OK (2021ITPDS..32...85G; IEEE TPDS 32(1), 85-97; doi 10.1109/TPDS.2020.3010016)
- Claim: K-Athena is a GPU code -> OK
  Evidence: Abstract: "K-Athena achieves >10^8 cell-updates/s on a single V100 GPU".

## Stone2020
- Metadata: OK (2020ApJS..249....4S; ApJS 249, 4; 4 authors match)
- Claim: "Athena++ ... large native program" (with FLASH, RAMSES) -> OK
  Evidence: Sec. 7: C++11 code that needs a C++ compiler, with optional MPI, OpenMP and HDF5.
- Claim: "The integrator is the van Leer predictor-corrector of StoneGardiner2009, as in Athena++" -> OK
  Evidence: Sec. 3: "in Athena++ the VL2 integrator is implemented"; listed as "VL2: (default) predictor-corrector midpoint method".
- Claim: "exported as an HDF5 file in the layout of Athena++ (Stone2020)" -> OK as a code citation
  Evidence: Stone2020 does not describe its HDF5 output layout. It mentions HDF5 only as an optional library dependency (Sec. 7). The citation identifies the code, which is acceptable. A pointer to the Athena++ documentation would be needed only if the layout itself is to be referenced.

# Citation check, batch 2

Metadata was checked against SciX records and Crossref (DataCite for the Zenodo DOI). Full texts saved under `paper/research/<key>/` where they could be had. Kaboudian2019, Fleischmann2020, Kappeli2014, Cada2009, Thornber2008 and Kestener2010 are not on arXiv. Their full texts could not be downloaded: PMC/NCBI could not be reached from here, and the journals are paywalled. For those, the checks rest on the abstract plus SciX full-text index term counts (`body:"term"` hit or miss), which show that a word appears but not the sentence it is in.

## Fleischmann2020
- Metadata: OK (2020JCoPh.40109004F; JCP 401, 109004; doi 10.1016/j.jcp.2019.109004; authors Fleischmann, Adami, Hu, Adams)
- Claim: "scaling the acoustic wave speeds instead, as Fleischmann2020 do" -> OK
  Evidence: Abstract: they cure the shock instability by "a very simple modification of the eigenvalue calculation in the Roe approximation", which raises the "numerical" Mach number locally. In other words they rescale the acoustic eigenvalues. Full text not readable (paywalled). Note: their fix was made for the Roe flux, as a cure for the carbuncle, not as a low-Mach fix for HLLC (the body does mention HLLC per the index). "Also removes the damping of pressure jumps" and "unstable at Mach 0.03" are our own findings, not attributed to them.
  Optional wording: "Scaling the acoustic wave speeds instead, as \citet{Fleischmann2020} do in the Roe flux to cure the carbuncle, ..."

## Mallet2026
- Metadata: OK (DataCite: "taranis: a spectral solver for nonlinear plasma models in JAX", Mallet, Alfred; Abraham, Esrom; 2026, v0.1.0, issued 2026-08-18; doi 10.5281/zenodo.21987739)
- Claim: "wrote a WebGPU solver for reduced MHD that runs in the browser, a port of their pseudospectral code taranis" -> OK
  Evidence: repo webgpu/README.md, "2D app": "A browser-based port of the repo's 2D RMHD solver: pseudospectral (numpy-rfft2 layout), LSRK33 integrating-factor stepper". The Zenodo abstract says it "Includes a browser-based WebGPU RMHD solver".
- Claim: "closest earlier code ... ports the pseudospectral algorithms of their JAX code taranis to the browser" -> OK
  Evidence: same as above. webgpu/SPEC.md: "implementing the 2D forced-RMHD solver of this repo (taranis) on WebGPU".
- Claim: "The feature [OU-driven turbulence forcing] and the problems that use it follow the WebGPU solver" -> OK (as attribution of inspiration)
  Evidence: SPEC.md §4 "Forcing: Ornstein–Uhlenbeck process", shell-restricted modes. Default preset is forced turbulence. Note: taranis forces the Elsasser fields of incompressible RMHD. It has no Federrath solenoidal/compressive projection, which is correctly credited to Federrath2010b.
- Claim: "Following the WebGPU solver, further problems drive turbulence in an isothermal box with and without a guide field, thread the KH layer with a field, and collide counter-propagating Alfvén waves, both as periodic waves and as packets" -> PARTLY
  Evidence: the presets are forced turbulence, "2D Kelvin–Helmholtz shear layers" ("ψ_eq is the same profile scaled by b₀, i.e. an in-plane field along ŷ"), and "3D Alfven-wave collision ... two counter-propagating sinusoidal packets". Both collision initial conditions (sinusoidal pair and letters) have Gaussian z-envelopes ("the same icGaussZ + packetGeom envelopes"), so taranis has packets only, not periodic waves that fill the box. Its solver is incompressible RMHD, which always has a guide field, so "isothermal" and "without a guide field" are ours.
  Fix: "Following the WebGPU solver of \citet{Mallet2026}, further problems drive turbulence in a box (here isothermal, with and without a guide field), thread the Kelvin-Helmholtz layer with a field, and collide Alfvén waves that travel in opposite directions along a guide field, as packets \citep{Verniero2018} and also as periodic waves \citep{HowesNielson2013}."

## Kaboudian2019
- Metadata: OK (2019CSF...121....6K; Chaos Solitons Fractals 121, 6-29; doi 10.1016/j.chaos.2019.01.005)
- Claim: "Simulators that use the GPU from a browser exist for reaction-diffusion and cardiac models" -> OK
  Evidence: Abstract: WebGL 2.0 with Abubu.js, "running in a web browser". Full text (PMC8580290) could not be fetched. The SciX full-text index has body hits for "cardiac", "reaction-diffusion", "Fenton-Karma", "Gray-Scott", "Barkley" and none for a control word, so the paper does contain such models. The exact sentences were not read.
- Claim: "browser simulators of Kaboudian2019 and Baggaley2026 show that interactive computing on the GPU through WebGL and WebGPU is practical" -> OK
  Evidence: Abstract: WebGL 2.0, interactive ("changing parameter values on the fly"), in a browser. No "WebGPU" in its body (index), so the WebGPU half must come from Baggaley2026, which the sentence order allows.

## Schive2018
- Metadata: OK (2018MNRAS.481.4815S; MNRAS 481, 4815-4840; doi 10.1093/mnras/sty2586; arXiv 1712.07070)
- Claim: GAMER is a GPU code for astrophysics -> OK
- Claim: "refine in fixed-size patches" -> OK
  Evidence: fulltext.tex: "performs refinement on patches of fixed size (e.g. 8^3 cells)", and "We assume a fixed patch size of 8^3 cells throughout this paper".

## Pekkila2017
- Metadata: OK (2017CoPhC.217...11P; Pekkilä, Väisälä, Käpylä M. J., Käpylä P. J., Anjum; CPC 217, 11-22; doi 10.1016/j.cpc.2017.03.011; arXiv 1707.08900). The "?" in the batch file is a parsing problem: the bib entry has full authors.
- Claim: "Astaroth (Pekkila2017)" -> OK
  Evidence: Abstract describes a sixth-order finite-difference GPU solver for compressible fluids. The SciX full-text index has "Astaroth" in the text. Pekkilä et al. 2017 is the usual citation for Astaroth's methods.

## BenitezLlambay2016
- Metadata: OK (2016ApJS..223...11B; Benítez-Llambay & Masset; ApJS 223, 11; doi 10.3847/0067-0049/223/1/11; arXiv 1602.02359)
- Claim: "FARGO3D" listed among GPU codes for astrophysics -> OK
  Evidence: title "FARGO3D: A New GPU-oriented MHD Code".

## Schneider2015
- Metadata: OK (2015ApJS..217...24S; ApJS 217, 24; doi 10.1088/0067-0049/217/2/24)
- Claim: "Cholla" among GPU codes -> OK
  Evidence: Abstract: "a new three-dimensional hydrodynamics code that harnesses the power of graphics processing units".

## Farris2014
- Metadata: OK (2014ApJ...783..134F; ApJ 783, 134; doi 10.1088/0004-637X/783/2/134)
- Claim: "circumbinary-disk studies, such as Farris2014, who remove gas on a local viscous timescale" -> OK
  Evidence: Sec. 2 (eq. rhosink): "(dΣ/dt)_sink,i = −Σ/t_vis,i ... This removes fluid from the vicinity of each BH at the rate expected according to the α-disk model."
- Claim: "α = 0.1, the value that Farris2014 chose" -> OK
  Evidence: "where α = 0.1 is chosen for this study". They also hold h/r = 0.1 fixed, so ν = α(h/r)²a²Ω = 10⁻³ a²Ω at r = a, which agrees with our conversion.

## Kappeli2014
- Metadata: OK (2014JCoPh.259..199K; Käppeli & Mishra; JCP 259, 199-219; doi 10.1016/j.jcp.2013.11.028)
- Claim: "Following Kappeli2014, what is limited and reconstructed is each neighbor's deviation from [hydrostatic] balance, δp = (p_n − p_c) + ½(ρ_c+ρ_n)(Φ_n − Φ_c)" -> PARTLY / UNVERIFIED for the exact form
  Evidence: Abstract: "Based on a novel local hydrostatic reconstruction ... requires no analytical or numerical integration". So the idea of reconstructing the deviation from a local hydrostatic state is theirs. Full text paywalled and not read. The index has hits for "isentropic", "isothermal" and "polytropic", which suggests the 2014 reconstruction is built on those equilibrium families. The specific form, a pressure difference plus a trapezoidal density-averaged potential term (a discrete hydrostatic balance for arbitrary entropy), is what Käppeli & Mishra 2016 describe: A&A 587, A94, doi 10.1051/0004-6361/201527815, "a discrete version of the hydrostatic balance ... local second-order hydrostatic equilibrium preserving pressure reconstruction". That full text was not read either.
  Fix: "Following \citet{Kappeli2014} and \citet{Kappeli2016}, what is limited and reconstructed is ..." (add 2016A&A...587A..94K), or read K&M 2014 Sec. 3 to confirm that eq. (wb) matches their discrete equilibrium before keeping the single citation.

## Guillochon2013
- Metadata: OK (2013ApJ...767...25G; ApJ 767, 25; doi 10.1088/0004-637X/767/1/25)
- Claim: "as in G09 and G13, the frame accelerates with the center of mass of the gas, so the star stays at the box center while the perturber's orbit is integrated alongside the gas" -> OK (minor caveat)
  Evidence: Sec. 2: "we perform our simulations in the rest-frame of the star". The method is "very similar to" Guillochon 2011. Neither G09 nor G13 describes how the black hole's orbit is integrated, but "as in" can be read as attaching to the frame only.
- Claim: "the rate falls faster than t^−5/3, as G13 found for encounters that leave a core" -> OK, with a caveat
  Evidence: Sec. 4: "When a significant fraction of the star's mass is removed in an encounter, n∞ steepens to values as large as ≃ −2.2. This behavior arises from the influence of the star's core". Caveat: for grazing encounters that lose little mass G13 find n∞ ≃ −5/3. Their γ=4/3 fit at β = 0.6 gives ΔM ≈ 1.4×10⁻⁴ M☉ and n∞ ≈ −1.72, whereas our β = 0.6 star loses 0.05 M☉. G13 reach 5% mass loss near β ≈ 0.9. The qualitative statement holds, but a reader comparing the same β will see a large mass-loss difference. Consider saying so, or wording it as "as G13 found for partial disruptions that remove a significant fraction of the mass and leave a core".
- Claim: "A polytrope of index 3 is evolved with gas of Γ = 5/3, as in G13" -> OK
  Evidence: Sec. 2: polytropic γ of 5/3 or 4/3; "the stars are evolved hydrodynamically according to a Γ = 5/3 equation of state".

## Turk2011
- Metadata: OK (2011ApJS..192....9T; ApJS 192, 9; doi 10.1088/0067-0049/192/1/9)
- Claim: "HDF5 file in the layout of Athena++, which yt reads directly (Turk2011)" -> OK as a software citation
  Evidence: Turk2011 is the yt paper. It predates Athena++ and does not mention Athena (no match in fulltext.tex). The Athena++ reader is a later yt frontend, so the citation is for yt, not for that capability.
- Claim: software list "yt (Turk2011)" -> OK

## Wang2010
- Metadata: OK (2010NewA...15..581W; New Astron. 15, 581-589; doi 10.1016/j.newast.2009.10.002)
- Claim: "the adaptive solver of Wang2010" among GPU codes -> OK
  Evidence: Abstract/Sec. 1: "compressible inviscid fluid solvers with block-structured adaptive mesh refinement on Graphics Processing Units".
- Claim: "GAMER and Wang2010 ... both refine in fixed-size patches or blocks, as Shaderous does" -> WRONG for Wang2010
  Evidence: Sec. 3: Berger & Colella AMR in Enzo; "Since the grids are dynamically created, arbitrary grid dimensions can arise ... the key issue ... is to let it work for arbitrary grid dimensions".
  Fix: "Among GPU codes for astrophysics, GAMER \citep{Schive2010, Schive2018} is the closest in structure: it refines in fixed-size patches ($8^3$ cells), as Shaderous does; the adaptive GPU solver of \citet{Wang2010} instead runs on Enzo's Berger-Colella grids of arbitrary size."
- Claim: "the GPU codes that follow them (Schive2010, Wang2010) store the fluxes the fine side computes on a coarse face and correct the coarse cell afterward" -> UNVERIFIED for Wang2010
  Evidence: Wang2010 never mentions flux correction or refluxing (searched for flux/correct/conserv). They run the hydro solver on the GPU and leave the AMR hierarchy to Enzo on the CPU. Enzo does flux correction, but this paper does not say so.
  Fix: cite only Schive2010 here ("and the GPU code GAMER that follows them \citep{Schive2010}"), or write "the GPU codes that follow them \citep{Schive2010}, and the Enzo framework on which \citet{Wang2010} build".

## Federrath2010
- Metadata: OK (2010ApJ...713..269F; ApJ 713, 269-290; doi 10.1088/0004-637X/713/1/269)
- Claim: "Sinks in grid codes go back to ... Federrath2010" -> OK
  Evidence: title and abstract: implementation of sink particles in the AMR code FLASH. (Bate1995 in the same sentence is an SPH paper. It belongs to another batch, but "sinks in grid codes" does not fit it.)

## Schive2010
- Metadata: OK (2010ApJS..186..457S; ApJS 186, 457-484; doi 10.1088/0067-0049/186/2/457)
- Claim: GAMER is a GPU code for astrophysics -> OK
- Claim: "refine in fixed-size patches" -> OK
  Evidence: Sec. 2.1: "a grid patch is defined to have a fixed number of grid cells in each spatial direction ... the optimized size of a single patch is set to 8^3".
- Claim: "store the fluxes that the fine side computes on a coarse face and correct the coarse cell afterward" -> OK
  Evidence: Sec. 2.1: "the flux correction operation (BC89) is applied ... store the difference between the coarse-grid flux and the fine-grid flux across the coarse-fine boundary". Sec. 3.1 describes the closing step that copies fine-side fluxes to the coarse neighbor.

## Springel2010
- Metadata: OK (2010MNRAS.401..791S; MNRAS 401, 791-851; doi 10.1111/j.1365-2966.2009.15715.x)
- Claim: "work on a cell is the mass flux through each face times the difference of the potential across it, as in Springel2010" -> OK
  Evidence: Sec. 5.4, eq. 96: "ΔE_i^grav = −Δt m_i w_i∇Φ_i − ½ Σ_j Δm_ij(Φ_i − Φ_j)". The first term vanishes on a fixed mesh, and they note it "is also applicable in ordinary Eulerian codes". (In practice Springel uses the force form, eq. 94, but eq. 96 is in the paper.)

## Kestener2010
- Metadata: OK (2010LNCS.6082..281K; Kestener, Château, Teyssier; LNCS 6082, 281-288; doi 10.1007/978-3-642-13136-3_29). The "?" in the batch file is a parsing problem: the bib entry has authors. The booktitle could be the volume title "Algorithms and Architectures for Parallel Processing" (ICA3PP 2010), with series LNCS, but the current entry is acceptable.
- Claim: "the Euler solver of Kestener2010" among GPU codes for astrophysics -> OK
  Evidence: Abstract: "solving the 2D compressible Euler equations on modern GPU with high-resolution methods ... applications in computational astrophysics".

## Guillochon2009
- Metadata: OK (2009ApJ...705..844G; ApJ 705, 844-853; doi 10.1088/0004-637X/705/1/844)
- Claim: "frame accelerates with the center of mass of the gas, so the star stays at the box center" -> OK
  Evidence: Sec. 2: "The center of the computational domain is fixed to the star's center of mass ... We subtract out the gravitational force the black hole exerts on the star's center of mass from every grid cell ... ensures the star remains centered". The orbit integration is not described (see Guillochon2013).

## Cada2009
- Metadata: OK (2009JCoPh.228.4118C; Čada & Torrilhon; JCP 228, 4118-4145; doi 10.1016/j.jcp.2009.02.020)
- Claim: "Following the idea of Cada2009, we apply a limiter only where a quantity changes by more than 1 percent of its natural scale" -> OK at the level of "the idea" (full text not read)
  Evidence: Abstract: third-order compact limiters that handle "local extrema" without clipping. The index has body hits for "asymptotic region", "radius" and "smooth extrema", which fits their switch to the unlimited reconstruction where the variation is small relative to a radius scale. The full text is paywalled, so the switch's exact definition was not read. The 1% threshold and the choice of scale variables are ours.

## Thornber2008
- Metadata: OK (2008JCoPh.227.4873T; JCP 227, 4873-4894; doi 10.1016/j.jcp.2008.01.036)
- Claim: "low-Mach variant follows Thornber2008: velocity jump scaled by the local Mach number" -> OK as source of the method; exact formula UNVERIFIED
  Evidence: Abstract: "a simple modification of the variable reconstruction process ... minimum dissipation of low Mach number features". The index has hits for "velocity jump" and "HLLC". Full text paywalled. That Thornber's z = min(1, max(M_L, M_R)) matches our equation without z_min could not be checked. The z_min floor is ours and the sentence does not attribute it to them.

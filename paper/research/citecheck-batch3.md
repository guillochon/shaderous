# Citation check, batch 3

Metadata: each bib DOI was looked up in SciX (ADS). Every DOI resolves to the matching bibcode, and the authors, year, title, journal, volume and first page agree with the bib entry unless a section below says otherwise. Full texts are saved in `paper/research/<key>/` where they could be read.

## MacFadyen2008
- Metadata: OK (2008ApJ...672...83M)
- Claim: "a binary accreting from a viscous circumbinary disk [Artymowicz1994, MacFadyen2008]" -> OK
  Evidence: Sec. 2 (fulltext.tex): "The viscosity is calculated based on the $\alpha$-disk prescription ... with $\alpha=0.01$." The paper simulates a binary inside a circumbinary disk.

## VazquezSemadeni2007
- Metadata: OK (2007ApJ...657..870V)
- Claim: "the cooling function of Koyama2002 with the correction of VazquezSemadeni2007" -> OK
  Evidence: Sec. 2 footnote: "eq. (4) in KI02 contains two typographical errors. The form used here incorporates the necessary corrections". (They credit H. Koyama for the corrections; they are published in this paper.)

## deValBorro2006
- Metadata: OK (2006MNRAS.370..529D; the bib lists all 23 authors)
- Claim: "a damping zone at the outer edge, of the kind deValBorro2006 used to compare codes for disk-planet interaction" -> OK
  Evidence: Sec. 3 (boundaries): "solid boundary conditions with wave killing zones next to the boundaries"; the state relaxes toward its initial value over the orbital period with a parabolic ramp R(r) in [2.1a, 2.5a].

## Hagen2006
- Metadata: OK (2006LNCS.3994..220H). Minor: Crossref gives the volume title as "Computational Science – ICCS 2006", in the series Lecture Notes in Computer Science, vol. 3994. The bib puts the series name in `booktitle`. That is acceptable, but `booktitle = {Computational Science -- ICCS 2006}` with `series = {LNCS}` would be more exact.
- Claim: "the compressible Euler equations with high-resolution finite-volume schemes in two and three dimensions" -> OK
  Evidence: abstract (SciX): "simulate the dynamics of ideal gases in two and three spatial dimensions ... using state-of-the-art high-resolution finite-volume schemes."
- Claim: "solvers written as fragment programs that read and wrote textures ... Hagen2006 solved the Euler equations on uniform grids in fragment programs" -> UNVERIFIED
  Evidence: the full text is not on arXiv. The Springer PDF is paywalled and the author copy (folk.uio.no) is gone. The abstract says "commodity graphics cards (GPUs)" but does not mention fragment programs, textures or uniform grids.

## Springel2005
- Metadata: OK (2005MNRAS.364.1105S)
- Claim: "power-of-two time bins in the manner of particle codes such as GADGET-2" -> OK
  Evidence: Sec. 4 (time integration): "we discretise the timesteps in a power of 2 hierarchy, where all timesteps are a power of 2 subdivision of a global timestep."

## Krumholz2004
- Metadata: OK (2004ApJ...611..399K)
- Claim: "sinks in grid codes go back to ... Krumholz2004" -> OK
  Evidence: abstract: "We have removed that limitation by extending the sink particle capability to Eulerian hydrodynamics codes."

## Zingale2002
- Metadata: OK (2002ApJS..143..539Z)
- Claim: "As Zingale2002 do for stellar models mapped onto a grid, Shaderous settles every star of a problem before the first step" -> PARTLY
  Evidence: Zingale et al. do not settle or relax models before the run. They map 1D models (plane-parallel atmospheres, not whole stars) and restore discrete HSE in the initialization by "differencing the model into HSE" (Sec. 3), using HSE boundary conditions and modified-states PPM. Conclusions: "initialization methods can cause at most a transient while the simulation settles into hydrostatic equilibrium." They share our problem (a model mapped onto a grid is out of balance) but use a different remedy.
  Fix: "\citet{Zingale2002} met the same problem for stellar models mapped onto a grid and restored hydrostatic equilibrium in the discretized initial model; Shaderous instead settles every star of a problem before the first step."

## Teyssier2002
- Metadata: OK (2002A&A...385..337T)
- Claim: "RAMSES ... a large native program" (grid code for astrophysical gas dynamics) -> OK
  Evidence: abstract: "A new N-body and hydrodynamical code, called RAMSES ... based on Adaptive Mesh Refinement". ("Large native program" is our own characterization.)

## Koyama2002
- Metadata: OK (2002ApJ...564L..97K)
- Claim: "the interstellar cooling function of Koyama2002", used for the thermal instability problem -> OK
  Evidence: Sec. 2, energy equation with $\Gamma$ and $\Lambda(T)$, "heating and cooling rate" (photoelectric heating, cosmic rays, and so on).

## Koyama2004
- Metadata: OK (2004ApJ...602L..25K)
- Claim: "simulations of this instability converge with resolution only if they include conduction and resolve the Field length with at least three cells" -> OK
  Evidence: abstract: "The results show convergence only when thermal conduction is included" and a cell size smaller than $\lambda_F/3$ is needed. Summary: "At least three cells are required to resolve ... the Field length".

## Fryxell2000
- Metadata: OK (2000ApJS..131..273F)
- Claim: "FLASH ... a large native program" -> OK (code paper)
- Claim: "the error indicator of Lohner1987 in the form that FLASH uses, including a filter that ignores ripples of about one percent of the local value" -> OK
  Evidence: Sec. 2 (publisher PDF): "The criteria provided as a default ... are based on the work described in Löhner (1987)"; "acts as a filter and prevents refinement in regions of small ripples. The constant ε is given a value of 0.01." (FLASH's default refine/derefine thresholds of 0.8/0.2 are given in Sec. 7.)

## Truelove1997
- Metadata: OK (1997ApJ...489L.179T)
- Claim: "a pressure floor keeps the Jeans length resolved by a chosen number of cells, at least the four of Truelove1997" -> OK
  Evidence: abstract: "artificial fragmentation can be avoided ... by ensuring ... the Jeans number, J ≡ Δx/λ_J, is kept below 0.25". (Truelove et al. refine the mesh to meet this condition rather than adding a pressure floor; our sentence only borrows the number.)

## Bate1995
- Metadata: OK (1995MNRAS.277..362B)
- Claim: "Sinks in grid codes go back to Bate1995, Krumholz2004 and Federrath2010" -> WRONG
  Evidence: Bate et al. 1995 is an SPH paper. Abstract: "A method for following fragmentation simulations further in time using smoothed particle hydrodynamics (SPH) is presented." Krumholz2004's own abstract says SPH codes had sink particles while Eulerian codes had only fixed sink cells, until Krumholz2004.
  Fix: "Sink particles were introduced in SPH by \citet{Bate1995} and brought to grid codes by \citet{Krumholz2004} and \citet{Federrath2010}; ..."

## Bryan1995
- Metadata: OK (1995CoPhC..89..149B)
- Claim: "evolve the internal energy alongside the total energy ... after the dual-energy formalism of Bryan1995" -> OK (attribution)
  Evidence: the full text (Elsevier, paywalled) was not read. The abstract mentions modifications to PPM needed for "pressure differences of up to fourteen orders of magnitude". Citing papers attribute it this way, e.g. Marcello & Tohline 2012 (2012ApJS..199...35M): "we adopt a dual energy formalism (Bryan et al. 1995)".

## Toro1994
- Metadata: OK (1994ShWav...4...25T)
- Claim: "the HLLC solver of Toro1994" -> OK
  Evidence: abstract: "The missing contact surface in the approximate Riemann solver of Harten, Lax, and van Leer is restored."

## Quirk1994
- Metadata: OK (1994IJNMF..18..555Q)
- Claim: "Next to a cell flagged by a pressure-jump shock detector, the flux blends from HLLC toward HLL, which suppresses the carbuncle instability (Quirk1994)" -> OK
  Evidence: read in ICASE Report 92-64 (the preprint of the journal paper; scanned pages rendered, research/quirk1994/), Sec. 4, p. 17: "Figure 9 shows how the carbuncle phenomena may be circumvented", using the HLLE solver in cells flagged near shocks. Quirk switches from Roe to HLLE rather than blending from HLLC, but the cited idea (a dissipative HLL-type flux near shocks cures the carbuncle) is his. The journal version itself was not read.

## Artymowicz1994
- Metadata: OK (1994ApJ...421..651A; no page range in bib, first page 651 correct)
- Claim: "a binary accreting from a viscous circumbinary disk" -> OK
  Evidence: abstract: "gravitational interaction of a generally eccentric binary star system with circumbinary and circumstellar gaseous disks ... governed by gas pressure and (turbulent) viscosity".

## Gresho1990
- Metadata: OK (1990IJNMF..11..621G)
- Claim: "a Gresho vortex (Gresho1990)" -> OK (source of the test)
  Evidence: abstract (the full text, Wiley, was not read): numerical results for "a lid-driven cavity, vortex shedding, and inviscid vortex translation". This is the standard citation for the Gresho vortex.

## Berger1989
- Metadata: OK (1989JCoPh..82...64B)
- Claim: "follow Berger1989 in correcting the coarse side of a refinement boundary with fluxes stored from the fine side" and "store the fluxes that the fine side computes on a coarse face and correct the coarse cell afterward" -> OK (source of the method)
  Evidence: the full text (Elsevier) was not read. The abstract covers AMR for conservation laws and discontinuities at mesh boundaries. Stone et al. 2020 (Athena++), Sec. on flux correction, attributes this method to Berger & Colella 1989.

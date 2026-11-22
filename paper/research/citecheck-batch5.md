# Citation check, batch 5

Metadata checked against SciX/ADS records (and Crossref/arXiv where noted). Full texts saved in `paper/research/<key-lowercase>/`.

## Raskin2009
- Metadata: OK (2009MNRAS.399L.156R; Raskin, Timmes, Scannapieco, Diehl, Fryer; MNRAS 399, L156-L159; doi 10.1111/j.1745-3933.2009.00743.x)
- Claim: "two white dwarfs that collide on a parabolic orbit (Rosswog2009, Raskin2009)" -> PARTLY
  Evidence: Sec. 2: velocities from a 2-body solver "for a given impact parameter and an initial relative velocity of 10 km s^-1" (globular-cluster virial speed). So the orbits are slightly hyperbolic, near-parabolic; they ran head-on and offset collisions of two 0.6 Msun dwarfs.
  Fix: "two white dwarfs that collide on a nearly parabolic orbit \citep{Rosswog2009, Raskin2009}" (or keep "parabolic" and cite only Rosswog2009 if that paper used exactly parabolic orbits; Rosswog2009 is not in this batch).

## Plewa1999
- Metadata: OK, small gap: pages are 179-191 (1999A&A...342..179P); the bib gives only 179. No journal DOI exists (fine).
- Claim: "Scaling the face fractions so that they sum to one, as in the consistent multi-fluid advection of Plewa1999" -> OK
  Evidence: Sec. 3, Eqs. (phi)-(xscale): "one simply has to scale the original mass fractions X_i,n according to" phi = 1/sum X, applied to the interface mass fractions from the Riemann problem.

## Timmes2000
- Metadata: OK (2000ApJS..129..377T; ApJS 129, 377-398; doi 10.1086/313407)
- Claim: "the 13-isotope alpha-chain of Timmes2000" -> OK
  Evidence: Abstract and Sec. 1: compares a 7-isotope network "with a standard 13 isotope alpha-chain reaction network" (4He ... 56Ni listed). The paper describes and uses this network (aprox13); it does not claim to have invented the alpha-chain, so "the 13-isotope alpha-chain of" is the conventional attribution.

## TimmesSwesty2000
- Metadata: OK (2000ApJS..126..501T; ApJS 126, 501-516; doi 10.1086/313304)
- Claim 1: "electrons and positrons at any degeneracy and temperature, in the manner of TimmesSwesty2000" -> OK
  Evidence: Sec. 1: the EOS must hold where electrons and positrons move "arbitrarily close to the causal limits and an arbitrary degree of degeneracy"; table of Helmholtz free energy for the e-/e+ plasma.
- Claim 2: "The ions are an ideal gas with the Coulomb energy of a one-component plasma, as in the equation of state of TimmesSwesty2000, and radiation is added" -> PARTLY
  Evidence: Sec. 3 (publisher PDF): photon and ion contributions are "assumed in this paper to be blackbody radiation and ideal gas, respectively". The word "Coulomb" does not occur anywhere in the paper (full text searched). Coulomb corrections exist in Timmes's public Helmholtz code, not in this paper.
  Fix: "The ions are an ideal gas and radiation is added, as in the equation of state of \citet{TimmesSwesty2000}, and we add the Coulomb energy of a one-component plasma \citep[e.g.][]{<OCP source, e.g. Yakovlev & Shalybkov 1989 or Chabrier & Potekhin 1998>}."

## SmithClark2023
- Metadata: WRONG: first author is published as "Smith, Alexander I." (Crossref for 10.3847/1538-4357/acbaff and ADS 2023ApJ...947...65S both give Smith, A. I.; Johnson, E. T.; Chen, Z.; Eiden, K.; Willcox, D. E.; Boyd, B.; Cao, L.; DeGrendele, C. J.; Zingale, M.). The bib's "{Smith Clark}, Alexander" renders as "Smith Clark et al. 2023", which does not match the published record. If the author's later name is used deliberately, keep it knowingly; otherwise use `{Smith}, Alexander I.` (and the key text becomes Smith et al. 2023). Title, ApJ 947, 65, DOI OK.
- Claim: "pynucastro builds the network from the rates of REACLIB with the screening that pynucastro applies" -> OK
  Evidence: Sec. 2: "ReacLibRate for rates from the REACLIB library"; screening used in the examples (`do_screening=True`).
- Claim (software list): pynucastro -> OK

## Ross2011
- Metadata: OK (arXiv 1011.0686; Ross, Gordon, Bagnell; AISTATS 2011 per arXiv comment). Pages (PMLR 15:627-635) could be added; not checked against PMLR.
- Claim: "30 rounds of the DAgger method of Ross2011" -> OK (method source)
  Evidence: Sec. 1: "present the DAgger (Dataset Aggregation) method".

## Trac2003
- Metadata: OK (2003PASP..115..303T; PASP 115, 303-321; doi 10.1086/367747)
- Claim: "the introduction of Trac2003" -> OK (title: "A Primer on Eulerian Computational Fluid Dynamics for Astrophysics").

## LeVeque2002
- Metadata: OK (Crossref 10.1017/CBO9780511791253: LeVeque, Randall J., Cambridge University Press, 2002)
- Claim: "the texts of LeVeque2002" -> OK

## Livne1990
- Metadata: OK (1990ApJ...354L..53L; ApJL 354, L53; doi 10.1086/185721)
- Claim: "double detonation of a white dwarf under a thin shell of helium (Livne1990, Fink2010)" -> OK
  Evidence: Abstract: exploding helium shells send an ingoing shock "strong enough to ignite the carbon at the center into an outgoing detonation wave".

## Fink2010
- Metadata: OK (2010A&A...514A..53F; Fink, Ropke, Hillebrandt, Seitenzahl, Sim, Kromer; A&A 514, A53; doi 10.1051/0004-6361/200913892)
- Claim: same sentence -> OK
  Evidence: Abstract: core detonations triggered for cores of 0.810-1.385 Msun with minimum helium shells of 0.126 down to 0.0035 Msun.

## Shen2018
- Metadata: OK (2018ApJ...854...52S; ApJ 854, 52; doi 10.3847/1538-4357/aaa8de)
- Claim: "detonation of a bare carbon-oxygen dwarf of the same mass (1 Msun) makes 0.58 Msun of 56Ni" -> OK
  Evidence: Sec. 3.1.2: "For a WD mass of 0.9 (1.0) Msun, our post-processed model yields a 56Ni mass of 0.30 (0.58) Msun" (50/50 C/O, zero metallicity).

## GuillochonMcCourt2017
- Metadata: OK (2017ApJ...834L..19G; ApJL 834, L19; doi 10.3847/2041-8213/834/2/L19)
- Claim 1: "lines colored by the magnetic pressure on a logarithmic scale as GuillochonMcCourt2017 colored theirs" -> PARTLY
  Evidence: Fig. 2 caption: "the magnetic field lines colored by their magnetic pressure B^2/8pi"; Fig. 3 uses the same scale. The text never says the scale is logarithmic (may be visible only on the colorbar, not checked).
  Fix: drop "on a logarithmic scale" from the attribution, e.g. "colored by the magnetic pressure, as \citet{GuillochonMcCourt2017} colored theirs, on a logarithmic scale" -> better: "colored by the magnetic pressure (on a logarithmic scale), as in \citet{GuillochonMcCourt2017}" only if the colorbar confirms; otherwise "colored by the magnetic pressure as \citet{GuillochonMcCourt2017} colored theirs, here on a logarithmic scale".
- Claim 2: "set the field as GM17 did: a dipole, or a random field with a Kolmogorov spectrum, whose strength follows the pressure of the star so that beta is about the same throughout it" -> PARTLY
  Evidence: Sec. 2: "a magnetic dipole ... with a strength defined by beta_m = 8 pi P/B^2 (evaluated at the center of the star), and ii) a Gaussian-random field with a power spectrum appropriate for a Kolmogorov cascade". Nothing says the field strength follows the local pressure or that beta is uniform; for the dipole, beta is set only at the center.
  Fix: "The magnetized disruptions use the field geometries of \citet{GuillochonMcCourt2017}: a dipole, or a random field with a Kolmogorov spectrum. Unlike theirs, whose strength was set by $\beta$ at the center of the star, our field strength follows the pressure of the star so that $\beta$ is about the same throughout it." (Adjust if Shaderous normalizes differently.)
  Note: the preceding sentence's "the 1.85 that they found" does not appear in GM17 (no "1.85" in its text); make sure "they" there refers to another paper (likely Guillochon & Ramirez-Ruiz 2013).

## Kushnir2013
- Metadata: OK (2013ApJ...778L..37K; Kushnir, Katz, Dong, Livne, Fernandez; ApJL 778, L37; doi 10.1088/2041-8205/778/2/L37)
- Claim 1: "Following Kushnir2013, limit the burning energy rate to f_b e/t_s, t_s = dx/c_s" -> OK
  Evidence: Sec. 2.1: "limiting the energy injection rate from burning to f epsilon/t_s, where epsilon is the internal energy of the cell and f << 1", t_s = dx/c_s.
- Claim 2: "head-on collision of two 0.64 Msun dwarfs detonated on contact on 940 km cells and made 0.45 Msun of 56Ni, as collisions of such dwarfs do in simulations that resolve them (Raskin2010, Kushnir2013)" -> PARTLY
  Evidence: Kushnir2013 Sec. 2: collisions of dwarfs <~0.7 Msun "ignite at the shock region ... while higher mass CO-WDs ignite at the contact region"; Sec. 3: a "premature ignition occurs at the contact region" in low-resolution runs without the limiter. Fig. 3 caption: 0.64-0.64 gives 0.41 Msun 56Ni. Raskin2010 Table 3: 0.64x2, b=0: 0.51 Msun; Sec. 3.1: detonations "on either side of the shocked region". The yield (0.45) lies between the two, but neither paper has ignition at contact for this pair; Kushnir calls contact ignition a resolution artifact.
  Fix: "... the head-on collision of two white dwarfs of $0.64\,\Msun$ detonated on cells of 940~km and made $0.45\,\Msun$ of $^{56}$Ni, between the 0.41 and $0.51\,\Msun$ of simulations that resolve such collisions \citep{Kushnir2013, Raskin2010}, although on our cells it ignited at the contact rather than in the shocked region as in theirs." (If the run did not ignite at contact, simply drop "on contact".)

## Lee2013
- Metadata: OK (2013JCoPh.243..269L; JCP 243, 269-292; doi 10.1016/j.jcp.2013.02.049)
- Claim: "the upwind selection of Lee2013" -> OK
  Evidence: Sec. 2.6.2 "Upwind Biased Averaging in MEC: upwind-MEC": switches from the sign of the corner velocity choose the upwind face electric fields.

## Hawley2011
- Metadata: OK (2011ApJ...738...84H; ApJ 738, 84; doi 10.1088/0004-637X/738/1/84)
- Claim 1: "quality factors Q_z and Q_phi of Hawley2011, the number of cells across the fastest-growing wavelength" -> OK
  Evidence: Conclusions: Q_z "the number of vertical cells across a wavelength of the fastest-growing poloidal field MRI mode"; Q_phi likewise for the toroidal mode. (Q_z itself goes back to Sano et al. 2004; Hawley2011 adds Q_phi.)
- Claim 2: "below the values of about 8 and 20 above which Hawley2011 found that the turbulence saturates as it does at higher resolution" -> PARTLY
  Evidence: Conclusions: "Whereas Sano (2004) found that a minimum Q_z of 6-8 is required ... linear growth ... If ... Q_phi >~ 20, then Q_z >~ 10 is necessary"; Sec. 4: convergence "does not begin until Q_z >~ 10-15" for Q_phi ~ 10. The 6-8 is Sano's linear-growth criterion, not Hawley's nonlinear one. Our values (4.2, 15) are below either way.
  Fix: "... $Q_z = 4.2$ and $Q_\phi = 15$, below the $Q_z \gtrsim 10$ with $Q_\phi \gtrsim 20$ that \citet{Hawley2011} found necessary for the turbulence to behave as it does at higher resolution."

## Raskin2010
- Metadata: OK (2010ApJ...724..111R; Raskin, Scannapieco, Rockefeller, Fryer, Diehl, Timmes; ApJ 724, 111-125; doi 10.1088/0004-637X/724/1/111)
- Claim: see Kushnir2013 claim 2 -> PARTLY (same sentence)
  Evidence: Table 3 row 1: 0.64+0.64, b=0, 56Ni 0.51 Msun; Sec. 3.1: "The detonations on either side of the shocked region are unique to the 0.64 Msun x2 mass pairing". SPH run, not a cell-resolved one.
  Fix: as for Kushnir2013.

## Cyburt2010
- Metadata: OK (2010ApJS..189..240C; 13 authors as in bib; ApJS 189, 240-252; doi 10.1088/0067-0049/189/1/240)
- Claim: "the rates of REACLIB (Cyburt2010)" -> OK (source of the JINA REACLIB database; title).

## LeeDeane2009
- Metadata: OK (2009JCoPh.228..952L; JCP 228, 952-975; doi 10.1016/j.jcp.2008.08.026)
- Claim: "the edge estimates of LeeDeane2009" -> OK
  Evidence: Abstract: "present a new modified electric field construction (MEC)"; Lee2013 Sec. 2.6.2 calls upwind-MEC an enhancement of "the previous MEC scheme [LeeDeane2009]". Full text (Elsevier) not read.

## StoneGardiner2009
- Metadata: OK (2009NewA...14..139S; New Astron. 14, 139-148; doi 10.1016/j.newast.2008.06.003)
- Claim 1: "K-Athena, Idefix and AthenaK follow the unsplit integrators of Athena (GardinerStone2005, Stone2008, StoneGardiner2009)" -> OK as attribution of the integrator (what those codes follow is a claim about other papers, not checked here).
- Claim 2: "The integrator is the van Leer predictor-corrector of StoneGardiner2009, as in Athena++" -> OK (abstract level)
  Evidence: Abstract: "a simple, dimensionally-unsplit integration method with ... constrained transport", with no characteristic tracing. SciX full-text index matches "van Leer", "predictor" and "MUSCL-Hancock" in the body, but returned no snippet text; publisher full text not downloadable. The integrator is widely known as the VL (van Leer) integrator of this paper.

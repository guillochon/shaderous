# Citation check, batch 8

Metadata was checked against SciX records (by DOI or bibcode) and Crossref. Full texts were saved under `paper/research/<key>/` for tomida2013, masunaga2000 (publisher PDF; its text layer is garbled, so pages 4-5 were read as images), larson1969 (ADS scan), skinner2013, skinner2019, he2024 and eggleton1983. he2026 was already there. Nature, Icarus, JCP and JQSRT papers that are not on arXiv could not be read: SciX full text, Unpaywall and the authors' sites failed for them.

## Ida1997
- Metadata: OK (1997Natur.389..353I; Nature 389, 353-357, doi 10.1038/38669)
- Claim: "the disk's mass and angular momentum give the mass of the moon it would gather into" -> UNVERIFIED
  Evidence: The Nature full text is paywalled and has no SciX abstract or indexed body. Canup (2004, Icarus, eq. 1; paper/research/canup2004) gives "M_M = 1.9 L_D/sqrt(G M a_R) - 1.1 M_D - 1.9 M_esc ... (Ida et al., 1997)". That is a citing paper, not the source, but it is consistent with our claim.

## Kokubo2000
- Metadata: OK (2000Icar..148..419K; Icarus 148, 419-436, doi 10.1006/icar.2000.6496)
- Claim: "disk mass and angular momentum give the moon mass" -> OK (abstract only)
  Evidence: Abstract: "The efficiency of incorporation of disk material into a moon is 10-55%, which increases with the initial specific angular momentum of the disk." The full text could not be read. The explicit mass formula is Ida et al.'s.

## Tomida2013
- Metadata: OK (2013ApJ...763....6T; ApJ 763, 6; doi 10.1088/0004-637X/763/1/6; arXiv 1206.3567)
- Claim: "an ideal mixture of H2, H, H+, He, He+, He2+ and electrons in chemical equilibrium" -> OK
  Evidence: Appendix A: "Only seven major species (H2, H, H+, He, He+, He2+ and e-) are considered", and "local thermodynamic and chemical equilibrium (except for the ortho/para ratio...)".
- Claim: "described by one free energy F per unit mass" -> PARTLY
  Evidence: Tomida builds the EOS from partition functions and a grand potential: "Omega = sum_i Omega_i", n_i from dOmega/dmu_i. A Helmholtz free energy per unit mass does not appear in the paper. The two formulations are equivalent, but the wording reads as if F were Tomida's.
  Fix: "We follow the model of \citet{Tomida2013}, an ideal mixture of ... in chemical equilibrium, which we describe by one free energy $F$ per unit mass."
- Claim: "Tomida's statistical weights weight H2 4.6 times more [than textbook], moving its dissociation about 6 percent hotter" -> OK
  Evidence: Appendix A, H2: "Z_spin,H2 = 4", "Z_elec,H2 = 2", Z_rot = (Z_even)^{1/4}[3 Z_odd exp(theta_rot/T)]^{3/4}. At high T this is 8*3^{3/4} z ~ 18.2 z, against the textbook (Z_even + 3 Z_odd) ~ 4z, a ratio of 4.56. The H weights are 4 in both. The 6 percent shift is our own derivation, not Tomida's, and it is consistent: kT ln4.6/chi_dis ~ 0.06 at 2000 K.

## Masunaga2000
- Metadata: OK (2000ApJ...531..350M; ApJ 531, 350-365, doi 10.1086/308439)
- Claim: "the adiabat of gas opaque at 1e-13 g/cc and 10 K reaches 2060 K at 1e-8 g/cc, on the track of Masunaga2000" -> WRONG
  Evidence: Sec. 3.1.2 says the evolution becomes "completely adiabatic when tau ~ 1 at rho_c ~ 10^-12 g cm^-3". Sec. 3.1.3 says the second collapse starts "when ... reach rho_c ~ 10^-7 g cm^-3 and T_c ~ 2000 K". Fig. 2b (read off the plot) puts the central temperature at log T ~ 3.0 (about 1000-1200 K) at 10^-8 g/cc, and at about 2000 K only near 10^-7 g/cc. Their track is about a decade denser than ours at 2000 K.
  Fix: "The adiabat of gas that became opaque at $10^{-13}$~g~cm$^{-3}$ and 10~K reaches 2060~K at $10^{-8}$~g~cm$^{-3}$; in the calculation of \citet{Masunaga2000}, where the gas became adiabatic near $10^{-12}$~g~cm$^{-3}$, the centre reaches 2000~K near $10^{-7}$~g~cm$^{-3}$." Alternatively, drop "on the track of".
- Claim: "the second collapse of a protostellar first core (Larson1969, Masunaga2000)" -> OK
  Evidence: Sec. 3.1.3: "The second collapse takes place ... where dissociation of molecular hydrogen reduces the effective adiabatic exponent to ~1.1."

## Larson1969
- Metadata: OK (1969MNRAS.145..271L; MNRAS 145, 271-295, doi 10.1093/mnras/145.3.271)
- Claim: "[the H2 dip below 4/3] is what starts the second collapse of a protostar" -> OK
  Evidence: p. 284 (scan): "the central temperature reaches ... about 2000 K ... hydrogen molecules begin to dissociate. This reduces the ratio of specific heats gamma below the critical value 4/3". Note: Larson also says that on his core's track, ionization does not take gamma below 4/3 "because of the high central density". Our statement that "both dips are below 4/3" is about our table, not his track, so there is no conflict.
- Claim: "second collapse of a first core (problem source)" -> OK (same passage)

## Ivanova2013
- Metadata: OK (2013A&ARv..21...59I; A&ARv 21, 59, 19 authors, doi 10.1007/s00159-013-0059-2)
- Claim: generic citation for common-envelope evolution -> OK
  Evidence: This is the standard CE review (title). The setup parameters are ours, not taken from the review.

## Kwatra2009
- Metadata: OK (2009JCoPh.228.4146K; JCP 228, 4146-4161; Kwatra, Su, Gretarsson, Fedkiw)
- Claim: "The implicit step follows Kwatra2009" -> OK (abstract)
  Evidence: Abstract: the method is "based on the pressure evolution equation", with "a standard Poisson equation ... but has an identity term", and gives the incompressible Poisson equation "as the sound speed goes to infinity".
- Claim: "face pressure (rho_R p_L + rho_L p_R)/(rho_L + rho_R), as in Kwatra2009" -> UNVERIFIED
  Evidence: The full text could not be read: it is paywalled, the PhysBAM copy is gone (404) and Unpaywall has no OA copy. SciX's body index matches "density weighted" for this paper but returns no snippet. That is suggestive but not proof of the formula.

## Bell1989
- Metadata: OK (1989JCoPh..85..257B; JCP 85, 257-283)
- Claim: "becomes the Poisson equation for the pressure in a projection method (Bell1989) as c0 -> inf" -> OK
  Evidence: Abstract: a second-order projection method in which intermediate velocities "are then projected onto the space of divergence-free vector fields". The c0 -> inf limit is our own algebra. Kwatra's abstract states the same limit.

## Monaghan1994
- Metadata: OK (1994JCoPh.110..399M; JCP 110, 399-406)
- Claim: "c0 about ten times the fastest flow, as in SPH (Monaghan1994)" -> UNVERIFIED
  Evidence: The full text could not be read (paywalled, not indexed in SciX). Citing papers attribute "c0 >= 10 v_max", for density fluctuations of about 1%, to Monaghan (1994). They also say his EOS is the stiff Tait/Batchelor form p = B[(rho/rho0)^gamma - 1], not the linear p = c0^2(rho - rho0). Our sentence could be read as giving Monaghan the linear EOS.
  Fix (suggested): "... a constant chosen to be about ten times the fastest flow, as in smoothed particle hydrodynamics, which uses a stiffer equation of state of the same kind \citep{Monaghan1994}." Check the primary text before committing to the factor of ten.

## Levermore1984
- Metadata: OK (1984JQSRT..31..149L; JQSRT 31, 149-160)
- Claim: "the closure of Levermore1984: chi = (3+4f^2)/(5+2 sqrt(4-3f^2))" -> OK (via secondary source)
  Evidence: The JQSRT full text could not be read. Skinner & Ostriker (2013, Sec. 2.3) state: "Levermore further showed that if I is isotropic in some inertial frame ... chi(f) = (3+4f^2)/(5+2 sqrt(4-3f^2))". The tensor form matches their eq. for T.

## Skinner2013
- Metadata: OK (2013ApJS..206...21S; ApJS 206, 21; arXiv 1306.0010)
- Claim: "two-moment module of Athena" -> OK
  Evidence: Abstract: "a module for the Athena code that solves the gray equations of RHD, based on the first two moments ... We adopt the M1 closure".
- Claim: "usual remedy replaces c by a reduced speed c-hat everywhere" -> OK
  Evidence: Sec. 3: "the radiation time step is set by the (constant) reduced speed of light, c-hat". Sec. 3.2 "The Reduced Speed of Light Approximation".

## Skinner2019
- Metadata: OK (2019ApJS..241....7S; ApJS 241, 7; arXiv 1806.07390)
- Claim: "two-moment module of Fornax" -> OK
  Evidence: Sec. 1: Fornax solves "the comoving-frame, multi-group, two-moment ... transport equations ... and uses the M1 tensor closure".
- Claim: "dissipation scaled by min(1, 1/tau_f), tau_f the optical depth of a cell at the face" -> OK
  Evidence: Sec. 3.3, eqs. (hll)/(epsilon): "epsilon = min(1, 1/tau_cell)", where tau_cell is "an approximation to the optical depth of a cell in the direction normal to the face". Note: Fornax applies this to the energy flux of an HLL solver, not to a local Lax-Friedrichs flux. Our sentence does not say otherwise.

## SuOlson1996
- Metadata: OK (doi 10.1016/0022-4073(96)84524-9; JQSRT 56, 337-351; Su & Olson). The SciX record garbles the author as "Bingjing, S." and gives the bibcode 1996JQSRT..56..337B. Crossref has families "Bingjing", "Olson". Our bib entry ({Su}, Bingjing and {Olson}, Gordon L.) is correct, so no change is needed.
- Claim: "the non-equilibrium Marshak wave of SuOlson1996" -> OK
  Evidence: Title: "Benchmark results for the non-equilibrium Marshak diffusion problem". The full text could not be read; the citation is used only as the test's source.

## He2024
- Metadata: OK (2024MNRAS.531.1228H; MNRAS 531, 1228-1242; arXiv 2404.08247)
- Claim: "the two-stage implicit-explicit scheme of He2024, which would double the cost of the sweeps" -> OK
  Evidence: Sec. 3: "the transport and source terms appear symmetrically at each of the two stages" (IMEX PD-ARS). He et al. call it "only marginally more costly than an update like RK2-SSP". The doubling is our cost relative to our single-stage sweeps, which is consistent with two transport stages. It is not a claim made by He2024.

## He2026
- Metadata: OK (2026arXiv261121353H; arXiv:2611.21353; He, Krumholz, Wibking)
- Claim: "proposed a reduction factor that is also chosen cell by cell" -> OK
  Evidence: Sec. 4: the reduction factor phi "is set to a value in the range [phi_0, 1], chosen cell-by-cell to be as large as possible while keeping the scheme stable".

## Eggleton1983
- Metadata: OK (1983ApJ...268..368E; ApJ 268, 368-369, doi 10.1086/160960)
- Claim: "Roche-lobe filling as approximated by Eggleton1983" -> OK
  Evidence: Eq. (2): r_L = 0.49 q^{2/3}/(0.6 q^{2/3} + ln(1+q^{1/3})), "to better than 1% over the whole range".

## BergerGiuliani2021
- Metadata: OK (2021JCoPh.42809820B; JCP 428, 109820; arXiv 2005.05734)
- Claim: "state redistribution keeps the small cells stable" -> OK
  Evidence: Abstract: "state redistribution ... The idea is to stabilize the cut cells by temporarily merging them into larger, possibly overlapping neighborhoods".

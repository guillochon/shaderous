# Citation check, batch 7

Sources read: Crossref records for every DOI; Zenodo API records for the six Zenodo items; OSTI record 4208328; SciX abstracts; arXiv full texts saved in
`thurgood2018/`, `federrath2010b/`, `stewart2020/`, `howesnielson2013/`, `verniero2018/`; the existing `canup2004/fulltext.txt`; file listings of the ststewart GitHub repos and of the Zenodo pyrolite v0.2.0 archive.

## EswaranPope1988
- Metadata: OK (Crossref 10.1016/0045-7930(88)90013-8: Eswaran & Pope, Computers & Fluids 16(3) 257-278, 1988; SciX 1988CF.....16..257E)
- Claim: "we follow the forcing of EP88: Fourier modes of the force in a band of wavenumbers with amplitudes following Ornstein-Uhlenbeck processes" -> OK (attribution); the OU and band details were checked only through a secondary source
  Evidence: The EP88 abstract says "A spectral forcing scheme is developed to provide the means to obtain statistically stationary velocity fields". Federrath et al. (2010) Sec. 2.1 says the OU process "can be used to excite turbulent motions ... as explained in \citet{EswaranPope1988}". I could not read the EP88 full text (Elsevier, not on arXiv), so the OU and wavenumber-band details are not checked against EP88 itself.

## Federrath2010b
- Metadata: OK (Crossref 10.1051/0004-6361/200912437; A&A 512, A81, 2010; five authors match; subtitle "Solenoidal versus compressive turbulence forcing" is correct)
- Claim: "the projection of Federrath2010b sets the share of the force that is free of divergence" -> OK
  Evidence: Sec. 2.1, eq. for P^zeta_ij: "The projection operator serves to construct a purely solenoidal force field by setting zeta=1. For zeta=0, a purely compressive force field is obtained."

## HowesNielson2013
- Metadata: OK (Crossref 10.1063/1.4812805; Phys. Plasmas 20, 072302, 2013)
- Claim: "collide counter-propagating Alfven waves along a guide field as periodic waves (HN2013)" -> OK
  Evidence: arXiv:1306.1455. The abstract covers "the nonlinear interaction between counterpropagating Alfvén waves", and the solution is set in a "triply periodic domain of dimensions L_x × L_y × L_z".

## Verniero2018
- Metadata: OK (Crossref 10.1017/S0022377817001003; J. Plasma Phys. 84, 905840103, 2018; Verniero, Howes, Klein)
- Claim: "... and as packets (Verniero2018)" -> OK
  Evidence: Abstract: "For the more realistic case of the collision between two initially separated Alfvén wavepackets, we use a nonlinear gyrokinetic simulation". (They use gyrokinetics, not MHD; the paper is cited as the source of the problem, which is fine.)

## Furth1963
- Metadata: OK (Crossref 10.1063/1.1706761; Phys. Fluids 6(4) 459-484, 1963)
- Claim: "with resistivity, a Harris current sheet tears into islands (Furth1963)" -> OK
  Evidence: Abstract: "a long-wave 'tearing' mode, corresponding to breakup of the layer along current-flow lines". FKR treat a general plane current layer, not specifically the Harris profile. That is acceptable for citing the tearing instability.

## FinnKaw1977
- Metadata: OK (Crossref 10.1063/1.861709; Phys. Fluids 20(1) 72-78, 1977)
- Claim: "two islands coalesce (FinnKaw1977)" -> OK
  Evidence: Abstract: "An instability is found which describes the tendency toward coalescence of parallel currents in the neighboring islands." (Their analysis is ideal MHD, with resistivity expected to dominate later; it fits the "with resistivity" sentence well enough.)

## Thurgood2018
- Metadata: OK (Crossref 10.1063/1.5035489; Phys. Plasmas 25, 072105, 2018; arXiv:1806.08157)
- Claim: "closed boundaries of Thurgood2018 that conserve the flux through them, holding field lines that cross the walls" -> OK
  Evidence: Appendix "Boundary conditions and reflectivity": "The normal component of the field is held fixed (line-tied) through the boundary." Sec. 2: "the outer boundaries are taken to be closed". Holding B_n fixed fixes the flux through each face.
- Claim: "a stressed X-point collapses between line-tied walls (Thurgood2018)" -> OK
  Evidence: The abstract covers "2D, null point containing fields which are locally imbalanced ('null-collapse' or 'X-point collapse')", and the appendix gives the line-tied boundaries. Note: their walls sit far away (r = 20), so the evolution is effectively open over the period they study.

## ThompsonLauson1972
- Metadata: OK, with one discrepancy to note. The OSTI record in the bib's own url (osti.gov/biblio/4208328, report SC-RR-71-0714) gives publication date 1974-03-01. SciX (1972snl..rept..714T) and the impact literature (e.g. Canup 2004: "ANEOS (Thompson and Lauson, 1972)") use 1972. Stewart et al. (2020) cite the same report as 1974. Keeping 1972 is defensible; if you want the year to match the linked OSTI record, use 1974.
- Claim: "ANEOS, the analytic equation of state of ThompsonLauson1972" -> OK
  Evidence: OSTI title: "Improvements in the CHART D radiation-hydrodynamic code III: revised analytic equations of state". The standard ANEOS citation, used by Canup 2004 and Melosh 2007. (The first ANEOS report is Thompson 1970, SC-RR-70-28, "Code I". 1972/74 is the revised version.)

## Melosh2007
- Metadata: OK (Crossref 10.1111/j.1945-5100.2007.tb01009.x; MAPS 42(12) 2079-2098, 2007)
- Claim: "later changes to ANEOS by Melosh2007" -> OK
  Evidence: Abstract: "two shortcomings of ANEOS are corrected ... treats the vapor phase as a monatomic mixture of atoms, rather than molecular clusters ... assumes a Morse potential".

## ThompsonMANEOS2019
- Metadata: OK (Zenodo 3525030: Thompson, Lauson, Melosh, Collins, Stewart; "M-ANEOS"; v1.0; 2019-11-01)
- Claim: "pyaneos, the Python port of M-ANEOS" -> OK
  Evidence: Zenodo 3525030: "A FORTRAN77 program ... which extends the ANEOS computer code". Zenodo 20736272: "A self-contained Python port of the ANEOS / M-ANEOS equation-of-state code".

## Stewart2020
- Metadata: OK (Crossref 10.1063/12.0000946: nine authors match; AIP Conf. Proc. 2272, 080003, 2020; arXiv:1910.04687)
- Claim: "later changes to ANEOS by Stewart2020" -> OK
  Evidence: Abstract: "we present an updated version of the widely-used ANEOS model that includes a user-defined heat capacity limit in the thermal free energy term."

## StewartPyrolite2022
- Metadata: OK (Zenodo 6968539: Stewart; title "ststewart/aneos-pyrolite-2022: ANEOS-PYROLITE"; v0.2.0; 2022-08-05)
- Claim: "a published input deck for pyrolite (mantle)" -> OK
  Evidence: The Zenodo v0.2.0 archive contains ANEOS.INPUT and the NEW-SESAME-*.TXT tables. The record says "DOCUMENTATION FOR ANEOS MODEL FOR PYROLITE GLASS/MAGMA". Note: the GitHub repo now holds only LICENSE and README, so the deck is reachable only through Zenodo.
- Claim: "published tables (with Fe85Si15) whose sound speed our form matches to 3e-6 median, 2e-4 p90" -> OK as an identification of the tables. The numbers are our own measurement and are not a claim about the source.
  Evidence: The archive includes NEW-SESAME-STD.TXT and NEW-SESAME-EXT.TXT.

## StewartFe85Si15_2020
- Metadata: OK (Zenodo 3866550: Sarah T. Stewart; title matches; 2020-05-30)
- Claim: "published input deck for the iron alloy Fe85Si15 (core)" -> OK
  Evidence: github.com/ststewart/aneos-Fe85Si15-2020 holds ANEOS.INPUT and the NEW-SESAME-*.TXT tables. The record says "ANEOS IRON ALLOY (Fe85Si15) MODEL".
- Claim: tables used in the sound-speed comparison -> OK (see above).

## StewartWater2024
- Metadata: OK (Zenodo 14226694: Stewart; "ANEOS Equation of State Model for Water"; v0.3; 2024-11-27)
- Claim: "published input deck for water" -> OK
  Evidence: github.com/ststewart/aneos-water holds ANEOS.INPUT. The README/files label the model Water-ANEOS-SLVTv0.3G1.

## StewartForsterite2019
- Metadata: OK (Zenodo 3478631: nine creators; title matches; v1.0.0; 2019-10-05)
- Claim: "published input deck for forsterite, the rock of older calculations" -> OK
  Evidence: The repo holds ANEOS.INPUT. Stewart et al. (2020) Sec. "Revised ANEOS model for forsterite" says forsterite "is widely used as a single-component proxy for the silicate mantle of planets".

## StewartPyaneos2026
- Metadata: OK (Zenodo 20736272: Stewart; "pyaneos v1.0 — oracle-validated Python port of M-ANEOS"; v1.0; 2026-06-17)
- Claim: "pyaneos, the Python port of M-ANEOS" -> OK
  Evidence: Zenodo description: "A self-contained Python port of the ANEOS / M-ANEOS equation-of-state code ... bit-identical ... to the original Fortran".

## FritschCarlson1980
- Metadata: OK (Crossref 10.1137/0717021; SIAM J. Numer. Anal. 17(2) 238-246, 1980)
- Claim: "slopes limited as those of a monotone cubic (FritschCarlson1980)" -> OK
  Evidence: The title, "Monotone Piecewise Cubic Interpolation", identifies the method source. Full text not read (SIAM, not open), but none is needed for a method attribution.

## Dziewonski1981
- Metadata: OK (Crossref 10.1016/0031-9201(81)90046-7; PEPI 25(4) 297-356, 1981)
- Claim: "364 GPa and 13.1 g/cm^3 at the center of the Earth (PREM)" -> UNVERIFIED
  Evidence: The full text could not be read. It is an Elsevier paper with no open copy: SciX fulltext failed, SciX highlight returned no body snippets, and the IRIS PREM tables refused access. From memory, the PREM inner-core density polynomial gives 13.0885 g/cm^3 at r = 0, and Table II lists a central pressure of about 363.85 GPa. Both round to the quoted values, but I could not see them in the source.

## Canup2004
- Metadata: OK (Crossref 10.1016/j.icarus.2003.09.028; Icarus 168(2) 433-456, 2004)
- Claim: "Theia strikes the proto-Earth in the moon-forming impact (Canup2004)" / "The moon-forming impact follows Canup2004" -> OK
  Evidence: The paper covers "about 100 hydrodynamic simulations of potential Moon-forming impacts". It never uses the name "Theia" (0 hits), so the name is ours, which is harmless.
- Claim: "Both bodies have 30% of their mass in a core" (context, consistent) -> OK
  Evidence: Sec. 3.2: "We consider impactors and targets that contain 30% iron".
- Claim: "As Canup2004 did, the code sorts the gas after the impact by the Kepler orbit from each cell's position and velocity" -> OK (with nuance)
  Evidence: Sec. 3: "We thus also compute equivalent keplerian orbits for the bound particles using their instantaneous r and v vectors", counting particles "with periapses above the equatorial radius". Her primary sort was by equivalent circular orbit from L_z; the Kepler sort was her cross-check, and "these two methods ... converge to within less than a few percent".
- Claim: "for impactors of 0.11-0.13 of total mass at the mutual escape speed and angles near 45 deg, disks of 1.0-1.6 lunar masses (0.013-0.020 Mearth), 75-89% from the impactor, predicted moons 1.0-1.6 lunar masses" -> OK
  Evidence: In Table 1, the rows with gamma = 0.11 and 0.13 and v_imp/v_esc = 1.00 (b' = 0.723-0.762) give M_D/M_L 1.04-1.62, M_imp/M_D 0.75-0.89 and M_M/M_L 1.00-1.62. With M_L = 7.35e25 g = 0.0123 Mearth, the disk masses are 0.013-0.020 Mearth. Successful impacts span b' 0.67-0.76, "42 to 50 degrees".
- Claim: "2 to 10 percent iron in those disks" -> OK
  Evidence: The same Table 1 rows give M_Fe/M_D = 0.02-0.10 (whole disk). The caption defines successful disks as those that "contain < 10% iron by mass in orbit".

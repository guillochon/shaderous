# Citation check, batch 4

Metadata checked against SciX (ADS) records and Crossref for every DOI. Full texts were read where they could be had: Rosswog2009 (arXiv source), Evrard1988, Chandrasekhar1935 and Plummer1911 (ADS scans). Most of the JCP, CMAME, Annual Reviews and Math. Comp. papers are behind paywalls. For those, the abstract was read, and where SciX indexes the body, term-presence queries (`count 'bibcode:X AND body:"term"'`) were run. Elsevier bodies give no snippets, only whether a term is present.

## Evrard1988
- Metadata: OK (1988MNRAS.235..911E; Crossref title has no trailing period, which is harmless)
- Claim: "the collapse of Evrard1988" (test problem source, used three times) -> OK
  Evidence: Sec. 3.3, "Self-gravitating collapse of a cold gas sphere": "a static gas sphere of mass M_t and radius R, with a shallow density profile, rho(r) ∝ r^-1. The initial temperature is much smaller than the virial temperature." (p. 922). The energy numbers in our sentence are our own results, not Evrard's.

## Lohner1987
- Metadata: OK (1987CMAME..61..323L; author is Löhner, Rainald. The batch file shows "?" only because the `{\"o}` did not parse. Crossref gives issue 3 and pages 323-338)
- Claim: "the error indicator of Lohner1987 (in the form FLASH uses), including a filter that ignores ripples of about one percent" -> OK (method source)
  Evidence: Full text not readable (Elsevier). The abstract describes h-refinement on triangles. The indexed body contains "error indicator", "second derivatives", "noise" and "filter", which fits Löhner's normalized second-derivative indicator with a noise filter. The one percent value is credited to the FLASH form (Fryxell2000, another batch), not to Löhner. Löhner's own filter coefficient was not checked.

## Roe1986
- Metadata: OK (1986AnRFM..18..337R; Crossref 18(1) 337-365)
- Claim: "minmod limiter (Roe1986)" -> UNVERIFIED
  Evidence: Full text not readable (Annual Reviews PDF, no download). The abstract mentions "high-order variants of Godunov's method" but no limiter. SciX's body index for this record seems to be incomplete: "Godunov" and "Riemann" both return 0, so the 0 for "minmod" proves nothing. Citing Roe 1986 for minmod is a common convention, for example in limiter lists that follow Sweby 1984. The minmod function itself is older.
  Fix: if this is kept, check a library copy for the minmod discussion. Otherwise "minmod \citep[e.g.][]{Roe1986}" is safer.

## Woodward1984
- Metadata: OK (1984JCoPh..54..115W; 54(1) 115-173)
- Claim: "Mach 3 wind tunnel with a step of Emery1968 and Woodward1984" -> OK
  Evidence: The indexed body contains "wind tunnel", "Mach 3", "step" and "Emery" (snippets withheld by the publisher). This is the standard source of the test.

## Berger1984
- Metadata: OK (1984JCoPh..53..484B; 53(3) 484-512)
- Claim 1: "The adaptive solvers of Berger1984 and Berger1989 ... store the fluxes that the fine side computes on a coarse face and correct the coarse cell afterward" -> WRONG (for Berger1984)
  Evidence: Full text not readable (Elsevier). However, SciX indexes the body: "flux" and "fluxes" each return 0, and the phrase "not conservative at" returns 1. Control phrases behave correctly ("Lax-Wendroff" and "refined in time" return 1, "banana" returns 0). Berger and Oliger 1984 used finite differences with interpolated boundary values and no flux bookkeeping. Refluxing was introduced by Berger & Colella 1989, and main.tex line 80 already credits Berger1989 alone for it.
  Fix: "The adaptive solver of \citet{Berger1989}, and the GPU codes that follow it \citep{Schive2010, Wang2010}, store the fluxes that the fine side computes on a coarse face and correct the coarse cell afterward."
- Claim 2: "The subcycling of Berger1984 lets fine levels take more steps than coarse ones" -> OK
  Evidence: Abstract: "The grids with finer mesh width in space also have a smaller mesh width in time, making this a mesh refinement algorithm in time and space."

## vanLeer1979
- Metadata: OK (1979JCoPh..32..101V; 32(1) 101-136)
- Claim: "an unsplit MUSCL-Hancock method (vanLeer1979)" -> PARTLY
  Evidence: The 1979 paper is the source of MUSCL: second order through linear distributions, a Lagrangean step with remap, and "Higher dimensions are covered through time splitting" (abstract). The indexed body contains "MUSCL" but not "Hancock". The Hancock predictor form is credited to van Leer 1984, SIAM J. Sci. Stat. Comput. 5, 1-20, doi:10.1137/0905001 (Crossref checked). Codes such as AREPO and PLUTO cite it that way. That attribution comes from citing papers, and the 1984 text was not read.
  Fix: "The scheme is an unsplit MUSCL-Hancock method \citep{vanLeer1979, vanLeer1984}, in the form given by \citet{Toro2009}." Add the bib entry vanLeer1984 = van Leer, B. 1984, "On the Relation Between the Upwind-Differencing Schemes of Godunov, Engquist-Osher and Roe", SIAM J. Sci. Stat. Comput. 5(1), 1-20, doi:10.1137/0905001.

## Sod1978
- Metadata: OK (1978JCoPh..27....1S; 27(1) 1-31). Minor: "Review." at the start of the title comes from ADS. Crossref's title has no such prefix, and dropping it would be cleaner.
- Claim: "the shock tubes of Sod1978" -> OK (standard source of the Sod shock tube; the ADS keywords include "Shock Tubes")

## vanLeer1977
- Metadata: WRONG (minor): pages give only "276". Correct pages are 276-299 and the issue is 3 (Crossref, doi:10.1016/0021-9991(77)90095-X).
- Claim: "bounds of the monotonized central limiter (vanLeer1977)" -> OK (method source, by convention)
  Evidence: Full text not readable. The abstract says "Several monotonicity algorithms, designed to prevent numerical oscillations, are indicated", and the body index contains "monotonicity" and "Fromm". The name "monotonized central" is later and does not occur in the paper (body "monotonized" returns 0). The MC slope is the standard attribution to this paper, as in LeVeque's textbook.

## Brandt1977
- Metadata: OK (1977MaCom..31..333B; 31(138) 333-390)
- Claim: "geometric multigrid (Brandt1977)" -> OK
  Evidence: Abstract: "discretized on several grids ... of widely different mesh sizes ... solve the possibly nonlinear system of n discrete equations in O(n) operations".
- Claim: "the full approximation scheme of Brandt1977" -> OK (method source); full text UNVERIFIED
  Evidence: The AMS PDF was blocked for automated download, and SciX has no body for this record. The abstract mentions nonlinear systems, which is what FAS serves. Brandt 1977 is the standard origin of FAS, though Brandt expands the acronym as "Full Approximation Storage". The sentence's details (red-black Gauss-Seidel, 8-cell restriction, trilinear prolongation, warm start) describe our code, not Brandt.

## vanLeer1974
- Metadata: OK (1974JCoPh..14..361V; 14(4) 361-370; the original title does say "Conservation Difference Scheme" in part II, as ADS has it. Crossref normalizes it to "Conservative")
- Claim: "the vanLeer1974 limiter" -> OK
  Evidence: Abstract: "Fromm's second-order scheme ... is made monotonic through the inclusion of nonlinear feedback terms". This is the origin of the van Leer (harmonic) limiter.

## Emery1968
- Metadata: OK (1968JCoPh...2..306E; 2(3) 306-331)
- Claim: "Mach 3 wind tunnel with a step of Emery1968" -> OK
  Evidence: The abstract lists "a flat-faced step in supersonic flow", and the indexed body contains "Mach 3" and "step".

## Field1965
- Metadata: OK (1965ApJ...142..531F)
- Claim: "the thermal instability (Field1965)" -> OK (the defining paper; title "Thermal Instability")

## Sedov1959
- Metadata: OK (Academic Press, New York, 1959, the English translation; 1959sdmm.book.....S)
- Claim: "the blast of Taylor1950 and Sedov1959" -> OK (standard source of the Sedov-Taylor solution)

## Taylor1950
- Metadata: OK (1950RSPSA.201..159T; 201(1065) 159-174)
- Claim 1: "the blast of Taylor1950 and Sedov1959" -> OK (Part I is the theoretical similarity solution)
- Claim 2: "how Taylor1950 estimated the energy of an explosion from photographs of its fireball" -> PARTLY
  Evidence: The energy estimate from Mack's photographs is in Part II, not Part I. The Part II abstract (1950RSPSA.201..175T) says "Photographs by J. E. Mack of the first atomic explosion ... were measured ... used ... to estimate the energy E". Part I, which is the cited entry, is "I. Theoretical Discussion".
  Fix: add an entry Taylor1950b = Taylor, G. 1950, "The Formation of a Blast Wave by a Very Intense Explosion. II. The Atomic Explosion of 1945", Proc. R. Soc. Lond. A 201(1065), 175-186, doi:10.1098/rspa.1950.0050. Then write "... how \citet{Taylor1950b} estimated the energy of an explosion from photographs of its fireball" (or \citet{Taylor1950, Taylor1950b}).

## Plummer1911
- Metadata: OK (1911MNRAS..71..460P; 71(5) 460-470)
- Claim: "Plummer-softened gravity (Plummer1911)" -> OK
  Evidence: Eq. (12) gives the density law ∝ (1 + r^2)^(-5/2), the Plummer sphere whose potential is the softened point-mass potential. Plummer did not use it for softening; the citation names the profile, which is the usual convention.

## Jeans1902
- Metadata: OK (1902RSPTA.199....1J; 199, 1-53)
- Claim: "the Jeans instability (Jeans1902)" -> OK (the defining paper, "The Stability of a Spherical Nebula")

## Saha1920
- Metadata: OK (Crossref doi:10.1080/14786441008636148: Saha, M. N., "LIII. Ionization in the solar chromosphere", Phil. Mag. 40(238), 472-488, Oct 1920; not in SciX)
- Claim: "hydrogen ionizes in Saha equilibrium (Saha1920)" -> OK (the paper that introduced the Saha ionization equation). The 13.6 eV and the EOS details describe our code.

## Chandrasekhar1935
- Metadata: OK (1935MNRAS..95..207C; 95(3) 207-225)
- Claim: "pressure of fully degenerate electrons at any degree of relativity (Chandrasekhar1935)" -> OK
  Evidence: Sec. 1, p. 208: the aim is "to take the exact equation describing the degenerate state and treat the whole degenerate parts of a star", in place of the two limiting polytropes of paper I.

## Rosswog2009
- Metadata: OK (2009ApJ...705L.128R; ApJ 705(2) L128-L132, a Letter, so \apjl is right; arXiv 0907.3196)
- Claim: "two white dwarfs that collide on a parabolic orbit (Rosswog2009, Raskin2009)" -> OK
  Evidence: Sec. 2: "placed at a mutual distance of three times their combined unperturbed stellar radii and with the relative free-fall velocities of the corresponding point mass values". The focus is "central collisions". Free fall from rest at infinity is a zero-energy orbit, so these are radial (head-on) parabolic collisions.

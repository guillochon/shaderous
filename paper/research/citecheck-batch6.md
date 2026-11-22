# Citation check, batch 6 (21 references)

Metadata checked against Crossref records (api.crossref.org/works/<doi>) and SciX/ADS. Full texts read: Stone2008 (arXiv LaTeX, research/stone2008), GardinerStone2005 (arXiv LaTeX, research/gardinerstone2005), Hawley2000 (arXiv PDF, research/hawley2000), Toth2000 (author's PDF, research/toth2000), TothRoe2002 (author's PDF, research/tothroe2002). Everything else was checked from abstracts only (JCP/JFM/SIAM/PoF are paywalled and not on arXiv), as noted per item.

## Stone2008
- Metadata: OK (10.1086/588755; ApJS 178, 137-177, 2008; five authors match)
- Claim: "Athena's unsplit integrators (with GS2005, SG2009) are followed by K-Athena, Idefix, AthenaK" -> OK (for what Stone2008 supports)
  Evidence: Sec. 1: "The numerical algorithms in Athena are based on directionally-unsplit, ... integration methods" (CTU+CT and a predictor-corrector VL method). Whether K-Athena/Idefix/AthenaK follow them is a claim about those codes, not checkable from this paper.
- Claim: "a blast in a magnetized medium (Stone2008)" -> OK
  Evidence: Sec. 8 (3D tests): "Blast wave in a strongly magnetized medium. As a final 3D test, we follow the growth of a strong, spherical blast wave"; a 2D version is also in Sec. 7.

## Miyoshi2005
- Metadata: OK (10.1016/j.jcp.2005.02.017; JCP 208, 315-344, 2005)
- Claim: "HLLD solver of Miyoshi2005, which resolves contacts and Alfven waves" -> OK
  Evidence: Abstract: Riemann fan "should consist of four intermediate states"; the solver "can exactly resolve isolated discontinuities formed in the MHD system" (contact and rotational discontinuities).

## GardinerStone2005
- Metadata: OK (10.1016/j.jcp.2004.11.016; JCP 205, 509-539, 2005)
- Claim: "unsplit integrators of Athena" -> OK
  Evidence: Abstract: "single step, second-order accurate Godunov scheme ... corner transport upwind (CTU) method ... constrained transport (CT)".
- Claim: "The electric fields on the edges follow the construction of GS2005" (and remains the default) -> OK
  Evidence: Abstract: "we ... derive new algorithms for constructing the CT fluxes at grid cell corners which reduce exactly to the equivalent one-dimensional solver".
- Claim: "the field loop of GS2005" -> OK
  Evidence: Sec. 4.x "Field Loop Advection": "The first problem we consider is the advection of a weak magnetic field loop."

## TothRoe2002
- Metadata: OK (10.1006/jcph.2002.7120; Toth, G. & Roe, P. L.; JCP 180, 736-750, 2002). The batch listing showed "?" for authors only because the parser choked on {T{\'o}th}; the bib entry itself is correct.
- Claim: "aims of TothRoe2002: exactly free of divergence, conservative, keeping the fine faces that exist" -> OK
  Evidence: Sec. 2: "If some of the faces of the coarse cell are shared with finer neighbor cells, then we should simply copy the fine fluxes"; prolongation must "satisfy a flux conservation"; divergence-preserving per abstract.

## Toth2000
- Metadata: OK (10.1006/jcph.2000.6519; Toth, Gabor; JCP 161, 605-652, 2000). Same parser note as TothRoe2002.
- Claim: "a circularly polarized Alfven wave (Toth2000)" -> OK
  Evidence: Sec. 6.3.1 "Smooth Alfven waves": "The circularly polarized Alfven waves are analytic solutions of the MHD equations for arbitrary amplitudes."

## Hawley2000
- Metadata: OK (10.1086/308180; ApJ 528, 462-479, 2000)
- Claim: "Four poloidal loops with no net flux, as in Hawley2000 but laid along the radius" -> OK (with a note)
  Evidence: Sec. 3: "A suitable initial configuration consists of magnetic field loops lying along equidensity surfaces in the torus", from A_phi = rho, contained within the torus. Hawley's loops are one nested set following the density contours, not four; our sentence already flags the different layout, so it is fair. Optional sharper wording below.
  Fix (optional): "Four poloidal loops of field with no net flux thread it; \citet{Hawley2000} used loops along the surfaces of constant density, while ours are laid along the radius, with ..."

## Balsara1999
- Metadata: OK (10.1006/jcph.1998.6153; Balsara & Spicer; JCP 149, 270-292, 1999)
- Claim: "the rotor of Balsara1999" -> OK
  Evidence: Own full text not readable (paywalled); abstract says "We have devised several stringent test problems". Toth (2000) Sec. 6.6 "Rotor": "This test problem is taken from BS" (BS = Balsara & Spicer 1999).

## Karni1996
- Metadata: OK (10.1137/S106482759528003X; SIAM J. Sci. Comput. 17, 1019-1039, 1996)
- Claim: "conservative scheme carrying mass fractions makes pressure oscillate where gases with different gamma meet" -> OK
  Evidence: Abstract only: the conservative update "may produce oscillations near material interfaces"; caused by "failure of the schemes to maintain pressure equilibrium between the fluid components".

## Abgrall1996
- Metadata: OK (10.1006/jcph.1996.0085; JCP 125, 150-160, 1996)
- Claim: same as Karni1996 -> OK
  Evidence: Title ("How to Prevent Pressure Oscillations in Multicomponent Flow Calculations") and abstract ("pressure oscillation free through the contact discontinuities", "positivity of mass fractions"). Full text not read.

## QuirkKarni1996
- Metadata: OK (10.1017/S0022112096007069; JFM 318, 129-163, 1996)
- Claim: "the helium bubble of QuirkKarni1996"; "a shock in air hits a bubble of helium" -> OK
  Evidence: Abstract: "a Mach 1.22 planar shock wave, moving through air, impinges on a cylindrical bubble which contains either helium or Refrigerant 22."

## Balbus1991
- Metadata: OK (10.1086/170270; ApJ 376, 214, 1991; paper I title matches ADS)
- Claim: "the magnetorotational instability (Balbus1991) grows" -> OK
  Evidence: Abstract: disks "dynamically unstable to axisymmetric disturbances in the presence of a weak magnetic field".

## Evans1988
- Metadata: WRONG (title): the publisher's title is "Simulation of Magnetohydrodynamic Flows: A Constrained Transport Method", not "... Model" (Crossref record of 10.1086/166684; the ADS record carries "Model", which our entry copied). Authors, ApJ 332, 659, 1988, DOI are correct.
- Claim: "constrained transport of Evans1988"; "field lives on the faces of the cells, as in CT of Evans1988" -> OK
  Evidence: Title/abstract (CT method, face-centered staggered field); standard attribution.

## Brio1988
- Metadata: OK (10.1016/0021-9991(88)90120-9; JCP 75, 400-422, 1988)
- Claim: "the shock tube of Brio1988" -> OK
  Evidence: Abstract: "we chose a coplanar MHD Riemann problem ... compound waves consisting of a shock and attached to it a rarefaction wave ... can exist".
- Claim: Table (MHD validation): "Compound wave, slow shock and least v_y in the ranges of Brio1988" -> PARTLY / UNVERIFIED
  Evidence: Full text not readable (paywalled). The ranges (compound wave x in [-0.05,-0.01], slow shock x in [0.13,0.16], min v_y in [-1.62,-1.55]) are set in our own src/validation/tests.ts; Brio & Wu publish profiles in figures, not ranges. "the ranges of Brio1988" reads as if they published the ranges.
  Fix: "Compound wave, slow shock and least $v_y$ in ranges set around the solution of \citet{Brio1988}".

## Haas1987
- Metadata: OK (10.1017/S0022112087002003; JFM 181, 41-76, 1987)
- Claim: "A shock in air hits a bubble of helium" -> OK
  Evidence: Abstract: weak shock in air on soap bubbles and cylinders filled "with gases both lighter and heavier than the ambient air" (helium and R22 per Quirk & Karni 1996 abstract).

## Fickett1979
- Metadata: WRONG (minor): second author is William C. Davis; "C. Davis" is ADS's truncation. Correct: Fickett, Wildon and Davis, William C., Detonation, University of California Press, Berkeley, 1979 (ISBN 978-0-520-03587-4; library catalogue: "Davis, William Chester, 1925-"). Reprinted by Dover (2000) as "Detonation: Theory and Experiment".
- Claim: "single-step Arrhenius reaction, the standard model in the theory of detonations (Fickett1979)" -> UNVERIFIED
  Evidence: Book not readable online. Plausible and widely attributed (the book's idealized models use a one-step irreversible reaction), but not checked against the text.

## Orszag1979
- Metadata: OK (10.1017/S002211207900210X; JFM 90, 129-143, 1979)
- Claim: "the vortex of Orszag1979" -> OK (note)
  Evidence: Abstract: 2D MHD turbulence by direct simulation. Note the original is incompressible; the compressible test is the usual adaptation (Picone & Dahlburg 1991), so the attribution is the conventional one. Numbers in the citing sentences are ours.

## Fickett1966
- Metadata: OK (10.1063/1.1761791; Phys. Fluids 9, 903-916, 1966)
- Claim: "At a higher activation energy the front pulses (Fickett1966)" -> PARTLY
  Evidence: Abstract: single-step Arrhenius "A -> B", results "consistent with Erpenbeck's linearized analysis", "the shock pressure is found to oscillate", peak "nearly 50% higher". That pulsation occurs is supported; that it is the activation energy that is raised is not in the abstract (the classic Fickett-Wood case varies the overdrive at fixed E = 50, per secondary sources). Full text not readable.
  Fix: "Beyond the stability limit of the steady solution the front pulses \citep{Fickett1966}, ..." (or keep "at a higher activation energy" and add Erpenbeck's stability analysis as the citation for the dependence).

## Colella1986
- Metadata: OK (10.1137/0907073; SIAM J. Sci. Stat. Comput. 7, 1059-1080, 1986)
- Claim: "burning in a captured shock thinner than a cell lets the scheme rather than the reaction set the front speed" -> OK
  Evidence: Abstract: "on sufficiently coarse meshes ... purely numerical bifurcating wave patterns with numerical weak detonations ... nonphysical discrete travelling waves for the difference scheme".

## Ghia1982
- Metadata: OK (10.1016/0021-9991(82)90058-4; JCP 48, 387-411, 1982)
- Claim: "Re=100 cavity, velocity along the vertical line through the center within 0.0065 of Ghia1982 at every height in their table"; "standard benchmark" -> OK
  Evidence: Publisher full text not readable, but a SciX body search for "vertical line" matches this paper's indexed text, and transcriptions of its Table I ("u-velocity along vertical line through geometric center of cavity", Re = 100 column) are widespread. The 0.0065/0.0045/0.0047 values are ours. The same holds for the two validation-table rows that cite it.

# Teaching with Shaderous

Shaderous runs at a web address on whatever GPU the student has. There is nothing to install,
no account, and no cluster time to request. A setup can be handed to a class as a link. These
properties make it practical for teaching, from a ten-minute demonstration in a general
education course to a graduate project on numerical methods.

This guide suggests uses at three levels. The activities are starting points: each comes with
a link that opens the setup it describes, and the questions can be shortened or extended to
fit the time available.

- [Before class: practicalities](#before-class-practicalities)
- [Undergraduates, non-majors](#undergraduates-non-majors)
- [Undergraduates, physics and astronomy majors](#undergraduates-physics-and-astronomy-majors)
- [Graduate students](#graduate-students)
- [Further reading for students](#further-reading-for-students)

---

## Before class: practicalities

**Hardware and browsers.** Shaderous needs a browser with WebGPU: current Chrome or Edge on
Windows, macOS and ChromeOS, recent Firefox, and Safari on recent macOS and iOS. Integrated
laptop GPUs run every 2D problem and 3D problems at $`64^3`$ to $`128^3`$ comfortably. Discrete GPUs
reach $`256^3`$ and deep adaptive meshes. Phones work for 2D problems and small 3D ones, and the
right-hand panel can be hidden on a small screen. Before class, open the
[validation page](https://guillochon.github.io/shaderous/validate.html) on a lab machine. If
the tests pass, that machine is ready.

**Links are the handout.** The address bar always reflects the current setup: preset, grid,
refinement, physics options and view. Set up what you want students to see, then copy the
address or use **Share**. A link opens paused or running, depending on the **Autoplay**
checkbox. A bare visit opens Kelvin-Helmholtz at $`1024^2`$ in 2D. Examples used below:

| What | Link |
|---|---|
| Kelvin-Helmholtz, 2D | [`?problem=kelvin-helmholtz`](https://guillochon.github.io/shaderous/?problem=kelvin-helmholtz) |
| Sod shock tube | [`?problem=sod-x&dim=2&n=256`](https://guillochon.github.io/shaderous/?problem=sod-x&dim=2&n=256) |
| Shock hitting a helium bubble, 2D | [`?problem=shock-bubble&dim=2&n=512`](https://guillochon.github.io/shaderous/?problem=shock-bubble&dim=2&n=512) |
| Sedov-Taylor blast, 3D | [`?problem=sedov&dim=3&n=128`](https://guillochon.github.io/shaderous/?problem=sedov&dim=3&n=128) |
| Tidal disruption, 3D with refinement | [`?problem=tidal-disruption-full`](https://guillochon.github.io/shaderous/?problem=tidal-disruption-full) |

**Interacting.** Brushes add mass, velocity or heat; the obstacle tool draws walls; the star
tool drops a polytropic star into a 3D box; the sink tool places point masses. The query tool
shows the full state of the cell under the pointer. The view can switch between a 3D volume
rendering and a slice, and between about twenty fields (density, temperature, Mach number,
vorticity, entropy, potential, Jeans length, refinement level and more).

**Getting data out.** **Export state** writes an HDF5 file in the Athena++ layout, which
[yt](https://yt-project.org/) opens with `yt.load("file.athdf")`. Students who know Python can
then make their own plots and profiles. **Save checkpoints** writes a file at regular
intervals, which is useful for a time series.

**What it is not.** Shaderous is a teaching and exploration tool with a validated scheme. It is
not a production code: it runs on one GPU in single precision, and has no magnetic fields or
radiation transport. Say so when students compare its results to the literature. For
advanced students, the reasons for its limits make good material in their own right.

---

## Undergraduates, non-majors

*Introductory astronomy, physics for poets, general-education science.*

**Goals.** Build intuition for how gas behaves in space. Pressure pushes, gravity pulls, and
the balance between them makes stars. Shocks are where information arrives faster than sound.
Instabilities turn smooth flows into turbulence. Students should come away knowing that much of
what astronomers know about such events comes from simulations, and that simulations are
experiments with rules.

**Format.** Short instructor demonstrations projected in lecture (5 to 15 minutes), or a
single lab period where pairs of students work through one activity on laptops. No equations
are needed. Use the default views and the brushes.

### Activity 1: Why do clouds look the way they do? (Kelvin-Helmholtz)

Open [Kelvin-Helmholtz](https://guillochon.github.io/shaderous/?problem=kelvin-helmholtz). Two
layers of gas slide past each other, and waves along the boundary curl into billows.

- Ask students where they have seen this: clouds on a windy day, ocean waves, Jupiter's bands.
- Pause, use the velocity brush to stir a calm region, and resume. Does a disturbance grow or
  fade?
- Switch the view to vorticity. What does the color show? (The spinning of the gas.)

*Takeaway:* the same physics shapes weather on Earth and gas in galaxies.

### Activity 2: Heavy fluid on top (Rayleigh-Taylor)

Open [Rayleigh-Taylor](https://guillochon.github.io/shaderous/?problem=rayleigh-taylor). Dense
gas sits on top of light gas in a gravitational field. Fingers of heavy gas sink and plumes of
light gas rise.

- Connect it to the Crab Nebula's filaments and to mixing in supernova explosions.
- Ask: what would happen if gravity pointed the other way?

### Activity 3: An explosion (Sedov-Taylor blast)

Open the [3D Sedov blast](https://guillochon.github.io/shaderous/?problem=sedov&dim=3&n=128).
A sudden release of energy drives a spherical shock outward.

- Use the slice view to show that the shell is thin and dense and the inside is hot and thin.
- Tell the story of G. I. Taylor estimating the yield of the first atomic bomb from photographs
  of its fireball, using only how its radius grew with time.
- Connect it to supernova remnants.

### Activity 4: How a star holds itself up (polytrope)

Open the [polytrope](https://guillochon.github.io/shaderous/?problem=polytrope&dim=3&n=64).
The star sits still because pressure pushing out balances gravity pulling in.

- Use the heat brush on the core, or add mass with the mass brush. Does the star swell, shrink,
  or ring?
- Ask: what does a star do when its fuel runs out and the pressure drops?

### Activity 5: A star meets a black hole (tidal disruption)

Open the [tidal disruption](https://guillochon.github.io/shaderous/?problem=tidal-disruption-full).
A star passes close to a black hole and is stretched into a stream. Part of the stream falls
back, and part escapes.

- Ask students to predict, before running it, whether the whole star will be swallowed.
- Relate it to observed flares from galaxy centers.
- Compare the variants. A star that only grazes the black hole loses its outer layers and
  survives (Partial). One that passes deeper is torn apart (Full). A centrally concentrated
  $`\gamma = 4/3`$ star must pass deeper than a $`\gamma = 5/3`$ star before it comes apart.
  The impact parameter $`\beta`$ (how deep the orbit goes) can also be changed by hand.

### Assessment ideas

- A one-paragraph "observation log" of one simulation: what they changed, what they predicted,
  what happened.
- Match screenshots to astronomical images: Kelvin-Helmholtz to Jupiter's bands,
  Rayleigh-Taylor to the Crab Nebula, Sedov to a supernova remnant.

---

## Undergraduates, physics and astronomy majors

*Fluid dynamics, astrophysics (stars, ISM, high-energy), mathematical or computational methods,
upper-division labs.*

**Goals.** Connect the equations of fluid dynamics to the flows they produce. Use simulations
quantitatively: measure, compare to analytic results, and judge resolution. Meet the basic ideas
of computational physics: discretization, convergence, stability, conservation.

**Format.** Problem-set questions built on links, or a two- to three-week lab module. Students
need the Euler equations (conservation of mass, momentum and energy) and some ideal-gas
thermodynamics. The [numerics page](numerics.md) is the reference for anyone who wants to know
exactly what the code does.

### Activity 1: The Riemann problem and shock tubes

Open the [Sod shock tube](https://guillochon.github.io/shaderous/?problem=sod-x&dim=2&n=256) in
slice view.

1. Identify the rarefaction, contact discontinuity and shock in density, pressure and velocity.
   Which quantities are continuous across the contact?
2. Use the query tool to read the post-shock state and check the Rankine-Hugoniot jump
   conditions.
3. Repeat at $`n = 64`$, $`128`$, $`256`$, $`512`$. How does the width of the shock (in cells and in
   physical units) change? Why is the contact smeared more than the shock?
4. Compare limiters (add `&limiter=minmod`, `&limiter=mc` to the link). Which keeps the contact
   sharpest? Which oscillates?
5. Try the [strong shock tube](https://guillochon.github.io/shaderous/?problem=strong-shock&dim=2&n=256)
   (pressure ratio $`10^5`$). What makes it harder?

### Activity 2: Sedov-Taylor and dimensional analysis

Using only $`E`$, $`\rho`$ and $`t`$, derive $`R(t) = \xi_0\,(Et^2/\rho)^{1/5}`$. Then open the
[3D blast](https://guillochon.github.io/shaderous/?problem=sedov&dim=3&n=128), export states at
several times (or read the shock radius from the slice), and fit the exponent. Discuss why the
fit deviates early on (the blast still remembers its initial size) and late (it reaches the box
edges).

### Activity 3: Growth rates of instabilities

- **Kelvin-Helmholtz:** measure the growth of the transverse velocity (from exported states,
  or the $`v_y`$ field in slice view) in the linear phase. Compare runs at two resolutions. Is the growth rate
  converged?
- **Rayleigh-Taylor:** for an incompressible interface, linear theory gives growth at
  $`\sigma = \sqrt{A g k}`$ with $`A`$ the Atwood number. Estimate $`\sigma`$ from the run and discuss
  why compressibility and the finite interface thickness change it.
- **Jeans instability:** open [Jeans](https://guillochon.github.io/shaderous/?problem=jeans).
  Derive the dispersion relation $`\omega^2 = c_s^2k^2 - 4\pi G\rho_0`$ and the Jeans length.
  Which perturbations grow and which oscillate?

### Activity 4: Hydrostatic equilibrium and stellar structure

1. Solve the Lane-Emden equation for $`n = 1`$ (it has the closed form $`\theta = \sin\xi/\xi`$).
2. Open the [polytrope](https://guillochon.github.io/shaderous/?problem=polytrope&dim=3&n=64)
   and compare the radial density profile (from the slice or an exported file) with the
   solution.
3. Watch the central density over several dynamical times. How well does the star stay in
   balance? Turn off the well-balanced reconstruction (`&wb=0`) and compare.
4. Open the [cold white dwarf](https://guillochon.github.io/shaderous/?problem=white-dwarf&dim=3&n=64&amr=2).
   Its pressure comes from degenerate electrons. Discuss why more massive white dwarfs are
   smaller, and what happens near the Chandrasekhar mass.

### Activity 5: Thermal instability and the phases of the ISM

Open the [thermal instability](https://guillochon.github.io/shaderous/?problem=thermal-instability).
Interstellar gas that starts in an unstable balance between heating and cooling separates into
cold clouds and a warm medium.

- Plot the cooling curve and identify the stable and unstable branches (Field's criterion).
- View temperature and density. What sets the size of the smallest clouds? Compare with
  conduction off (`&kappa=0`) at two resolutions. With conduction, the clouds stop shrinking
  as the grid is refined; without it they keep shrinking.

### Activity 6: Accretion onto a binary

Open the [binary](https://guillochon.github.io/shaderous/?problem=binary). Two point masses
orbit inside a viscous disk, clear a cavity, and feed through streams. Discuss angular
momentum transport, the role of viscosity ($`\alpha`$ prescription), and why the cavity is about
twice the binary's separation.

### Lab module: "Is it converged?"

A two- to three-week module that teaches the most important habit in computational physics:

1. Pick a problem with a known answer (Sod, Sedov, Jeans growth rate, polytrope).
2. Run it at four resolutions, export each, and compute an error norm against the analytic
   solution in Python with yt.
3. Plot error against cell size on log axes and measure the order of convergence. Explain why
   shock problems converge at first order regardless of the scheme's formal order.
4. Repeat with a different limiter or Riemann solver (`&riemann=hll`) and explain the
   differences.

Deliverable: a short report with figures, written like a methods section.

---

## Graduate students

*Computational astrophysics, numerical methods for PDEs, astrophysical fluid dynamics,
research training.*

**Goals.** Understand modern Godunov methods well enough to choose, configure and criticize
them. Learn what conservation, positivity and well-balancedness mean in practice and how they
are tested. Use Shaderous to prototype a question quickly before committing to a production
code. For students interested in computing, study GPU architecture through a complete,
readable code.

**Format.** Assigned reading from [numerics.md](numerics.md) and
[architecture.md](architecture.md) alongside a textbook (see below), problem sets that
include changing the source, and term projects. The code is about 6,000 lines of WGSL and
15,000 of TypeScript, and each physics module is a small file. Students can clone the repository, run
`npm install && npm run dev`, and change a shader in minutes.

### Topics and exercises

**Riemann solvers and dissipation.**
- Compare HLL, HLLC and low-Mach HLLC on a contact discontinuity and on a Gresho vortex
  (`&riemann=hll|hllc|hllclm`). Measure the kinetic energy lost. Explain, from the wave
  structure, why HLL smears contacts and why Godunov fluxes overdamp slow flow.
- Read [Quirk (1994)](https://doi.org/10.1002/fld.1650180603) and find the carbuncle. Disable
  the shock-flag HLL blend in `hydro/flux.wgsl` and run the
  [Mach 3 forward step](https://guillochon.github.io/shaderous/?problem=forward-step).

**Reconstruction and limiters.**
- Derive the third-order face value $`q_c + (\delta_\text{far} + 2\delta_\text{near})/6`$ and show
  where TVD limiters clip extrema. Measure the convergence order of smooth advection
  (`?problem=advection`) with each limiter.
- Change the smoothness threshold `SMOOTH` in `hydro/recon.wgsl` and explore the trade between
  peak preservation and oscillations.

**Hydrostatic equilibrium in finite precision.**
- Run a stratified atmosphere with and without the well-balanced reconstruction (`wb=0`).
  Explain from [Käppeli & Mishra (2014)](https://doi.org/10.1016/j.jcp.2013.11.028) why a
  standard scheme generates spurious velocities and why the discrete trapezoid rule removes
  them.
- Discuss why mapping a 1D stellar model onto a 3D grid excites oscillations, and how settling
  against the grid's potential fixes it
  ([Zingale et al. 2002](https://doi.org/10.1086/342754)).

**Self-gravity and energy conservation.**
- Run the [Evrard collapse](https://guillochon.github.io/shaderous/?problem=evrard&dim=3&n=128)
  and track kinetic, thermal and potential energy. Compare the flux and kick forms of the
  gravitational energy source and explain the trade (total-energy conservation against
  overheating of thin gas).
- Explain why applying the whole gravity kick after the flux update cools a star in balance
  (kick-drift-kick versus a single kick).

**Adaptive mesh refinement.**
- Compare the uniform grid with AMR (`&amr=1..3`) on a Sedov blast: cost, accuracy, and where
  the mesh refines. Explain the Löhner indicator and why first differences over-refine smooth
  gradients.
- Explain how Berger-Colella flux correction keeps AMR conservative, then how Shaderous gets
  the same result by recomputing fluxes instead of storing them.
- Turn on local timesteps (`&tbins=3` with `&amr=3`) and measure the speedup on a problem with a small refined
  region and on one where refinement fills the box. Explain the difference.

**Resolution criteria.**
- Collapse an isothermal cloud
  ([`?problem=isothermal-collapse`](https://guillochon.github.io/shaderous/?problem=isothermal-collapse&dim=3&n=128&amr=2))
  and view the Jeans length in cells. Relate it to the Truelove criterion and to artificial
  fragmentation.
- On the thermal instability, relate the size of the smallest clouds to the Field length
  ([Koyama & Inutsuka 2004](https://doi.org/10.1086/382478)).

### Term projects

1. **Add a Riemann solver** (e.g. the exact solver or a Roe solver with an entropy fix) as a
   new module in `hydro/riemann/`, and compare it with HLLC on the validation suite.
2. **Add a test problem** with an analytic solution (a Noh implosion, an isentropic vortex, a
   Bondi accretion flow) in `src/problems/`, plus a validation test with a pass criterion.
3. **Implement an implicit or super-time-stepping diffusion solver** and measure when it beats
   the explicit one.
4. **Single versus double precision.** Use the float64 CPU reference (`src/reference/hydro.ts`)
   to quantify where 32-bit arithmetic limits accuracy (long advection, a uniform background on a
   power of two, high Mach numbers) and how the dual-energy formalism helps.
5. **Performance study.** Use **Time GPU passes** to profile each pipeline on two GPUs. Build a
   roofline argument for why the flux update is limited by memory, and test one optimization.
6. **A science question.** Use Shaderous as a fast exploration tool, for example how the
   disruption of a star depends on $`\beta`$ and the polytropic index, or how a binary's
   accretion rate depends on viscosity. Check the key result against a production code
   such as Athena++ ([Stone et al. 2020](https://doi.org/10.3847/1538-4365/ab929b)) or FLASH
   ([Fryxell et al. 2000](https://doi.org/10.1086/317361)).

---

## Further reading for students

Introductory and textbook treatments, roughly in order of depth:

- H. Trac & U.-L. Pen (2003), *A Primer on Eulerian Computational Fluid Dynamics for
  Astrophysics*, PASP 115, 303. [doi:10.1086/367747](https://doi.org/10.1086/367747),
  [arXiv:astro-ph/0210611](https://arxiv.org/abs/astro-ph/0210611). A short, readable
  introduction for advanced undergraduates.
- M. Zingale, *Computational Hydrodynamics for Astrophysics* (open lecture notes).
  [doi:10.6084/m9.figshare.5339668](https://doi.org/10.6084/m9.figshare.5339668). Free, with
  Python examples; the notes have been revised since the archived version.
- R. J. LeVeque (2002), *Finite Volume Methods for Hyperbolic Problems*, Cambridge University
  Press. [doi:10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253). The standard
  graduate text on the mathematics.
- E. F. Toro (2009), *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed.,
  Springer. [doi:10.1007/b79761](https://doi.org/10.1007/b79761). The reference for every solver
  in Shaderous.
- J. M. Stone et al. (2008), *Athena: A New Code for Astrophysical MHD*, ApJS 178, 137.
  [doi:10.1086/588755](https://doi.org/10.1086/588755). A model of how to describe and test a
  production code.
- A. Mignone et al. (2007), *PLUTO: A Numerical Code for Computational Astrophysics*, ApJS 170,
  228. [doi:10.1086/513316](https://doi.org/10.1086/513316).

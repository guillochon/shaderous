# Plan: turbulence tools after the taranis WebGPU solver

Status: 2026-11-17. Landed: stage 0's credit (README, preset blurbs, `docs/numerics.md`), stage 3's
driving (without the mean-force removal, the power control or the stir brush), and stage 4's
problems: driven turbulence with and without a field, Elsasser packets, the Alfvén-wave collision,
magnetized Kelvin-Helmholtz, and, with the explicit resistivity of open decision 4 (done as part of
this plan, `mhd/ohm.wgsl`), the tearing mode, island coalescence, and X-point collapse inside the
line-tied walls it needs (boundary kind 4, after consolidating the boundary code into
`hydro/boundary.wgsl` and `problems/boundaries.ts`). Not done: stages 1, 2 and 5; validation of
the tearing growth rate.

Source: the WebGPU reduced-MHD solver by Alfred Mallet and Esrom Abraham, a browser port of
their JAX code taranis (<https://alfredmallet.github.io/taranis/webgpu/rmhd2d.html>, with a 3D page
at `rmhd3d.html`; code at <https://github.com/alfredmallet/taranis>, cited as Mallet & Abraham
2026, Zenodo, doi:10.5281/zenodo.21987739). It solves reduced MHD pseudospectrally on a uniform
periodic grid, in fp32 on WebGPU, as Shaderous does.

Goal: bring over the parts of that page that don't depend on its spectral solver. Together they
would make Shaderous a driven, compressible MHD turbulence code, a regime the reduced-MHD
solver can't reach (shocks, compressibility, gravity, adaptive mesh).

| Stage | What it adds | Depends on |
|---|---|---|
| 0 | Credit, in place before any code lands | nothing |
| 1 | Derived fields: current, Elsasser variables, cross helicity, residual energy | nothing |
| 2 | Energy spectra as a diagnostic | nothing |
| 3 | Turbulence driving (Ornstein–Uhlenbeck forcing in a band of wavenumbers) | 2, for checking |
| 4 | New problems: driven MHD turbulence, Elsasser packets, Alfvén-wave collision, tearing and X-point setups | 1–3; tearing also needs resistivity |
| 5 | Smaller ideas: pinned "ghost" curves, typed-expression initial conditions | 2 for ghosts |

Each stage is useful alone and ends in tests, a section of `docs/numerics.md`, and the credit
lines of stage 0.

**Not transferable:** the integrating-factor time stepping, which relies on a linear operator
that is diagonal in Fourier space. Alfvén waves in Shaderous live in the Riemann solver, and the
CFL condition sets the step. Also not transferable: spectral hyperdissipation (k^2n), since the
Godunov scheme's dissipation is mostly numerical, and the reduced-MHD equations themselves,
which would need a second solver.

---

## 0. Credit (applies to every stage)

The ideas come from the taranis WebGPU page, and every stage that uses one says so.

**License.** taranis is GPL-3.0-or-later and Shaderous is MIT. **No taranis code is copied,
translated or ported.** Each feature is written from its physical description and from the
standard literature: the page's text, our own derivations, and the papers named below. Where
the page only suggested a feature (which quantities to show, which setups to offer), crediting
the idea is enough. Anyone writing a stage should not read the taranis WebGPU source while
doing so.

**Where the credit goes**, as each stage lands:

- **README**, in an acknowledgments line: "The turbulence driving, energy spectra, Elsasser
  fields and the reconnection setups follow the WebGPU reduced-MHD solver of Alfred Mallet and
  Esrom Abraham, built on their code taranis." The line lists only what has landed.
- **The app.** Each problem from stage 4 says in its blurb that it follows the taranis WebGPU
  page and links to it. The driving controls' tooltip says the same. If the app gains an
  about/credits panel, the page is listed there.
- **`docs/numerics.md`**: the new sections on spectra and driving cite Mallet & Abraham (2026)
  beside the method papers.
- **The paper** (`paper/paper/main.tex`): the related-work note is already in (the
  introduction and Section `sec:prior` cite `Mallet2026`). When a feature lands, the section
  that describes it cites `Mallet2026` again, and the related-work paragraph stops listing that
  feature as one Shaderous lacks.
- **Commit messages** for each stage name the source.
- **Telling the authors.** Before a release announces these features, James lets Alfred know,
  so the authors see the credit before the public does (James's call; see open decisions).

The paper note had to come first, since a browser MHD solver already existed before Shaderous's
MHD was described. Our novelty claim is now limited to compressible flow, shocks, gravity and
adaptive refinement in the browser.

## 1. Derived fields

Pointwise quantities, computed in `render/field_value.wgsl` and added to `FIELDS` in
`problems/types.ts` under "Magnetic field". Today the menu has B components, |B|, plasma beta,
div B and the MRI quality factors, but no current or Elsasser fields.

- **Current density** J = curl B (signed J_z in 2D, |J| in 3D), from the face fields by the
  same centered differences the vorticity uses.
- **Alfvén speed** |B|/sqrt(ρ), with the factor of 4π (or not) matching the convention in
  `mhd/` (check it before writing the shader).
- **Elsasser fields** z± = u ± B/sqrt(ρ), shown as |z+| and |z−|.
- **Normalized cross helicity** σ_c = (|z+|² − |z−|²)/(|z+|² + |z−|²) and **residual energy**
  σ_r = (|u|² − |b|²)/(|u|² + |b|²), with b = B/sqrt(ρ). Both lie in [−1, 1] and go on the
  signed, linear list.

In compressible flow, z± should use the fluctuating velocity about the box mean, plus a mean
field set by the problem. Decide whether to subtract the volume-averaged B (open decision 1).

**Done when:** the fields show for every MHD problem; a test checks them against a hand-computed
Alfvén wave (σ_c = ±1 for a wave travelling one way); the problem tips list them; and the
README line mentions them.

## 2. Energy spectra

A diagnostic only, so the solver never sees an FFT.

- **What:** shell-binned kinetic and magnetic spectra E_u(k) and E_b(k), and E±(k) from the
  Elsasser fields. For compressible runs, also the √ρ·u spectrum (kinetic energy density) and
  the solenoidal/compressive split of u from a Helmholtz decomposition in Fourier space.
- **Where:** on the uniform grid directly. On the adaptive mesh, on the base level, or on one
  level resampled to a uniform grid. Report which level, and the resolution the spectrum
  represents.
- **How:** a GPU FFT in WGSL (radix-2 and radix-4 passes, n = 2^m, which every uniform grid
  in the menu already is), then shell binning with atomics or a staged reduction like the
  existing ones. For a first version, read the data back and run the FFT on the CPU up to
  256²/128³, then move it to the GPU once the readout format is settled.
- **Shown:** a log-log chart panel beside the existing diagnostics, with optional
  k^(−5/3), k^(−3/2) and k^(−2) guide lines that can be pinned to the curve, and a compensated
  view (k^α E(k)).
- **Scripting:** `window.shaderous`, the MCP `diagnostics`/`stats` tools and the command-line
  runner can return the spectrum as arrays, and a run can save spectra at a cadence next to its
  `.athdf` checkpoints.

**Done when:** a single Fourier mode lands in its shell with the right energy (Parseval to
round-off); an Orszag–Tang run shows its known spectral evolution; and the API is documented in
`docs/api.md` and `docs/headless.md`.

## 3. Turbulence driving

Forcing as a momentum source term (and the matching energy term), following the page's
Ornstein–Uhlenbeck idea, written from Eswaran & Pope (1988) and, for the
solenoidal/compressive mix, Federrath et al. (2010). Check both references with the `scix`
skill before citing them anywhere.

- **Modes:** every wavevector with |k|L/2π in a band [n_min, n_max] (default 1–3: a few dozen
  modes in 3D). Each mode's complex amplitude follows an OU process with correlation time τ,
  advanced on the CPU from a seeded generator and uploaded as a small uniform or storage buffer
  once per root step.
- **No FFT:** each cell sums the modes directly (sin and cos of k·x). At a few dozen modes this
  is cheap next to the flux stencil, and it works on any AMR level, since it only needs the cell
  position.
- **Projection:** per mode, a projection tensor with weight ζ (1 solenoidal, 0 compressive,
  1/2 natural mix).
- **Power control:** either a fixed amplitude, or rescaling each step so that the injected power
  ⟨ρ u·f⟩ matches a target ε. Remove the mean force so the box gains no net momentum.
- **Steady state:** driving heats the gas. Offer isothermal runs (the barotropic EOS with
  γ = 1) and a simple cooling term toward a target temperature.
- **Per-block timesteps:** the forcing pattern changes with time, so a block on a finer time bin
  must see the force at its own substep time. Interpolate the OU amplitudes between root steps
  (open decision 2).
- **Painting:** the page lets a person stir with forcing blobs. Shaderous already has brushes; a
  "stir" brush that injects a localized solenoidal kick is the matching feature.
- **Reported:** injected power, and the time-averaged dissipation rate, in the diagnostics.

**Done when:** a hydro run reaches a statistically steady rms Mach number set by ε; the
injected power matches the measured kinetic energy budget; the velocity spectrum peaks in the
forcing band; runs are reproducible from a seed; and the URL carries every driving parameter.

## 4. Problems

Each problem's blurb credits the taranis WebGPU page and links it.

- **Driven MHD turbulence**: periodic box, 2D and 3D, mean guide field B0 along z, driven by
  stage 3. Presets: subsonic with a strong guide field (the regime where compressible MHD
  approaches reduced MHD, and Shaderous can be compared against the taranis page directly),
  and supersonic isothermal.
- **Decaying Elsasser packets**: two counter-propagating Alfvén-wave packets of opposite σ_c,
  as letters or blobs, decaying through their collisions.
- **Alfvén-wave collision (3D)**: two perpendicular, counter-propagating waves along a guide
  field. Check the setup against the published strong-interaction results (Howes & Nielson
  2013, to be checked).
- **Tearing mode and Harris sheet**: a current sheet in pressure balance, perturbed. Shaderous
  has no explicit resistivity yet (`docs/numerics.md` §11 has viscosity and conduction only), so
  this problem needs an Ohmic term in the CT update first, or else it measures numerical
  resistivity. The growth rate and island width plotted against the Furth–Killeen–Rosenbluth
  scaling make a validation test once resistivity exists.
- **X-point collapse and island chain (coalescence)**: same dependency on resistivity. The
  diagnostic is the reconnected flux over time.
- **Kelvin–Helmholtz with a field**: the existing KH problem with an in-plane field option,
  matching the page's shear-layer preset.

**Diagnostics** the page has and these problems need: energy traces split into E_u and E_b (and
E±), a cut along a line (the `slice` tool already does most of this), and island width or
reconnected flux for the tearing family.

**Done when:** each problem runs in 2D and 3D where it makes sense; it has a validation-suite
entry where a known answer exists (wave collision, tearing growth rate); and its blurb carries
the credit.

## 5. Smaller ideas

- **Pinned ghost curves:** freeze a spectrum or a time trace as a faint curve, change a
  parameter, and compare. Cheap once stage 2's chart exists, and useful for teaching.
- **Typed-expression initial conditions:** a field such as ρ(x, y, z) or A_z(x, y) entered as a
  formula with a safe parser (no `eval`), using the same function list as the page. The URL
  carries it.
- **Cross-code check:** in the strong-guide-field, low-Mach limit, driven or decaying Shaderous
  runs should match reduced MHD statistically. Comparing spectra against a taranis run (run by
  its own code, not ours) is a validation test that also exercises stages 1–3.

## 6. Open decisions

1. Elsasser fields in compressible flow: subtract the mean velocity and field, or use the
   problem's B0? Density-weighted (√ρ) or not?
2. Driving with per-block timesteps: interpolate the OU amplitudes in time, or hold them fixed
   over a root step and accept a small error?
3. Spectra on the adaptive mesh: base level only, or a resampled finest level up to a memory
   limit?
4. Resistivity: add an explicit Ohmic term (needed for stage 4's tearing family) as its own
   plan, or as part of this one?
5. When and how to tell Alfred Mallet before release (stage 0): James's call.

## 7. References

Check each with the `scix` skill before it goes into `numerics.md` or the paper.

- Mallet, A. & Abraham, E. 2026, *taranis: a spectral solver for nonlinear plasma models in
  JAX*, v0.1.0, Zenodo, doi:10.5281/zenodo.21987739. WebGPU page:
  <https://alfredmallet.github.io/taranis/webgpu/rmhd2d.html>. Already in
  `paper/paper/references.bib` as `Mallet2026`.
- Eswaran, V. & Pope, S. B. 1988, Ornstein–Uhlenbeck forcing for spectral turbulence
  simulations.
- Federrath, C. et al. 2010, solenoidal versus compressive forcing.
- Furth, H. P., Killeen, J. & Rosenbluth, M. N. 1963, the tearing instability.
- Howes, G. G. & Nielson, K. D. 2013, Alfvén-wave collisions.

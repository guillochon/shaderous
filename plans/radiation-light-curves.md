# Plan: light curves from radiation transport

Status (2026-11-18): not started, and set aside for now. What exists is in section 1. Written
after stage 5 of `plans/radiation-m1.md` (radiation on the adaptive mesh).

Goal: light curves a reader would compare with observations, for the two kinds of transient the
code can reach: a shock breaking out of a star or a shell, and a tidal disruption.

## 1. What there is

- Gray M1 radiation on the uniform grid and on blocks (`docs/numerics.md` section 24).
- `L rad` in the Diagnostics panel: the radiation energy leaving through the open faces of the
  box per unit time, from a total the GPU counts step by step (`radiation/escape.wgsl`,
  `amr/rad_escape.wgsl`). Its history and CSV are a bolometric light curve.
- Sinks report their accretion rate (`Ṁ` in the panel, `sink1_mdot` in the CSV).

What that light curve is: the light at the box's edge, in the gas's frame, gray. It suits a
source whose light is made by the gas on the grid: a hot body cooling, a shock breaking out.

## 2. Why it is not yet a tidal disruption's light curve

The light of a tidal disruption that people fit is powered by accretion onto the black hole
(and by the stream's self-intersection shocks), not by the star's own heat. In our runs the gas
that reaches the hole enters a sink and is gone: none of its energy comes back as radiation. So
`L rad` for a disruption today shows the debris cooling, which is not the observed transient.

What is missing, in order of need:

1. **A sink that shines.** Radiation energy put into the cells around a sink at
   `L = η Ṁ c²` (η a setting, 0.1 by default; capped at a multiple of the Eddington luminosity,
   also a setting), isotropic, as a source term in the sweeps. `plans/radiation-m1.md` section
   10 notes that the `Mass` record has no room for a luminosity; either it grows, or the
   luminosity rides on `SourceSpec`. `Ṁ` is already measured between steps.
2. **Reprocessing by the debris.** The emitted light must diffuse out through the bound debris
   and any outflow: that is the transport we have, but it needs the debris resolved where it is
   opaque, so the adaptive mesh, and a refinement criterion on the radiation energy
   (`plans/radiation-m1.md` section 5: Löhner's indicator on `E`, not yet written).
3. **The simplest stand-in, with no transport.** `L = η Ṁ c²` from the sink's accretion rate,
   as a column of the diagnostics and its CSV. No radiation run is needed for it, and it is
   what most disruption papers plot. Worth doing first: an afternoon's work.
4. **Per-block timesteps with radiation** (stage 6 of `plans/radiation-m1.md`). A disruption
   spans a large range of cell sizes, and one timestep for all blocks makes it slow.
5. **The reduced speed of light near the hole.** The rule is `ĉ ≫ |v| + c_s`; debris near
   pericenter moves at a good fraction of `c`, so `ĉ` cannot be much reduced there. To measure:
   what `ĉ/c` a run can afford, and what it costs.

## 3. Shock breakout: what the present light curve is for

A preset, and a check against a known answer:

- **Preset.** A strong explosion at the center of a star with a steep outer density profile
  (the polytrope, energy put in as the Sedov presets do), radiation on, open boundaries. The
  shock steepens down the density gradient, and when the optical depth ahead of it falls to
  about `c / v_shock` the radiation behind it leaks out: a flash, then a cooling tail.
- **To check.** The time of the peak against where `τ = c / v_shock`; the energy in the flash
  against the radiation energy behind the shock at that moment; the tail's slope against the
  planar and spherical self-similar solutions (Nakar & Sari 2010; Sapir, Katz & Waxman 2011).
- **Resolution.** The breakout layer is thin, a fraction of a percent of the radius: the
  adaptive mesh with refinement on `E` (item 2 above) or on the density gradient, in 2D
  axisymmetry first. Radiation is Cartesian only today (`docs/numerics.md` section 24, "Not
  yet"), so (R, z) geometry for the radiation passes is a prerequisite, or the test runs in a
  3D octant at lower resolution.
- **The delay.** The light crosses from the surface to the box's edge in the box's size over
  `ĉ`. For a flash that is short compared with that, either the box is kept tight around the
  star or the luminosity is measured on a sphere just outside the surface instead of at the
  faces: a second reduction over the cells of a shell, `F · r̂ dA`.
- **Frame and color.** The flux is the gas's frame's; at the surface the gas moves at
  `v_shock`, so the lab-frame flux differs by `v E (1 + χ)`: add it to the measured luminosity.
  Gray transport gives no color temperature; `(L / 4πR²σ)^(1/4)` is the most to report.

## 4. Order of work, when this is taken up

1. Accretion luminosity `η Ṁ c²` as a diagnostics column (2.3).
2. Luminosity through a sphere of chosen radius, in the lab frame (section 3's delay and frame).
3. Refinement on the radiation energy.
4. The shock-breakout preset and its checks.
5. Sinks that shine (2.1), then a disruption with reprocessing (2.2), which wants stage 6 of
   `plans/radiation-m1.md` first.

## 5. Open questions

- Whether `η` should fall at low `Ṁ` (radiatively inefficient flow) and saturate above
  Eddington, or stay a constant the user sets.
- Whether a shining sink should also push: its radiation force on the gas around it is what
  drives an outflow, and M1 from a point source in a few cells is poorly resolved (RAMSES-RT
  prefers global Lax-Friedrichs fluxes around sources for that reason; we have the option).
- How to show two light curves at once (accretion, and through the faces) in a panel that plots
  one column on hover. The larger docked plot was turned down for now; this is when it would
  come back.

# Self-gravity's net force on the adaptive mesh

Measured 2026-11-24 with the headless runner (`tools/headless/cli.mjs run`) on an RTX 3080, builds 4dca9bc (before) and ac348fd (after the correction).

## The problem

An isolated body's gravity exerts no net force on it. On the adaptive mesh it did. A double white dwarf (0.6 + 0.9 Msun, box 16 at 2048^3 with five levels, in a frame that turns with the binary) moved its center of mass away from the axis on an outward spiral. Its distance from the axis e-folded every 36 code units, about 1.8 orbits (the orbital period is 20.25 code units). The growth rate (0.028 per code unit) and the rate at which the spiral turned in the inertial frame (0.035) were nearly equal. That is the signature of a drag toward rest in the rotating frame: s^2 = i gamma Omega, with gamma about 6e-3 per code unit.

## The test

One white dwarf of 0.6 Msun (Helmholtz equation of state, isolated gravity) in a box 8 code units across, n = 512 with three levels (finest cell 8/512 = 0.0156 code units, 156 km), started at x = -0.5 moving at 0.05 code units along x (182 km/s) through a background of density 1e-6 at rest. The frame is inertial. Code units: 10^9 cm, 1 Msun, so the time unit is 2.74 s and 10 code units are 27.4 s. The table gives the change of the total momentum along x over the first 10 code units.

| Run | Momentum change in 10 code units | Wall time to t = 10 |
|---|---|---|
| 0.6 Msun, adaptive mesh, before | -1.81 percent | 53.3 s, 2103 steps |
| 0.6 Msun, adaptive mesh, levels solved one by one (no lag) | +1.06 percent | 68.5 s |
| 0.6 Msun, adaptive mesh, after | -0.023 percent | 53.4 s, 2115 steps |
| 0.6 Msun, uniform grid of 256^3 | +0.023 percent | |
| 0.9 Msun, adaptive mesh, before | -3.84 percent | 102.0 s, 3088 steps |
| 0.9 Msun, adaptive mesh, after | +0.20 percent | 104.5 s, 3094 steps |

The force was proportional to the velocity (0.025, 0.05 and 0.1 gave the same fractional rate, about 1.8e-3 per code unit) and to the timestep (half the CFL number nearly halved it). It grew with the star's density. More V-cycles, remeshing every step, switching off the balanced reconstruction or the `fluff` limiter, and taking gravity out of the predictor each left it nearly unchanged. Pure hydrodynamics on the same mesh (gravity off) conserved momentum to 1e-5.

## The two parts

1. All levels relax together against the boundary values that their coarser level had at the start of the cycle. A body that moves across the mesh therefore feels its own pull from where it was: a drag, -8.7e-5 in force here.
2. A converged potential (the levels solved one by one, or a cold start each step, which gave the same result) still leaves a net force where the refined patch lies unevenly around a moving body: +3.1e-5 here.

## The correction

Each step a reduction over the leaves sums m g of the field just written. The next field subtracts that net force per unit mass as a uniform acceleration a, and adds a . x to the potential so that the flux-form energy and the balanced reconstruction difference the field that the kicks use. It applies with isolated boundaries in 3D Cartesian geometry, on every path of the step, and never along a plane of mirror symmetry. The step time did not change measurably (25.3 ms before, 25.2 ms after, for the 0.6 Msun star). The 12 GPU validation tests with gravity on the adaptive mesh pass.

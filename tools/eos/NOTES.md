# Equation of state for burning white dwarf matter

Shaderous's white dwarf presets used `eos/degenerate.wgsl`: zero-temperature degenerate electrons
plus an ideal gas of ions *and electrons*, plus radiation. That ideal part gives every electron
3/2 kT, but degenerate electrons take up far less heat. In shocked, still-unburned fuel the
temperature reads 2-5 times too low (at 1e7 g/cm^3, gas really at 1e9 K reads 4.5e8 K). Carbon
burning goes as T^20 or more, so detonations failed to ignite or decoupled from their shocks.
The burning presets now use a tabulated electron-positron gas, as every collision paper does
(Timmes & Swesty's Helmholtz): `src/eos/helmholtz.ts` and `eos/helmholtz.wgsl`.

## What ships

- `electron_gas.py`: the electron-positron gas exactly, at any degeneracy and relativity, pairs
  included, and Sommerfeld where very degenerate. It is checked against the classical,
  degenerate and consistency limits (run it).
- The target is the thermal free energy, F_th = -P_pair(T) + n k T h(n, T), with the
  zero-charge pair gas split out (a Chebyshev series in log T, 1e-5 accurate). That keeps h of
  order one everywhere, where g = F_th / (n k T) reaches -1e9 in the hot, thin corner.
- `gen_data.py` puts h and its exact first derivatives on a 0.02 dex grid in log rho Ye
  (0 to 11) and log T (4 to 11).
- `make_table.py` writes `src/eos/helmholtz_table.bin`, one 4.3 MB rgba32float texture:
  - (h, h_x, h_y, h_xy) per node, interpolated by bicubic Hermite;
  - inverse tables log10 T(rho, e_th) and log10 T(rho, p_th / rho) at Abar = 4, 14 and 56,
    which give Newton's method its starting point.
- Accuracy against the exact gas (300 held-out white dwarf states, rho Ye >= 10 g/cm^3):

  | Quantity | Float64 mirror | Single precision on the GPU |
  |---|---|---|
  | e_th, p_th | 7e-6 worst | about 2e-5 |
  | c_s^2 | 2e-4 at p99, 1e-3 worst | 2e-4 |
  | T from e | 2e-6 | — |

- Newton takes 2.6 iterations on average (4 at most) from the inverse tables, against 9.3 from
  a classical-gas guess. With the burning network running, the C/O tube costs 15% more per step
  than with the old EOS (21.8 against 19.0 ms at 512^2 on an RTX 3080).

- The ions' Coulomb energy (added 2026-11-16) is
  analytic and needs no table: the one-component plasma's free energy as Helmholtz has it,
  split into a part that does not depend on T, which joins the zero-temperature electrons, and
  a thermal part, and faded out below about 1e3 g/cm^3 (for C/O), where the cold part would
  otherwise make the pressure negative. `src/eos/helmholtz.ts` has the derivation,
  `tests/unit/helmholtz.test.ts` checks it against differences of the free energy.
  - `EosParams.helmCoulomb` (link key `coulomb`) multiplies it; 1 by default.
  - White dwarfs are built on the same cold curve (`src/problems/white_dwarf.ts`). They are
    smaller: 8586 to 8127 km at 0.6 M_sun, 5428 to 5094 km at 1 M_sun, 2771 to 2290 km at
    1.3 M_sun (the term lowers the Chandrasekhar mass by some 3%).
  - Newton took 3.3 iterations on average with it (5 at most; 2.7 and 4 since the work
    below): the inverse tables were made
    without the Coulomb heat, which the starting point allows for in one analytic step.
  - `helm.py`, which the burner's reference burns and training rollouts use for the heat
    capacity, does not have the term. At burning temperatures Gamma is near one and the
    difference is a few percent of the ions' heat capacity.

- Every white dwarf preset uses this equation of state since 2026-11-16 (`white-dwarf` and
  `wd-collision` were on `eos/degenerate.wgsl`, which remains a choice: link key
  `eos=degenerate`). Without burning the cost shows in full. The `white-dwarf` preset at 128^3,
  uniform, 400 steps on an RTX 3080, twice each:

  | Equation of state | ms per step | after the work below |
  |---|---|---|
  | `degenerate` | 7.3, 7.9 | 7.3, 8.0 |
  | tabulated, Coulomb term off | 21.6, 21.6 | 14.1, 13.7 |
  | tabulated, Coulomb term on | 26.0, 26.9 | 15.3, 15.2 |

  So it was 2.9 times the plain degenerate gas, and 3.5 times with the Coulomb term. The 15%
  above is with the burning network running, which then dominates. The star sits as still
  either way (central density within 3.6% over three dynamical times at 64^3, against 3.3%).

- Making it cheaper (2026-11-16), with nothing given up in accuracy: twice the plain
  degenerate gas now, from 3.5 times. Three cases, 300 steps each, twice: the `white-dwarf`
  preset at 128^3 (cold: most cells lie below the table), the C/O detonation tube at 1024^2
  with burning off (warm: every cell takes the full iteration), and the same tube at 512^2
  with burning.

  | Change | Cold star | Warm tube | Burning tube |
  |---|---|---|---|
  | Before | 25.1 | 6.9 | 13.0 |
  | One solve per cell for pressure and sound speed (`eos_prim`; `hydro/prim.wgsl` solved by the energy, then again by the pressure) | 20.7 | 6.1 | 12.5 |
  | Newton's iterations evaluate only the two table derivatives they need, not five; no pair series below 10^8.3 K; a step under 3e-4 in ln T is taken without another evaluation | 14.6 | 5.6 | 12.6 |
  | The Coulomb term's density factors computed once per solve | 13.8 | 5.8 | 12.2 |
  | The table's four texels fetched again only when the temperature's cell changes | 14.0 | 5.3 | 12.3 |
  | What is returned corrected by one Newton step from the final evaluation (`helm_fix`) | 15.0 | 5.6 | 12.3 |
  | `degenerate`, for scale | 7.5 | 3.0 | |

  ms per step; run to run they differ by 2 to 5%. The last row above `degenerate` costs a
  little and is there for accuracy: stopping on trust alone widened the round trip (pressure
  to energy and back) from 1.6e-5 to 2.6e-5 at worst, since the final point was no longer
  one whose computed residual had been checked. With the correction it is 1.1e-5, better
  than before, and the energy against the float64 mirror is 1.5e-5 (1.7e-5 before); the
  sound speed is unchanged at 2.4e-4.

  Newton now takes 2.7 evaluations on average with the Coulomb term (4 at most).

  What is left is mostly not in this module. On a uniform grid every cell's primitives are
  converted from its conserved state by each cell whose stencil holds it, some 20 times a
  step (13 in the update, 7 in the predictor), each a solve for the temperature; the faces
  add 6 to 12 more. Converting once per cell into a texture of pressure and sound speed
  would take most of the remaining gap, at the cost of that texture and a pass, and of
  changing how every grid loads its stencil. Not done.

Tried and not used:
- Bicubic tables of h's two first derivatives, with exact second derivatives as slopes.
  Worse: the pair onset near 5e8 K at low density is too sharp for differenced slopes.
- Keeping T per cell to warm-start Newton. That would need a new field through every pass,
  remesh and restart, and it would not help the many EOS calls on intermediate states. The
  inverse tables give the same gain everywhere.

## Abandoned experiment: a neural network instead of the table

The idea was a network small enough to be cheaper than a table lookup with its Newton iteration.

1. **Forward network** (`fit.py`). h(log rho Ye, log T) by a tanh MLP, fit to h and both first
   derivatives at once (Sobolev loss), Adam then L-BFGS in double precision, 243k points.

   | Network | Weights | rms error h / h_n / h_T | Worst h_T |
   |---|---|---|---|
   | 2 x 24 | 697 | 1.1e-3 / 1.0e-3 / 2.4e-3 | 5.3e-2 |
   | 2 x 48 | 2,545 | 6.0e-4 / 6.6e-4 / 1.5e-3 | 4.4e-2 |
   | 3 x 32 | 2,241 | 8.4e-4 / 8.6e-4 / 1.3e-3 | 3.0e-2 |
   | bicubic table, 0.02 dex | — | 1.3e-6 / 1.1e-5 / 1.0e-4 | 5.9e-3 |

   L-BFGS levelled off near 1.8e-3. Larger networks hardly helped: the hard parts are a few
   sharp features (the pair onset, the edge of degeneracy). This network would still need
   Newton for T, so it could not save the inversion either.

2. **Inverse network** (`gen_inverse.py`, `fit_inverse.py`): straight from what the hydro holds,
   (log rho, log e_th, log Abar), to (log T, p_th / (rho e_th), thermal Gamma), with no iteration.
   150k exact states. 2 x 64 tanh, 4,611 weights: T to 0.9% rms (3% at p99), p to 0.4% rms,
   Gamma to 1e-2.

3. **GPU benchmark** (`bench/`, `node tools/eos/bench/run.mjs`): 1M states on an RTX 3080.

   | Method | ns per state (median / best) | Worst T error | Worst p error | c_s^2 error |
   |---|---|---|---|---|
   | table + Newton (9.3 iterations, classical guess) | 2.2 / 1.0 | 3e-5 | 1e-4 | 5e-4 at p99 |
   | inverse network 2 x 64 | 6.2 / 2.4 | 5e-2 | 2e-2 | 4e-2 at p99 |

   The table with Newton is 3 times faster and about 1000 times more accurate. One 64-wide
   layer is 4,096 multiply-adds; one Newton iteration is four neighbouring texel fetches and
   about 100 operations. With the inverse tables as the starting point (2.6 iterations instead
   of 9.3), the gap is larger still.

Conclusion: on this smooth, two-dimensional function, a GPU table beats a network on both speed
and accuracy, even counting the inversion. `fit.py`, `fit_inverse.py`, `gen_inverse.py` and
`bench/` are kept as the record of the experiment.

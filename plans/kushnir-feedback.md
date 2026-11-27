# Doron Kushnir's feedback on v0.4.0: status and plan

**Progress (2026-11-27):** points 1 and 2 are done. The tube and collision sweeps confirmed his
numbers (`paper/research/measurements.md`; scripts in `tools/burn/limiter/`). The paper, the
preset tooltips, the option tip and `docs/numerics.md` §20 are rewritten, and both papers are
cited. The collision's 470 km cell run is still to be added. Point 3: the reference is fixed
(step 2 below). `build_net.py` builds pynucastro's aprox13, which burns carbon 1.8 times and
oxygen 3-4 times faster than the first reference. The first reference is kept as
`aprox13_features.py` for the inputs. The data is not re-labeled and the network is not
retrained (steps 1 and 3-6).

Checked against `main` at bdbc31e (2026-11-27). Since v0.4.0 the only burning commits are the
burning watch (fec7208) and a dispatch fix (dda05d8). The network weights
(`src/burn/aprox13_net.ts`, last changed in b04be38), the tube and collision presets, and the
text he quotes are all unchanged. So every point still applies.

## 1. The limiter: the igniter fails, not the limiter

**Status: still wrong in the text.** The claim appears in four places:

- `paper/paper/main.tex:294`: "$f_{\rm b}$ below about 0.2 stopped even resolved detonations".
- The tooltips in `src/presets/detonation-co.json:98` and `src/presets/wd-collision-burning.json:64`:
  "They use 0.1, but here anything under about 0.2 stops even resolved detonations".
- `docs/numerics.md:1965` and `docs/setups.md:529` only cite Kushnir et al. 2013. They don't
  repeat the claim, but they give no condition for when the limiter applies.

His diagnosis is consistent with the setup. The default igniter is a 30 km layer at
`tign` = 4e9 K (`src/setup/parts/flows/co_detonation.ts:58`), which only raises the pressure. It
ignites without the limiter because a 2 km cell burns completely in one step. With the limiter,
the ignition region has to be about dx/f across.

Plan:

1. Reproduce his runs on a static build (see the memory note on GPU tests). Run the tube at
   n = 512 with `tign` = 8 for f = 0.05, 0.1, 0.2, 0.4 and 0, and for layer widths of 30 and
   200 km. Record the front speed (he reports 12,040 and 13,580 km/s against 13,620 km/s with
   the limiter off). Also run `tign` = 4 at widths of 15 and 60 km to confirm that those fail.
2. Rewrite the paper paragraph. The limiter does not stop resolved detonations. A hot-spot
   igniter needs a region about dx/f across to start one. With `tign` = 8, or a wide enough
   layer, f = 0.05 to 0.1 gives the unlimited front speed to within 0.3%. Cite Kushnir, Wygoda &
   Sharon 2020 (arXiv:2008.08592, Sec. 2.2, Table 1) and Kushnir & Katz 2020 (MNRAS 493, 5413),
   including the condition that dx/f must be much smaller than the other scales. Also say that
   0.5 to 4 km cells at 1e7 g/cc don't resolve carbon or oxygen burning, which is the regime the
   limiter is for.
3. Rewrite both preset tooltips the same way, in short form: what f does, that the ignition
   region must exceed about dx/f, and the citations. Add one sentence with the dx/f condition to
   `docs/numerics.md` §20, then regenerate `docs/setups.md` if it is generated from the option
   table.
4. Add both papers to `paper/paper/references.bib` and `bibcodes.json`, and to the reference
   list in `docs/numerics.md`. Run the citecheck pass on the new claims.
5. Optional: make the tube's igniter work with the limiter on, so turning the limiter on in the
   preset doesn't look like a failure. Either raise `tign` when the limiter is on, or add a
   velocity-gradient igniter, which he says needs less than a hot spot. Decide after step 1.

## 2. The collision: unresolved, so not comparable with 0.41 Msun

**Status: still in the paper (`main.tex:294`) and the preset blurb.** The paper places the
yield (0.45 earlier, then 0.52 Msun) between the 0.41 and 0.51 Msun of Kushnir 2013 and Raskin
2010. It also reads the limiter's near-zero yield as the limiter suppressing the detonation.
He gets 0.64 Msun on v0.4.0, 0.67 and 0.74 at CFL 0.6 and 0.4, and 0.04 at n = 64. With 940 km
cells, dx/f at f = 0.1 is larger than the star.

Plan:

1. Re-measure the shipped preset on current `main`, then sweep CFL (0.4, 0.6, default) and
   resolution (n = 64, 128, default, and finer if memory allows). Compare 0.52 (b04be38)
   with his 0.64 (v0.4.0) to see whether hydro changes since b04be38 moved the yield.
2. Rewrite the paper: present the collision as a demonstration, not a converged yield. Report
   the spread with CFL and resolution. Drop the comparison with the 0.41/0.51 Msun values, or
   keep it only with the caveat that those needed km-scale cells and a resolved shock ignition.
   Restate the limiter result: at 940 km cells dx/f is larger than the star, so a near-zero
   yield with the limiter means the run is unresolved, not that the limiter is wrong.
3. Update `paper/research/measurements.md` with the new numbers. Check the preset blurb ("detonate
   where they meet") and the tooltip against what the runs show.

## 3. The network: silicon burning too fast and dependent on the step

**Status: reproduced.** I ran the shipped network (two-unit edit, half-precision weights,
`train.roll`, temperature following the energy release) against `burner.py`. The script is
`stepdep.py` in the session scratchpad.

| Case | Reference X(Ni56) | Network |
|---|---|---|
| 1e7 g/cc, 3.5e9 K, C/O, at 0.1 s | 0.647 | 0.049 at 1e-5 s steps; 0.801 at 3e-5 s; 0.786 at 1e-4 s; 0.704 at 1e-3 s |
| 2.1e7 g/cc, 4.4e9 K, C/O, at 1.2e-4 s | 0.186 | 0.705 in one step; 0.292 at 1e-6 s steps; 0.65-0.70 at 3e-6 s and longer |

The jump between 1e-5 and 3e-5 s steps is a split between two branches: short steps stall in
silicon and long steps run to nickel. Neither branch matches the reference. Our earlier
evaluation missed this because it scored at 1, 0.3 and 3 ms steps and scored errors at times,
not across step sizes. The weight edit fixed a silicon-sulfur runaway near 4e9 K, but these
states (fresh C/O at 3.5-4.4e9 K, at 10-100 μs steps) are not in its gates' data.

The proton point is also confirmed. `tools/burn/build_net.py` folds (a,p)(p,g) and then
removes `p`, which also drops the p-exit branches of C12+C12, C12+O16 and O16+O16. aprox13
(Timmes 2000) keeps those rates at their full totals. So the reference is not aprox13 for
carbon and oxygen burning, even though the docs and paper call it "like aprox13".

Plan:

1. Add a test that would have caught this: step-size consistency. For a grid of hot C/O and
   Si/S states (1e6.5 to 1e8 g/cc, 3 to 5.5e9 K), roll the network to 1 ms and 0.1 s with steps
   from 1e-6 to 1e-3 s, and compare each with the reference. Put it in `weightprog/paired.py`
   as a fifth regime, "silicon burning across steps", and add a few cases to the unit tests as
   regression bounds.
2. Fix the reference first, because retraining against a reference that will change wastes a
   round. In `build_net.py`, keep the p-exit branches of the three heavy-ion reactions
   aprox13's way. Assume the proton is captured at once and route each branch to the
   alpha-chain product it effectively makes (the Timmes 2000 / aprox13 treatment, e.g. C12+C12
   → Na23+p counted as → Ne20+α at the same Q, or follow pynucastro's aprox13 example if it has
   one). Then check induction times and final compositions against an independent aprox13,
   such as pynucastro's or the Timmes Fortran, on his cases. He found a factor 1.9 for carbon
   and 4.7 for oxygen, so this should close.
3. Regenerate the trajectory data with the corrected reference (`reburn.py` re-labels the
   stored states). Add trajectories that start in his regime at hydro-like steps of 1e-6 to
   1e-4 s.
4. Retrain from 35b with DAgger rounds that include rollouts at several step sizes. Add a loss
   term for step consistency: one step of 2dt should match two steps of dt, which targets the
   branch split directly. Then decide whether the two gated units are still needed or should be
   refitted.
5. If retraining can't remove the step dependence, add a runtime safeguard. Subcycle the
   network in the shader so no substep in the silicon-burning regime exceeds a fixed step, or
   switch to an NSE/QSE relaxation scheme above about 4.5e9 K (Kushnir's own approach). This is
   a fallback, not the first choice.
6. Rerun the tube (front speed, ash), the collision and the sub-Chandra yields with the new
   network, and update the paper's network paragraph (`main.tex:292`), `docs/numerics.md` §20
   and `CHANGELOG.md`. In the paper, say that the reference keeps aprox13's heavy-ion branches
   and report the step-consistency test.

## Order

1 and 2 are text changes after a few GPU runs and can ship soon. 3 needs the reference rebuilt,
the data regenerated and a retrain. Do 3.1 (the test) first, so the current failure is on record
before anything changes. A reply to Doron can go out after 1 and 2, with 3 described as in
progress.

---
name: random-test
description: Run Shaderous's randomized setup testing - setup documents drawn at random from the option table (every option, choice, kind of part, built-in flow and kind of box face it lists, each with its rules met), composed from a box of gas or built on a preset - and triage what it flags, or report which features the seeds reach. Use when asked to fuzz, stress-test or "randomly test" the simulator, after a large change to the solver, the setup compiler or the option rules, or before a release.
---

# Random-setup testing

Three files do the work:

- `tools/random-test/generate.ts` makes the documents, in node (or in a dev-server page). It
  reads the option table (`src/setup/options.ts`): each option set, choice made, group form,
  kind of part, part field, built-in flow and kind of box face is a *feature*. A document is a
  box of gas (odd seeds) or a preset on a random grid (even seeds), plus a few features drawn
  uniformly from the whole list. Applying a feature sets it, meets its `requires` and
  `excludes` through `FIXERS` (switching on the self-gravity a star needs, opening a periodic
  face a point mass cannot have), then takes out whatever the page's check flags elsewhere. A
  feature whose document still fails the check is dropped. So every document passes
  `checkSetup` and the simulators' refusal: what it does when run is the test. Numbers come
  from the table's ranges around its defaults, or from `HINTS` where those are too wide.
- `tools/random-test/random-test.js` runs in the page: `randomTest.runDoc(gen, { steps })`
  loads and runs one and returns a row with its flags.
- `tools/random-test/run.mjs` generates (loading `generate.ts` through Vite in node), opens the
  simulator in Chrome (through `tools/headless/session.mjs`), runs each document, saves
  screenshots and `results.json` (each row holds its document), and prints what to look at.

A new option in the table becomes a target with no change here. `tests/unit/random_setup.test.ts`
checks that every feature can be put into a document: an option it cannot place fails `npm test`
until it gets a hint, a hook or a fixer in `generate.ts` (or an entry in `EXCLUDED` with why).

## Running it

```bash
npm run random-test -- --runs 96 --seed 1000 --steps 400 --out random-test-out
npm run random-test -- --coverage 3000 --seed 1000      # no browser: what the seeds reach
```

- `--shot-all` saves a screenshot of every run's end state (`run-<seed>.png`); `--shot-every K`
  every K-th (default 8). Look at a few: the numbers do not catch a picture that is wrong.
- `--seeds 5001,5017` reruns given seeds; `--mode compose|preset` forces one way.
- `--steps 1000` finds slower failures. `--gpu-memory 4` (the default) keeps runs under about
  4 GB; 3D grids are capped at 256³ on blocks and 128³ uniform.
- The runner serves the page itself with reloads off, so other edits do not kill a batch;
  `--url http://localhost:PORT` runs against a page that is already up. `--headed` shows it.
- A run takes 5 to 25 s; 96 runs about 15 to 20 minutes. Generating takes 60 to 100 ms a
  document (the page's check builds the stars).
- `--coverage N` prints how many of the features N documents reach, how many pairs of the main
  features (choices, groups, kinds of part, flows, faces) they hold together, and the features
  the check most often turned away; `coverage.json` has the counts.

In a dev-server page you are driving yourself (the built-in browser pane), the same works
without the runner: evaluate `random-test.js`, then
`await randomTest.runBatch([...Array(48)].map((_, i) => 1000 + i))` (it imports `generate.ts`
from the source) and read `randomTest.serious(randomTest.rows)`.

## Reading the results

A row has the seed, the preset it started from (if any), its features, the time reached, the
steps, the mass change and its flags. `run.mjs` prints what each seed added.

| Flag | Usually means |
| --- | --- |
| `non-finite` | a real failure: find the step it starts |
| `mass … in a closed box` | a real failure: walls or periodic faces, nothing adding or taking gas (no sink, inflow, tunnel, damping zone or Alfven cap), yet mass changed |
| `errors: …` | the page threw |
| `skipped (…)` | the page refused a document that passed the check in node: the two checks disagree (a bug), or the grid needs more GPU memory than the budget |
| `floored N` | more than 1% of cells hit the density or pressure floor: expected with vacuum around stars, disks or tidal debris; suspicious in a box of ordinary gas |
| `mach …` | Mach above 100: expected in cold gas (p ≈ 1e-8 around stars, cold cosmological flows) |

`run.mjs` exits 1 when a row errored, was skipped, or has a flag other than floors or Mach.

Random combinations are legal but not always sensible (a blast in a liquid, a planet in a
rotating frame): a failure is a bug when the simulator should have refused the setup, or should
have run it. Judge which before fixing.

To reproduce a row: its `doc` is the document (`shaderous.load({}, doc)`), or
`generate(seed).doc` gives it again with the same option table. Then bisect: drop one part or
one physics option at a time until the failure goes, and follow the first bad steps with
`run({ steps: 1 })` and `diagnostics()`.

To find out why a feature is rarely reached, `attempt('<feature id>', seeds)` (exported by
`generate.ts`) tries it on the box-of-gas base of each seed and says what turned it away.

## What it has found

The first night of this testing (November 2026, with an earlier hand-written generator) found
and fixed about a dozen bugs, among them: the tabulated white dwarf equation of state going NaN
on Apple GPUs, swirls emptying the axis cells of (R, z) and spherical runs in one step, cold gas
under a uniform field falling through walls, magnetic fields flowing into obstacles, thin disks
with field loops crashing the page, and setup checks that passed documents the simulators then
refused. Seeds from before the table-driven generator (20 November 2026) make other documents
now. The first 1000-run batch of the table-driven generator (20 November 2026, seeds 10000 to
10999) flagged 26 runs, all fixed: mostly setups the check passed that then failed (flows that
needed units, species, a shearing box or comoving coordinates; a wind tunnel's liquid and its
implicit pressure; gamma 1 outside the barotropic gas; units that carried the equations of
state's constants past 32-bit numbers; face-defined fields in (R, z)), plus MHD on the adaptive
mesh with species that set the equation of state (refused), the magnetic diagnostics' missing
composition binding, a star field without beta, and a radiation readback racing a reload.
Open issues it found are on GitHub (label them `random-test`). When a new failure is real,
fix it, add a validation test that would have caught it (`src/validation/*.ts`, listed in
`tests/gpu/validation.spec.ts`), and rerun the seeds that found it.

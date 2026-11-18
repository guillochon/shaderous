---
name: random-test
description: Run Shaderous's randomized setup testing - random presets with random grid and solver options, and setup documents assembled at random from parts (blasts, blobs, jets, obstacles, stars, point masses, disks, fields, rotating frames, comoving coordinates, shearing boxes, mirrors, per-block timesteps, in Cartesian, (R, z) and (r, theta) geometry) - and triage what it flags. Use when asked to fuzz, stress-test or "randomly test" the simulator, after a large change to the solver, the setup compiler or the option rules, or before a release.
---

# Random-setup testing

Two files do the work:

- `tools/random-test/random-test.js` runs in the page and defines `window.randomTest`:
  `presetSetup(seed)`, `composedSetup(seed)` (returns `{ doc, kinds, geometry, ... }`),
  `runOne(seed, { steps, kind })` and `runBatch(seeds, opts)`. Every setup is a deterministic
  function of its seed.
- `tools/random-test/run.mjs` opens the simulator in Chrome (through
  `tools/headless/session.mjs`, the same launcher as `npm run headless`), injects the script,
  runs a list of seeds, saves screenshots and `results.json`, and prints what to look at.

## Running it

```bash
npm run random-test -- --runs 96 --seed 1000 --steps 400 --out random-test-out
```

- Odd seeds are setup documents, even seeds presets. `--steps 1000` finds slower failures.
- `--gpu-memory 4` (the default) keeps runs under about 4 GB of GPU memory; 3D presets are
  capped at 256³ on blocks and 128³ uniform.
- `--headed` shows the browser; `--shot-every K` saves a screenshot every K runs (look at a
  few: the numbers do not catch a picture that is wrong, such as a mapping or a misplaced part).
- `--url http://localhost:PORT` runs against a page that is already up. When you are editing
  code at the same time, use a frozen build (`npx vite build --outDir /tmp/frozen && npx vite
  preview --outDir /tmp/frozen --port 5197`): a dev server reloads the page on every save and
  kills the batch.
- A run takes 5 to 25 s; 96 runs take about 15 to 20 minutes.

In a browser you are driving yourself (the built-in browser pane, say), the same thing works
without the runner: load the simulator, evaluate the contents of `random-test.js`, then
`await randomTest.runBatch([...Array(48)].map((_, i) => 1000 + i))` and read
`randomTest.serious(randomTest.rows)`.

## Reading the results

A row has the seed, what ran (`query` for a preset, `kinds` and `geometry` for a document),
the time reached, the steps, the mass change, and its flags:

| Flag | Usually means |
| --- | --- |
| `non-finite` | a real failure: find the step it starts |
| `mass … in a closed box` | a real failure: walls or periodic faces and no sink, yet mass changed |
| `errors: …` | the page threw: often a refusal that the setup check should have caught |
| `floored N` | more than 1% of cells hit the density or pressure floor: expected with vacuum around stars, disks or tidal debris; suspicious in a box of ordinary gas |
| `mach …` | Mach above 100: expected in cold gas (p ≈ 1e-8 around stars, cold cosmological flows) |
| `skipped (…)` | the page's own check refused the document: fine, unless the reason looks wrong |

`run.mjs` exits 1 when a row errored or has a flag other than floors or Mach.

To reproduce a row: `randomTest.composedSetup(seed).doc` is the document (load it with
`shaderous.load({}, doc)`), and `randomTest.presetSetup(seed)` the link keys. Then bisect: drop
one part or one physics option at a time (refinement, per-block timesteps, the field, the
equation of state, viscosity) until the failure goes, and follow the first bad steps with
`run({ steps: 1 })` and `diagnostics()`.

## What it has found

The first night of this testing (November 2026) found and fixed about a dozen bugs, among them:
the tabulated white dwarf equation of state going NaN on Apple GPUs, swirls emptying the axis
cells of (R, z) and spherical runs in one step, cold gas under a uniform field falling through
walls, magnetic fields flowing into obstacles, thin disks with field loops crashing the page,
and setup checks that passed documents the simulators then refused. Open issues it found are
on GitHub (label them `random-test`). When a new failure is real, fix it, add a validation
test that would have caught it (`src/validation/*.ts`, listed in
`tests/gpu/validation.spec.ts`), and rerun the seeds that found it.

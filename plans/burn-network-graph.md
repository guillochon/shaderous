# Plan: the burning network as a graph in the query readout

Status (2026-11-27): stages 1-4 built (the graph sits under the readout's equations, collapsed by default). WATCH_REACTIONS is not yet derived from the table: it splits C+O half to Mg24 + a, half to Si28, where the table (from the REACLIB names) has Mg24 + a only.

When a run burns with the network (`physics.burn.network: 'aprox13'`), the query readout gets a
graph of the 13-isotope alpha chain. The nodes are the species. The edges are the reactions
between them, each labelled with its rate in the cell under the pointer (or the pinned cell).
A glance should show what is burning there and which way. Is carbon going to neon and
magnesium? Is the chain climbing to nickel, or is silicon in quasi-equilibrium with the
captures above it?

## What there is to build on

- **The reaction rates are already computed in TypeScript.** `netFeatures()`
  (`src/burn/network.ts`) evaluates every term of the reference's dY/dt as one of the network's
  inputs: 32 terms, each reaction forward and reverse, with REACLIB fits and screen5 screening,
  in float64. `termLogs()` turns them back into ln r (mol/g/s). `netEnergyRate()` already uses
  this for the burning watch. The graph needs no new physics, only these numbers.
- **The terms come in forward/reverse pairs.** `train.FULL_TERMS` lists them as 16 pairs:
  terms 2k and 2k+1 are one reaction forward and backward. The first eight pairs are
  C12(a,g)O16, O16(a,g)Ne20, Ne20(a,g)Mg24, C12+C12, C12+O16, O16+O16, C12+Ne20 and 3a. The
  last eight are the (a,g)+(a,p)(p,g) captures from Mg24 up to Ni56.
- **The terms know their reactants but not their products.** `NET_DATA.terms` carries
  `powers` (reactants), `nrho` and `coef`, but no products. Two terms share the same reactants:
  Ne20(a,g)Mg24 and the reverse of C12+C12, Ne20(a,C12)C12. So products can't be inferred from
  the data that ships. `WATCH_REACTIONS` in `network.ts` gets around this with a hand-written
  table of the eight reactions a He/C/O cell runs, matched by reactants. That breaks down for
  the full set.
- **The query readout** (`src/ui/app.ts`, "query tool"; `src/ui/readout_view.ts`) shows the
  cell's values and the equation terms (`src/ui/terms.ts`). Each species already has a "made or
  used up by burning" row, from `burnRates()` (`src/ui/burn_rates.ts`): the *network's* change
  over one step, not the reference's rates. The readout refreshes on every read when hovering
  and every 150 ms when pinned. A pinned readout records each number as a series with
  sparklines and CSV. `readout_view.ts` updates its DOM in place rather than rebuilding it,
  because rebuilding loses hover state. The graph has to do the same.
- **The importance ramp** (`importance()`, `rampColor()` in `terms.ts`) colours numbers by how
  close they are to the largest, over 6 decades. The equations use it, and the edges should
  too, so the two read the same way.

## Decisions

1. **Edges show the reference's reaction rates, not the network's.** The trained network has
   no per-reaction fluxes: each stage removes a share of every species and redistributes the
   total by a softmax (`flowStage`). The rates on the edges are the reference's instantaneous
   rates at the cell's ρ, T, X. Those are what the network reads as inputs, and they are the
   physically meaningful quantity. The readout already shows the network's actual change per
   species in the species equations, so the two sit side by side. The graph's caption says
   "reaction rates (aprox13 reference, at this cell's state)" so nobody mistakes them for the
   step the code took. The nodes can show both: an outline coloured by the reference's net
   Ẏ_i, which is the sum over edges, and a small mark for the network's (see 6). Where the two
   disagree, that is a diagnostic in itself.

2. **One edge per reaction, net of its reverse.** Each forward/reverse pair is drawn as one
   edge. The arrow points the way the net flux goes, and the label is |r_fwd − r_rev|. Hovering
   the edge gives r_fwd, r_rev, the energy rate Q·r_net, and the reactant's timescale Y / r.
   When |r_fwd − r_rev| < 1% of r_fwd, the edge is drawn dashed and labelled "≈ equilibrium".
   This is what silicon burning and NSE-like states look like, and a net rate alone would hide
   it.

3. **Units: mol g⁻¹ s⁻¹ on the labels**, the reaction's extent rate. This is independent of
   the readout's unit system, the same for every edge, and the standard way to quote it. The
   hover adds the energy rate in the readout's units (erg/g/s in cgs) and the mass rate ρ A r
   in the readout's units for comparison with the species equation rows. (Open question 1
   asks whether to offer energy as the label instead.)

4. **Helium is a node, but not the end of an edge.** Nearly every reaction takes in or gives
   off an alpha. Drawing each as an edge to he4 would bury the graph in spokes. Captures are
   drawn heavy reactant → heavy product, labelled "+α". Alpha exits are written into the label
   ("C+C → Ne + α"). The he4 node shows its net Ẏ from all of them. Triple alpha is the one
   edge out of he4 (he4 → c12).

5. **Heavy-ion reactions are hyperedges.** C+O, C+Ne and the pairs drawn with them have two
   heavy reactants. Each reactant draws a line into a small junction dot, and one arrow leaves
   the dot for the product, carrying the label. C+C and O+O have a single reactant, so they are
   ordinary arcs that skip along the chain.

6. **Nodes:** fill opacity follows log X (from 1e-6 to 1). The outline colour follows the sign
   of the reference's net Ẏ (made or used up), and its width follows the ramp. A species the
   cell doesn't hold (X < 1e-12) is drawn faint, but stays in place so the layout doesn't jump.

7. **Fixed layout, built once per run.** The 13 nuclei snake in two rows: he4 c12 o16 ne20
   mg24 si28 along the top, s32 … ni56 back along the bottom. Heavy-ion arcs go above the top
   row. Triple alpha goes from he4 down to c12. The graph is an SVG with one `<path>` and one
   label per edge and one `<g>` per node, made when the readout first shows it for a run. Each
   read only changes attributes: stroke width, colour, arrow direction, dash and label text.
   The graph matches the readout's width (about 300 px). Stage 1 tunes the exact geometry.

8. **Where nothing burns, say why.** The pass skips cells below `tMin` (default 1e8 K) or
   `rhoMin` (1e4 g/cm³). There the graph still shows the rates, greyed out, with a line saying
   "not burning: T < 1e8 K". The rates are still informative (how close is ignition?), but the
   code doesn't apply them. A run that only watches (he4, c12, o16 carried) shows the graph with
   the other ten species at zero, so only the eight He/C/O reactions light up. It is captioned
   "watching". This is exactly what the watch measures.

9. **The products come from data, exported with the network.** A table of each term's
   products (`nu`: species → change in count) and its label is generated from
   `train.FULL_TERMS` by `tools/burn/export_net.py`. That way the graph follows a retrain. The
   pending retrain in `plans/kushnir-feedback.md` point 3 changes the terms by folding in the
   heavy-ion proton exits. Hand-writing the table would go stale silently. Because the shipped
   weights must not change, the table goes in a file of its own, `src/burn/aprox13_reactions.ts`,
   and `export_net.py` gets a `--reactions` mode that writes only it, without loading a model or
   torch weights. The model file and venv are not in every checkout. If the Python environment
   isn't at hand, the file can be written once by hand from `FULL_TERMS`, with the test below as
   the guard. Each entry is `{ nu: [[i, n], …], label: 'C12(a,g)O16', pair: k, dir: 'fwd' | 'rev' }`.
   `WATCH_REACTIONS` could then be derived from the same table (optional cleanup, stage 4).

## What to build

- `src/burn/aprox13_reactions.ts` (generated, decision 9).
- `src/burn/network.ts`: export `netReactionRates(logRho, logT, X)`, which returns the 32 term
  rates r (mol/g/s) as `termLogs` computes them. `termLogs` stays private and `netEnergyRate`
  goes through the new function.
- `src/ui/burn_graph.ts` (new): the pure part and the view.
  - `reactionEdges(rates)`: pairs the terms into 16 edges `{ pair, fwd, rev, net, dir, eq, Q }`,
    plus each species' net Ẏ_i = Σ nu_i r. No DOM, unit-tested.
  - `class BurnGraphView`: builds the SVG once (decision 7) and has `update(edges, nodes, state)`
    to change it in place, plus hover on an edge or node for the details popup.
- `src/ui/app.ts`, `showQuery()`: when `isNetworkBurn(burnSpec(cfg))` and the run carries the
  network's species (all 13, or the watched three padded with zeros), compute the rates from
  the probed cell's ρ, T, X (converted with `netUnits`, as `burnRates` does) and update the
  graph. A "Network" toggle in the readout's title row shows or hides it. It starts open for a
  network run, and the choice is kept per viewer in localStorage. With the graph shown, the
  readout lays the dl and equations on one side and the graph below them (or beside them, if
  there is room in the view, as the profile panel does).
- `src/ui/style.css`: `#cell-pop .burn-graph` styles, with the existing ramp colours.

## Stages

1. **Data and numbers.** `aprox13_reactions.ts`, `netReactionRates`, `reactionEdges`. Unit
   tests (`tests/unit/burn_graph.test.ts`):
   - the table lines up with `NET_DATA.terms`: same length, each term's reactants (`powers`)
     equal the negative entries of its `nu`, each reaction conserves nucleons (Σ A_i nu_i = 0)
     and charge, and each pair's reverse is the forward's `nu` negated;
   - Σ_t nu_i r_t at a few states (the cases in `aprox13_net_cases.json`) matches the
     reference's dY/dt. `export_net.py` would need to add the reference's `rhs` to the cases;
     until then, check against an independent sum written out in the test for a He/C/O cell;
   - `netEnergyRate` is unchanged (its existing test passes) once routed through the new
     function;
   - equilibrium flagging, sign and direction on hand-made rates.
2. **The graph in the readout, static layout.** `BurnGraphView`, the toggle, hover details.
   Verify in the browser pane on `detonation-co` and `wd-collision-burning`: hover the
   detonation front and behind it, check the edges move from C+C / C+O to the captures up to
   Ni56, and check nothing is rebuilt per read (hover stays open while the readout refreshes).
   Check the greyed state in cold gas and the "watching" state on a preset that watches. Take
   screenshots in light and dark.
3. **Pinned series.** While pinned, each edge's net rate is recorded as a series in
   `queryView`, like the readout's other numbers. Hovering an edge shows its sparkline popup,
   and the CSV gets a column per reaction (`r:C12(a,g)O16`). The node series are already the
   species' X.
4. **API and MCP.** `probe(…, { reactions: true })` in `window.shaderous`, passed through by
   MCP `probe`, returns `[{ label, fwd, rev, net, energy }]` for the cell. That lets an agent
   or a test read the rates without the page. Optional: derive `WATCH_REACTIONS` from the
   table.

## Not covered

- Screenshots don't draw the readout (it's HTML over the canvas), and so they won't draw the
  graph either. Same limitation as the profile line.
- A field of reaction rates over the whole grid (for example, "colour by C+C rate") would be a
  render field computed on the GPU. That is a separate plan if it's wanted.
- Burning by the one-step fuel → ash model (`BurnSpec`, not the network) has a single
  reaction, so it gets no graph. Its species equation rows already say everything.

## Open questions

1. Labels in mol/g/s (decision 3), or a switch to energy rate (erg/g/s), which shows directly
   which reactions heat the gas?
2. Should the graph also be offered outside the query readout, as a panel of its own that
   follows the pinned cell, like the profile panel? Stage 2 would show whether the readout gets
   too tall with it.
3. Once the retrain folds in the proton exits (Kushnir point 3), C+C, C+O and O+O each have two
   exits. Should those be one edge with a split label, or two edges to different products?
   Decision 9 makes either a change to the table, not to the view.

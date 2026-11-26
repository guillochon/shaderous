# Plan: a profile tool — a field's value along a line

Status (2026-11-26): stages 1 to 3 built; stage 4 (runner `--profile`, the link, several
profiles, exact cell crossings) not started.

**As built.**

- GPU pass: `profile` in `render/project.wgsl`, shared by both field shaders, each of which gives
  `profile_cell()` (the cell under a point; the finest leaf on blocks). `Renderer.profile()`
  compiles it the first time it is asked for, with the code of the run attached (not of the last
  one asked for: `fieldPipelines` runs ahead of a load, and a profile read during a load once
  paired the new code with the old run's resources). It matches `probe` to float precision for
  density, pressure, velocity, temperature and level on 2D blocks, 3D uniform and 3D blocks.
- Volume view: both ends lie on the plane facing the camera through the point orbited, not on the
  gas under the pointer as planned. In a blast in a uniform medium the gas "seen" under the
  pointer is wherever the fog's opacity reaches the pick's threshold, so a line meant to go
  through the center missed it. Alt puts an end on the gas.
- The plot's x axis is distance in the units shown; AMR level changes are marked on its bottom
  edge; the field menu is the view's own, with what the run cannot show left out (the API
  refuses those too). The panel goes beside the line inside the view area: right of it, left,
  below or above, the first that fits.
- Not done: screenshots do not draw the line (it is an SVG over the canvas); `view({ profile })`
  became `profile({ show: true })`; mirror images shown: the ends fold into the computed part,
  so a line across a mirror plane is not the line drawn.

The person picks the Profile tool, presses at a start point in the box, and a line follows the
pointer until they place the end. A panel then opens beside the line with a plot of one field
against distance along it, and a menu to choose the field. The same profile can be asked for
through the scripting API, the MCP server and the command-line runner, on a run in progress.

## What is there to build on

- **Pointer to box.** `locate()` / `locateInView()` in `src/ui/app.ts` map a canvas point to a
  point of the box in either view: the slice or projection plane (2D runs always; 3D in slice
  or projection mode), or, in the volume view, a point on the camera-facing plane at a given
  depth. `boxPointAt()` gets that depth from `renderer.pick()` (the gas seen under the pointer).
  Spherical runs (`sphFromView`) and mirror images (`mirrorComputed`) are already folded in.
- **Drawing over the view.** The `#brush` SVG (`index.html`, `drawBrush()` in `app.ts`) already
  draws obstacle outlines and the projected 3D orbit paths of point masses being placed, re-drawn
  as the camera moves. The profile line and its end handles go there.
- **Popups.** `#cell-pop` (query readout, placed beside the pointer, pinnable) and `#diag-pop`
  (a line plot of a diagnostic, `drawFull()` in `src/ui/diag_plot.ts`). The profile panel is a
  third of these, with its own plot code modelled on `drawFull` (axes, ticks, `fmt3`).
- **Field values.** `FIELDS` / `FIELD_MENU` in `src/problems/types.ts` are what the view can
  show (37 of them, plus one mass fraction per species). `field_value()` in
  `src/render/field_value.wgsl` computes each one's raw value from the state, gradients, level
  and potential; `render/field.wgsl` (uniform grid) and `amr/field.wgsl` (blocks) normalize it
  for display and hold the `plane()` and projection `column()` readers.
- **API sampling.** `slice()` and `probe()` in `src/ui/api.ts` read the whole state back
  (`capture()`), then sample on the CPU with `SnapshotSampler` via `variableOf()` — only
  `rho, press, vel1-3, r0, level, temp, tempK` and species. That is a different, smaller
  vocabulary than the view's fields, and a whole-state readback per call.
- **Units.** `src/ui/units_view.ts` (`fieldMeasure`, `readout`) turns a field's code value into
  the displayed units, as the color bar and query readout do.

## Decisions

1. **Sample on the GPU with the view's own field code, not with `SnapshotSampler`.** Add a
   `profile` compute entry point to both `render/field.wgsl` and `amr/field.wgsl`: it reads N
   points (box fractions) from a storage buffer and writes `field_value()` at each one, raw
   (not normalized). Why: every field the view offers (vorticity, Mach number, plasma beta,
   radiation temperature, …) comes for free and agrees exactly with what is on screen; and it
   reads back N floats instead of the whole state, so it can refresh live on a large 3D AMR run.
   Fields that need gradients use the same neighbour reads the display already does.
2. **Nearest cell, finest leaf, no interpolation.** Each sample takes the cell of the finest
   leaf that holds it (what the slice view and `probe` show). The plot is then honest about
   resolution: steps at cell edges, AMR level changes visible. Default sample count is about
   two per finest cell along the line (`ceil(2 L / dx_min)`), held to 64…4096.
   A later option: exact cell crossings (a DDA through the leaves) returning one value per cell
   with its entry and exit distance.
3. **Sample what the view shows.** In slice mode the line lies in the slice plane; in a 2D run,
   in the plane. In projection mode the samples are the projected quantity (`column()` at each
   point of the line on the map), labelled as such. In the volume view the line is a 3D segment
   through the gas.
4. **One vocabulary for the API:** field names as `view({ field })` takes them (name or index
   into `describe().fields`), plus species names for mass fractions, plus the existing
   `variableOf` names (`rho`, `press`, `vel1`, `temp`, `tempK`, `level`, …) mapped onto their
   field indices so scripts written against `slice` keep working.
5. **One profile at a time** in the page (stage 2). Several, each with its own colour and panel,
   is a later stage if wanted.
6. **Values in code units from the API, with the units named;** the panel shows the displayed
   units (as the color bar does). The API result carries `units` and, where the run has physical
   units, a `scale` to convert, like `tempK`.

## The interaction

**Placing (2D runs, 3D slice and projection).** With the Profile tool in hand (a new button in
the tool rail beside Query):

- Press, drag, release: start at the press, end at the release.
- Or click, move, click: a press released where it began (< 5 px, as the query pin does) starts
  a line that follows the pointer until the next click.
- Shift holds the line to multiples of 15° (as obstacle shapes do with modifiers) — most
  profiles people want are along an axis or a diagonal.
- Esc or right click cancels a line being placed. Panning and zooming keep working with the
  right/middle button and the wheel while placing.
- The endpoints are clipped to the box; a line that never enters the box is refused.

**Placing in the volume view.** The start is the gas under the pointer (`renderer.pick`); with
no gas there, the camera-facing plane through the box centre. While the end is placed it moves
on the camera-facing plane through the start, so it does not jump about as density changes under
the pointer; holding Alt snaps it to the gas under the pointer instead (one pick in flight at a
time, as the query tool reads). Orbiting while placing is not allowed (the left button is taken);
the right button pans and the wheel zooms.

**After placing.** The line stays drawn with a small handle at each end and tick marks every
tenth of its length; it is re-projected as the view moves. Dragging a handle moves that end (in
the plane, or on the camera-facing plane through that end in 3D) and the plot follows. The
profile stays when another tool is picked up, until its panel is closed (as a pinned query
readout stays). A new line with the Profile tool replaces it.

**The panel.** Opens beside the line's end, on whichever side has more room, kept on screen and
clear of the rail (the `cell-pop` placement rules); draggable by its title; on a phone it docks
as a sheet at the bottom. It holds:

- a field menu: `FIELD_MENU`'s groups, with the fields this run cannot show left out (no
  magnetic fields without MHD, …) and the species' mass fractions; it starts on the field
  the view shows;
- the plot: value against distance from the start, in code length or the run's physical length
  unit; log y when the field is shown in log in the view (a toggle beside the menu); AMR level
  changes marked faintly on the x axis;
- a readout of the start, end and length;
- hovering the plot shows the value there and moves a marker along the line in the view, and
  hovering the line in the view does the reverse;
- buttons: swap ends, copy as CSV, save CSV, close.

While the run plays, the profile is sampled again at most every 250 ms with one read in flight
(the query tool's rule), and the y range eases rather than jumps.

## The API, MCP and runner

**`window.shaderous.profile({ from, to, field?, n?, encoding? })`**, `serial` like `slice`:

```
from, to   [x, y, z] in fractions of the box ([x, y] in 2D)
field      name or index (decision 4); default the field the view shows
n          samples, 2…16384; default as decision 2
encoding   'json' | 'base64' (EncodedArray, as slice)
→ { t, from, to, length, field, units, scale?, n, s: [...], values: [...], level: [...],
    min, max }
```

`s` is distance from `from` in code length. `level` is the refinement level of each sample (free
from the same pass). With no arguments it returns the profile placed in the page (its ends and
field), or an error if there is none, so an agent can read what the person drew.
`view({ profile: { from, to, field } | null })` places or clears one in the page, so a screenshot
can show it; it is not an action in the replay log (it does not change the run).

**MCP tool `profile`** in `src/ui/mcp_schema.ts`: the same arguments, plus `save_path` (`.npy`
float32 of shape `[n]`, `.csv` with `s,value,level`, else JSON) and values inlined when n ≤ 256,
as `slice` does. Works on a session's run whatever its state (running or paused: it queues behind
`run` like every serial call; a call made while `run` is under way waits for that call's end —
if that is too late for agents watching long runs, the follow-up is a non-serial read path,
see open questions).

**Runner** (`tools/headless/cli.mjs`): `--profile NAME:x0,y0,z0:x1,y1,z1[:FIELD[,FIELD..]]`,
repeatable, writes `NNNN-tT-profile-NAME.json` (or `.npy` / `.csv` with `--profile-format`) at
each output, like `--slice`.

**View-only runs** (`viewAthdf`): only the held field is on the GPU, so the profile offers that
field, and the primitives through `viewProbe` on the CPU; the menu says so.

Schema and docs to update together: `src/ui/api_schema.ts`, `src/ui/mcp_schema.ts`,
`tools/headless/schema.json`, `docs/api.md` (a `profile` section after `probe`),
`docs/headless.md` (outputs table and options), `.claude/skills/shaderous-sim/SKILL.md` and
`plugins/shaderous/skills/simulate/SKILL.md`.

## Stages

1. **Engine and API.** The `profile` entry points in both field shaders (uniform grid and
   blocks, projection variant), a `Renderer.profile(points, field)` returning raw values and
   levels, the name→index mapping (decision 4), `api.profile`, the MCP tool, the schemas and
   docs. This alone answers the "on running jobs" half and is the easiest to verify.
   Tests: vitest for the pure parts (clipping a segment to the box, sample spacing, the field
   name mapping, CSV/npy writing); a GPU test (static build, per memory) that a profile of
   density, pressure and velocity matches `probe()` at the same points to float precision on a
   uniform 2D run and on a 3D AMR run, and that a profile across a Sod tube shows the shock where
   a `slice` row does. An MCP selftest call.
2. **The tool in 2D and slice views.** Tool button, placing (both gestures, Shift, Esc), the line
   and handles in `#brush`, the panel with the field menu, the plot, live refresh and CSV.
   Verify in the page with the dev server (screenshots of placing and of the panel), and on a
   phone-width window.
3. **The volume view and projections.** 3D placement (decision on depth above), handles on
   camera-facing planes, the line re-projected while orbiting, the projection-mode label;
   hover sync between plot and line.
4. **Extras, if wanted.** Runner `--profile`; the profile in the shareable link; several
   profiles at once; exact cell crossings (decision 2).

## Open questions

- **Interpolated values?** Decision 2 picks nearest cell. A smooth (trilinear) option would make
  prettier plots of smooth flows but hides resolution; recommend leaving it out unless asked.
- **A non-blocking read for long runs.** `serial` makes a `profile` call wait for a running
  `run` call to return. If agents need profiles *during* one long `run` call, the profile pass
  could be enqueued between steps instead (it only reads); worth doing only if that use appears.
- **Spherical runs.** The slice there is the (R, z) half-plane; a straight line on screen is a
  curve in (r, θ) cells. Sampling it point by point through `sphFromView` works; the distance
  axis is then the on-screen (R, z) distance. Fine as is, but say so in the panel.
- **Mirror images.** Samples on the mirrored side fold back into the computed part; the
  components normal to the mirror plane (velocity, field) must flip sign there, as the display
  does. Check the display's rule and reuse it.

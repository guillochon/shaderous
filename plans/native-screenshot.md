# Plan: a screenshot at the grid's own resolution

Status: built 2026-11-17, sections 1 to 8 as written. The open questions of section 11 went the
proposed way: margins outside the map, the button in the Movie section (now "Movie and
screenshot"), the axes under the color bar in the right gutter. "As the view" takes the part of
the box left of the rail. Tried at 8192² (KH, 5 levels): a 9917 × 9078 PNG, 7.7 s to encode.

## 1. Goal

A Screenshot button that saves a PNG of the plane on view in which one cell of the finest
level is exactly one pixel, with the color bar, the axes and the status line drawn beside the
map, not over it, so no cell is covered. The same through `window.shaderous`, the headless CLI
and the MCP tools.

It applies wherever the page shows a plane: a 2D run, and the slice and projection views of a
3D run. In the volume view there is no cell-to-pixel mapping, so the button there saves the
view at its size on the page with the overlays over it, as a movie frame is drawn.

### Why it is cheap

Most of it exists:

- The slice and projection passes already write a map with one texel per cell of the finest
  level, on a uniform or an adaptive grid ([renderer.ts:474](../src/render/renderer.ts)).
- `render/slice.wgsl` reads that map with `textureLoad`, without interpolation. Drawn so that
  the slice spans as many pixels as the map has texels, each pixel is one texel.
- `renderer.fixedSize` and `renderer.slice` already let a draw choose its size and the slice's
  span ([app.ts:4204](../src/ui/app.ts)), and `screenshot()` already draws, reads the canvas in
  the same task and paints overlays ([api.ts:529](../src/ui/api.ts)).
- `render/overlays.ts` has the color bar's ticks, the triad's arrows and the status line.

What is new: a draw whose span is the map's side, a layout that puts the overlays in margins,
a button, and saving a large PNG without a data URL.

### Non-goals for v1

- The DOM markers (sources, sinks, the brush, the orbit preview) in the picture: as in movies,
  only what the renderer draws is captured.
- More than one pixel per cell (an integer scale for small grids): section 9.
- Tiling a map larger than a texture or a canvas can be: section 7.

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| How the map is drawn | The renderer's own slice pass, at `fixedSize` = the region in cells, with the span set so a texel is a pixel | Contours, block outlines, solids and flat field lines come out as on the page. Reading the map back and coloring it on the CPU would be a second renderer to keep in step |
| Which region | The whole box by default; the part in view (zoom and pan) as an option | "The picture of the run" is the usual want, and it does not depend on the window's shape |
| Where the overlays go | In margins added around the map: the status line in a strip above, the color bar and the axes in a gutter to its right | Nothing may cover a cell. The output is larger than the map by the margins |
| Overlay sizes | From a reference length `max(map height, 600)` | `paintOverlays` scales with the frame, so a 128-cell map would get unreadable 10 px labels |
| Line widths in the map | Unchanged: contours and block outlines are about a pixel wide whatever the size | They are drawn per pixel already |
| Saving | `canvas.toBlob()` and `download()` for the button; a data URL only through the API | A data URL of an 8192² PNG is tens of MB of base64 |
| Device pixel ratio | Ignored | The picture is in cells, not in screen pixels |
| Checkboxes | The Movie section's Color bar, Axes and Status | One set of choices for what a capture carries |

## 3. Drawing the map

In `fs` of `slice.wgsl` a pixel's place on the slice is

```
st = (uv * frame / (2 * span)) / zoom + (cx, cy)
```

so a frame `w` x `h` pixels with `span * zoom = side` (the map's side in texels) puts `side`
pixels across the box. Pixel centers then fall on texel centers when `cx * side - w / 2` and
`cy * side - h / 2` are whole numbers.

- **Whole box**: `w = h = side`, `cx = cy = 0.5`, zoom 1, span `side`.
- **Part in view**: the rectangle the page shows (from `sliceLayout()`, the canvas's size,
  `view.zoom`, `view.cx`, `view.cy`), clipped to the box, its edges rounded outward to cell
  edges. `w` and `h` are its size in cells and `cx`, `cy` its middle.

`app.draw()` gains a fourth form beside the page's, a sized screenshot's and a movie's:

```ts
/** One pixel per texel of the slice's map: the box, or the part of it in view. */
drawNative(region: 'box' | 'view', capture: boolean): { width: number; height: number; side: number; n: number }
```

It sets `renderer.fixedSize = [w, h]`, `renderer.slice = { span: side, shift: 0 }`, and
encodes with a copy of the view, `{ ...view, zoom: 1, cx, cy }`, so the page's view is never
written. `drawFresh()` takes the draw as a function so the range logic (draw twice when the
range comes from the display pass) is shared.

Points to respect:

- **`side` may be less than `n`.** The map is capped at `maxTextureDimension2D`
  (typically 8192 or 16384). The result reports both, and the button's notice says when the
  picture is coarser than the grid.
- **An adaptive mesh** shows a coarse cell as a square of `2^k` identical pixels: native means
  the finest level.
- **The page's canvas** is resized for the draw. As `setRecording()` does, the canvas gets
  `object-fit: contain` and no pointer events until `draw(null)` restores it in a `finally`.
  With the headless read-back path (`renderer.capture`) the frame comes from the capture
  texture, as now.
- **Hover state**: `renderer.hoverLevel` is cleared for the draw, so a contour under the
  pointer is not drawn highlighted.
- **To check when building**: that rows are not off by one at the box's edges (`st >= 1` is
  background), and the picture's y direction against `slice()`.

## 4. The overlays beside the map

A new function in `render/overlays.ts`, beside `paintOverlays`:

```ts
interface BesideLayout {
  width: number; height: number;                  // the whole picture
  map: { x: number; y: number };                  // where the map's first pixel goes
  bar: { x: number; y: number; w: number; h: number } | null;
  triad: { x: number; y: number } | null;         // its origin
  status: { x: number; y: number } | null;
  px: number;                                     // text size
}
/** Where everything goes, from the map's size and the widths of the labels (pure: unit tested). */
function besideLayout(mapW: number, mapH: number, widths: { ticks: number; title: number; status: number }, opts: OverlayOptions): BesideLayout
/** A new canvas: the background, the map at whole pixels, and the overlays in its margins. */
function composeBeside(map: CanvasImageSource, mapW: number, mapH: number, info: OverlayInfo, opts: OverlayOptions): HTMLCanvasElement
```

Layout, with `s = max(mapH, 600)` and `px = round(0.024 * s)`:

```
+--------------------------------------+-----------+
| status line                          |           |
+--------------------------------------+  title    |
|                                      |  |█| 10²  |
|                                      |  |█|      |
|               map                    |  |█| 10   |
|          (w x h cells)               |  |█|      |
|                                      |  |█| 1    |
|                                      |           |
|                                      |  y        |
|                                      |  └ x      |
+--------------------------------------+-----------+
```

- **Top strip** (only with Status on): one line of text, left-aligned with the map, cut at
  a `•` separator if it is wider than the picture, as `paintStatus` cuts it.
- **Right gutter**: the color bar from the top of the map down, with its title above and the
  ticks and labels to the right of the bar (over a frame they are to its left); the axes in
  the gutter's lower corner, bottom-aligned with the map. The gutter is as wide as the widest
  of title, bar plus labels, and triad, plus padding.
- **Map shorter than the gutter needs** (a small grid): the picture is as tall as the bar and
  triad need, and the map sits at the top of its column.
- **Background**: the view's (`BG_LIGHT` or `BG_DARK`), with the same ink as now. The text
  needs no contrasting edge on a flat background, but keeping it costs nothing and keeps one
  `label()`.
- With all three off, there are no margins: the PNG is the map alone, `w` x `h`.

`paintColorbar`, `paintTriad` and `paintStatus` are refactored to take a place (a rectangle
or an origin, a text size, and which side the tick labels go) from their caller.
`paintOverlays` computes the places it uses today, so movie frames do not change: checked by
comparing a frame before and after.

## 5. API

In `src/ui/api.ts` and `docs/api.md`:

```ts
screenshot({
  width?, height?,                       // as now
  native?: boolean | 'box' | 'view',     // one pixel per finest cell; true is 'box'
  overlays?: { colorbar?, triad?, status? },
  layout?: 'over' | 'beside',            // default 'beside' with native, 'over' otherwise
}) → string                              // a PNG data URL, as now
```

- `native` with `width` or `height` is an error. In the volume view of a 3D run `native` is
  an error through the API ("the volume view has no pixel per cell: use slice or projection").
- `layout: 'beside'` also works for a sized screenshot of a plane or the volume, the frame
  taking the map's place: cheap once `composeBeside` exists.
- A picture too large for a canvas (section 7) is an error that names the size.
- Inside, `screenshot()` is split: `shoot(opts) → { canvas, width, height, side, n, t }`, which
  the API turns into a data URL and the button into a blob.

Headless: `--native` beside `--png` in `tools/headless/cli.mjs` (not with `--width` or
`--height`), `native` and `layout` arguments on the MCP `screenshot` tool, a line in
`selftest.mjs`, and `docs/headless.md`.

## 6. Interface

- A **Screenshot** button in the Movie section, whose summary becomes "Movie and screenshot",
  on the row of the Color bar, Axes and Status checkboxes it shares with Record. Its title:
  "Save the view as a PNG. A slice, a projection or a 2D run is saved at the grid's own
  resolution: one pixel per cell of the finest level, with the color bar and axes beside it."
- A select beside it, **Region**: Whole box, or As the view (shown only in a flat view).
- Enabled paused or running. A running run is paused for the shot through the API's `serial()`
  queue and resumed; while recording, the button is disabled.
- After the save, a short notice with the size: "Saved 2048 × 2048 cells (2310 × 2104 px)",
  and "the map is capped at 8192: 2 cells per pixel" when `side < n`.
- File name as the movie's: `shaderous-<problem>-<ndim>d-<n>-t<time>.png`, through
  `download()` from `io/checkpoints.ts`.
- Not part of a shared link.

## 7. Limits

- **Texture size.** The frame is a render target, so each side is at most
  `maxTextureDimension2D`; the map already is.
- **Canvas size.** Chrome allows 16384 a side and about 268 million pixels; Safari less. The
  composed canvas (map plus margins) is checked against a conservative cap (16384 a side,
  2^28 pixels) and by reading back `canvas.width` after setting it; over it, the error says
  so and suggests As the view.
- **Memory.** A map of N² cells costs 4 N² bytes on the WebGPU canvas, the same again on the
  2D canvas, and the PNG while it is encoded: about 0.6 GB in passing at 8192², next to the
  run's own memory. If that fails in practice on the largest grids, the fallback is to draw
  and compose in horizontal bands, which needs a PNG encoder that takes rows (section 9).
- **Time.** `toBlob()` on a large canvas takes seconds; the button shows "Saving…" meanwhile.

## 8. Tests

- **Unit** (`src/render/overlays.test.ts`): `besideLayout()` for a large map, a small one
  (taller gutter than map), each overlay off, all off (no margins); no rectangle overlaps the
  map's; everything inside the picture.
- **GPU** (`tests/gpu/api.spec.ts`):
  - a 2D uniform run at 64²: `native` with overlays off is 64 × 64 pixels; with a linear
    grayscale colormap and a fixed range, each pixel's gray matches `slice()`'s value for
    that cell, in the right row and column;
  - the same with overlays on: larger than 64 × 64, and the pixels at the layout's map
    rectangle equal the picture without overlays;
  - an adaptive run with two levels: a coarse cell is a 4 × 4 square of equal pixels;
  - `'view'` at zoom 4 about an off-center point: the size in cells, and its pixels equal the
    matching crop of the whole box;
  - a movie frame's overlays before and after the refactor (a sized screenshot with overlays,
    compared with one saved from before).
- **By eye**: Kelvin-Helmholtz at 2048² with contours and block outlines, a slice and a
  projection of a 3D run, light and dark backgrounds, a 64² grid (the small-map layout), the
  largest 2D grid the machine holds.

## 9. Order of work

1. `drawNative()` and `screenshot({ native })` without overlays; the pixel-for-cell GPU test.
2. The overlay refactor (places passed in), `besideLayout()` with unit tests,
   `composeBeside()`, `layout: 'beside'`.
3. The button, the region select, the notice, `toBlob()` saving.
4. The size checks of section 7, tried on the largest grids.
5. Headless (`--native`, the MCP arguments), `docs/api.md`, `docs/headless.md`, a README line.

Step 1 is usable alone from the console; each step leaves the page working.

## 10. Later, if wanted

- An integer scale (2, 3, 4 pixels per cell) for small grids, still without interpolation.
- Banded drawing for maps past the canvas limit.
- The source and sink markers painted from their data onto the composed canvas.
- A length scale bar in the displayed units beside the axes.
- Native-resolution movie frames (`record({ native })`) for grids that fit a video frame.

## 11. Open questions

- **"Beside"**: read here as margins outside the map. Movie frames today paint the overlays
  over the frame's edges; on a wide frame they happen to fall outside the square slice. If
  the overlays should sit over the map as in a movie, `layout: 'over'` is that and becomes
  the default.
- **Where the button goes**: the Movie section (proposed), or the top row beside Record,
  which already holds six buttons.
- **Axes in the right gutter** under the color bar (proposed, the narrowest picture), or a
  left gutter of their own so they sit at the lower left as in movies.

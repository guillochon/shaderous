# Plan: MP4 export of a run

Status: built 2026-11-14 (steps 1 to 6). Decisions made while building:

- **Muxer**: Mediabunny (MPL-2.0), pinned; `mp4-muxer` is deprecated in its favor. Its
  `CanvasSource` encodes from the compositing canvas and applies the encoder's backpressure.
- **Every nth frame** is read as every nth simulation step (section 3), as proposed.
- **Checkpoints**: the API's stepping loop writes a checkpoint the clock is held at and arms
  the next, as the page's own loop does when it plays, so recordings (and `run()`) go through
  them (section 4).
- **The soft stop** lives in `P.refine.z`, a spare slot of the params block, so the block's
  layout is unchanged; 1e30 means none. Uniform and adaptive runs (with local timesteps) share
  `hydro/clock.wgsl`, so one rule covers both. The GPU test confirms a `dt` recording leaves
  both bit for bit unchanged.
- **Defaults**: the `dt` cadence at a 600th of each problem's dynamical time (`Problem.tDyn`, or
  its end time or ten checkpoint intervals), played at 60 fps, so a dynamical time lasts 10 s;
  and the size of the view area as the page draws it.

## 1. Goal

Record a run as an MP4 and download it, from a Record button in the page and from
`window.shaderous` (so the headless CLI and the MCP tools get it too). Each movie frame is the
view of the gas with a color bar and an axis triad drawn over it.

### Non-goals for v1

Audio, WebM or GIF output, the DOM interface (panel, HUD, source markers) in the picture,
recording at the display's own rate, and moving the camera with the pointer while recording
(view changes through the panel and the API still work, and are recorded).

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| What is kept in memory | Encoded H.264 chunks, not images | A raw 1080p frame is 8 MB; 10 s at 60 fps is 5 GB. Encoded, the same clip is tens of MB |
| Encoder | WebCodecs `VideoEncoder`, one frame at a time as it is drawn | Hardware encode, in every browser that has WebGPU; no wasm ffmpeg |
| Container | A small MP4 muxer library writing to an `ArrayBuffer` | The one new dependency. `mp4-muxer` or its successor Mediabunny: check which is maintained before choosing |
| Frame cadence | By simulation steps or by simulation time, never by wall clock | A slow run still gives a smooth movie, and the same setup gives the same movie |
| Every nth frame | `every: n` keeps one movie frame per n steps | See section 3 |
| Who steps the run | The recorder, with the API's stepping loop | `run()` already takes an exact number of steps ([api.ts:188](../src/ui/api.ts)); the `dt` cadence adds a soft stop to it |
| Landing on frame times | Never: no step is shortened; the frame is recorded at the nearest step boundary | The recording must not change the run (section 3.1) |
| Frame size | A size of the recording's own (default 1920 × 1080), not the canvas's | H.264 needs even sides and a level that fits; the page's layout should not set the movie's shape |
| Overlays | Drawn with 2D canvas calls onto a compositing canvas, over a copy of the WebGPU frame | The page's color bar is DOM, which a canvas capture does not see; there is no triad today |
| `MediaRecorder` on `captureStream()` | Not used | Records in real time: drops frames when the run stalls, and bakes in the wall-clock step rate |

## 3. Frame cadence, and every nth frame

One of two cadences per recording:

- **`every: n`** (steps). A frame after every n-th step: `every: 1` records every step,
  `every: 10` every tenth. This is the "only every nth frame" setting. Frames are evenly spaced
  in steps, so not in simulation time when the timestep changes.
- **`dt: x`** (code time). A frame near each multiple of `x`, at the nearest step boundary (section 3.1),
  so the recording does not change the run. Evenly spaced in simulation time to within
  half a step.

### 3.1 The `dt` cadence never changes a step

Landing on a frame time exactly would mean shortening the step that reaches it, and that
changes the run. So a frame time is a soft stop, unlike a checkpoint or the API's `until`: no
step is ever shortened for it, and the frame is recorded at whichever step boundary is nearest.

- Let `dt` be the step the clock is about to take, and `remaining` the time left to the next
  frame time (negative once the clock has passed it).
- `remaining >= dt / 2`: take the full step.
- `remaining < dt / 2`: this boundary is the nearest one. Record the frame now, then go on with
  the same full step.

Consequences:

- A frame's true time is within half a step of its nominal time, before or after. Movie
  timestamps stay at `i / fps`; a time stamp overlay, if added, shows the true time.
- When the frame interval is shorter than a step, a step boundary can be nearest to several
  frame times. It records one frame and the others are skipped, not duplicated: the movie
  cannot have more frames than the run has steps.
- A run recorded with `dt` should be bit-for-bit the unrecorded run. That rests on a held
  clock being an exact no-op (below), and gets a test.
- The rule lives in `clock.wgsl` ([clock.wgsl:56](../src/hydro/clock.wgsl)), because the
  timestep is computed and consumed on the GPU and steps are submitted in batches: the page
  cannot stop a batch at the right step itself. Today the clock clamps `dt` to one stop time
  (`t_end`, which `sim.ts` fills with the smaller of `tEnd` and `tStop`). The frame time
  becomes a second, soft stop time in the params: the clock holds (`dt = 0`) while
  `t_frame - t < CYCLE * dt / 2`, and never clamps `dt` to it. Hard stops keep their exact
  landing.
- The recorder sees the hold in the diagnostics, as `run()` sees one today, records, and moves
  the soft stop to the next multiple of `x` more than half a step beyond the clock. The clock
  then computes the same `dt` it would have taken.

The default is `every: n` with n chosen from the measured step rate so that recording costs
little (a draw and an encode per frame are not free: at `every: 1` a fast 2D run slows to the
encoder's pace).

Playback rate is separate: `fps` (default 30) only sets the timestamps, frame `i` at `i / fps`.

**Interpretation to confirm:** "every nth frame" is read here as every nth simulation step,
since the recorder paces itself by steps and has no display frames of its own. If it should
mean every nth display refresh of a live run instead, section 3 changes and the movie stops
being reproducible.

## 4. The recording loop

```
configure encoder and muxer
draw and capture frame 0 (the state at the start)
until stopped, or tEnd, or the frame limit:
    run({ steps: n })  or  run to the next frame time as a soft stop (section 3.1)
    refresh diagnostics (the auto range follows the data)
    draw the view at the movie's size
    in the same task: copy the canvas to the compositing canvas, paint overlays,
        new VideoFrame(compositing canvas, { timestamp }), encoder.encode(), frame.close()
    wait while encoder.encodeQueueSize is above a few frames
encoder.flush(), muxer.finalize(), download
```

Points that the code has to respect:

- **Same task as the draw.** A WebGPU canvas holds its frame only until the task ends.
  `screenshot()` already works this way ([api.ts:287](../src/ui/api.ts)); the recorder shares
  its draw-twice step for fields whose range comes from the display pass.
- **The page's own frame loop stands down.** While recording, `tick()`
  ([app.ts:3383](../src/ui/app.ts)) keeps the readouts and the HUD going and neither steps nor
  draws: `setDriving` already stops its stepping, and a new `recording` flag stops its draw, so
  the two do not fight over the canvas's size.
- **What the page shows meanwhile.** The canvas holds the movie's frames at the movie's size
  (`renderer.fixedSize`), letterboxed in its place with `object-fit: contain`, so the user
  watches exactly what is recorded. Pointer input on the canvas is off until the recording
  ends, since the pointer mapping assumes the page's size. The slice is centered, as in a
  sized screenshot ([app.ts:3564](../src/ui/app.ts)).
- **Stopping.** Stop button, the run's end time, a frame limit, a replaced run (load, reset,
  rebuild: finish the file with what was recorded), or an encoder error. `draw(null)` and the
  flags are restored in a `finally`.
- **Checkpoints.** Both hold the clock with `cfg.tStop`. `run()` saves and restores it; the
  recorder writes any checkpoint the clock lands on, or v1 refuses to record with checkpoints
  on. Decide during implementation, after reading `armCheckpoints`.
- **Adaptive runs.** `encodeSteps` may take fewer steps than asked; `run()` already loops until
  the count is reached.

## 5. Encoding and muxing

- **Codec.** H.264 High profile, the level chosen from size and frame rate (4.0 covers
  1080p30, 4.2 1080p60, 5.1 4K), confirmed with `VideoEncoder.isConfigSupported`. If the
  browser has no H.264 encoder, fail with a message that says so; no silent fallback to another
  format.
- **Size.** Sides rounded down to even numbers, capped by `maxTextureDimension2D` and by what
  the encoder accepts.
- **Bitrate.** Default about 0.2 bits per pixel per frame (12 Mbit/s at 1080p30). Smooth
  color-mapped gradients band at the usual web bitrates. A `bitrate` option overrides it.
- **Keyframes.** One every 2 s of movie, so scrubbing works.
- **Memory.** The muxer holds the whole file. At the default bitrate that is 90 MB per minute
  of movie: fine in memory for the common case. A frame limit (default 10 minutes of movie)
  bounds it. Later: stream to disk through the File System Access API where the browser has it,
  as checkpoints do ([checkpoints.ts](../src/io/checkpoints.ts)).
- **Download.** `download()` from `io/checkpoints.ts` with type `video/mp4`, named from the
  run's stem as `.athdf` exports are.

## 6. Overlays

A new module, `src/render/overlays.ts`: pure functions that paint onto a
`CanvasRenderingContext2D`, given the view, the units and the frame's size. No DOM lookups, so
they can be unit tested and used by `screenshot()` as well.

Sizes scale with the frame's height, so a 4K movie's labels are as large on screen as a 720p
movie's. Text and lines are light on the dark background and dark on the light one
(`view.light`), with a thin contrasting edge so they read over bright gas.

### 6.1 Color bar

Right edge, vertically centered, about 60% of the frame's height.

- The gradient from `sampleColormap`, as the page's bar draws it
  ([transfer.ts:211](../src/render/transfer.ts)), without the opacity curve.
- Ticks and labels in the displayed units (`view.lo`, `view.hi` times the unit scale, as
  `showRange()` does at [app.ts:789](../src/ui/app.ts)): the two ends plus round values
  between, decades when `view.log` is on.
- A title: the field's name and unit (`FIELDS[view.field]`, `units().label`).
- Read fresh each frame, so with Auto range on the labels follow the data. A fixed range gives
  a steadier movie; the Record controls say so but do not force it.

### 6.2 Triad

Lower left corner. There is no triad in the app today, so this is new drawing, not a copy.

- **Volume view.** The three box axes projected with the camera's `right` and `up` from
  `cameraFrame()` ([renderer.ts:209](../src/render/renderer.ts)), orthographically, drawn back
  to front so the nearest arrow is on top, labelled x, y, z. An axis pointing at the viewer
  shortens to a dot.
- **Slice and projection views.** Two arrows along the plane's axes, labelled with their names
  in the convention `slice()` uses ([api.ts:481](../src/ui/api.ts)), and the slice axis as a
  dot-in-circle or cross-in-circle for out of or into the screen.
- **2D runs.** x and y only.

### 6.3 Options

`overlays: { colorbar: true, triad: true }`, both on by default, each can be turned off.
A time stamp (`t = …` in the displayed units) is a natural third and is cheap once the module
exists; left out of v1 unless wanted.

## 7. API

In `src/ui/api.ts`, documented in `docs/api.md`:

```ts
record({
  every?: number,            // a frame every n steps
  dt?: number,               // or a frame every dt of code time (one of the two)
  frames?: number,           // stop after this many frames
  until?: number,            // stop when the clock reaches this time
  fps?: number,              // playback rate, default 30
  width?: number, height?: number,   // default 1920 x 1080
  bitrate?: number,
  overlays?: { colorbar?: boolean, triad?: boolean },
  download?: boolean,        // default true in the page
}) → { frames, seconds, t, bytes, name, base64? }
stopRecording()
```

`screenshot()` gains `overlays` with the same meaning (default off, so existing callers see no
change).

Headless: `record` through the existing `call()` bridge; the CLI gets `--mp4` beside `--png`
([cli.mjs:33](../tools/headless/cli.mjs)) and the MCP server a `record` tool, both writing the
bytes to a file. The headless session launches Chrome proper (`channel: 'chrome'`), which has
an H.264 encoder; a bundled Chromium may not, and the error message covers that.

## 8. Interface

A Record button beside Run, Step and Reset ([app.ts:76](../src/ui/app.ts)), which becomes Stop
with a frame count and movie length while recording. In the Run section of the panel: frame
every (n steps), or every (code time); playback fps; size (720p, 1080p, 1440p, 4K, or the
view's own shape); color bar and triad checkboxes. Hidden when the browser has no
`VideoEncoder`.

## 9. Steps

1. **Overlays module.** `overlays.ts` with the color bar and the triad, unit tests for tick
   placement and the triad's projection, and `screenshot({ overlays })` so they can be looked
   at in PNGs before any video exists.
2. **Recorder core.** `src/io/movie.ts`: encoder and muxer set-up, the frame queue with
   backpressure, finalize. Takes a canvas per frame; knows nothing of the simulation.
3. **Soft stop.** The frame time as a second stop in `clock.wgsl` and the params, which holds
   the clock at the nearest step boundary and never shortens a step, and its GPU test.
4. **API `record()`.** The loop of section 4 with both cadences, the `recording` flag in
   `tick()`, the letterboxed canvas, stop conditions. A GPU test in `tests/gpu/api.spec.ts`:
   record a few frames of a small 2D run, check the file is an MP4 with the expected frame
   count and size.
5. **Page controls.** The button, the settings, the progress readout, the link encoding left
   alone (recording settings are not part of a shared link).
6. **Headless.** `--mp4` in the CLI, the MCP `record` tool, a line in `selftest.mjs`,
   `docs/api.md` and `docs/headless.md`.

Steps 1, 2 and 3 are independent of each other; 4 needs all three.

## 10. Risks and open questions

- **Meaning of "every nth frame"** (section 3).
- **Soft stop in the clock** (section 3.1). Needs a slot in the params block and the same rule
  for the adaptive mesh’s time bins, where a cycle is `CYCLE` steps (read `amr_sim.ts` before
  writing it).
- **Is a held clock an exact no-op?** Steps left in a batch after the hold run with `dt = 0`.
  Floors, cooling, the dual-energy sync and re-meshing must leave the state untouched then, or
  the recorded run differs. Checkpoints rely on the same hold already. The GPU test: the same
  setup run with and without a `dt` recording ends in identical state, with the same step count.
- **Muxer library choice**, and its license and size, checked at the time.
- **Readback cost.** `run()` reads diagnostics once per round, so `every: 1` on a fast run is
  paced by readbacks and encodes. Acceptable; the default n avoids it.
- **Firefox and Safari.** Both have `VideoEncoder`; H.264 encode support varies by platform.
  `isConfigSupported` decides, and the button says why when it is unavailable.
- **Hidden tabs.** The API's draw works without layout, and `VideoFrame` from a canvas should
  too, but timers are throttled: verify a recording survives the tab going to the background.
- **A separate render target.** Rendering the movie into its own WebGPU canvas would let the
  page keep its interactive view (and pointer camera moves) during a recording. The renderer
  reads its size from the one canvas in several places ([renderer.ts:567](../src/render/renderer.ts)),
  so this is a later refactor, not v1.

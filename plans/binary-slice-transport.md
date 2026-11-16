# Plan: binary array transport for `slice()`

Status: built 2026-11-16. Measurements and what changed from the plan are in section 8; the
sizes in section 2 were estimates made before them.

## 1. Goal

Let `shaderous.slice()` return its values as packed single-precision bytes instead of a JSON
list of numbers, and let the command-line runner and the MCP server write them as a file numpy
reads directly. The default result stays what it is today, so no existing script breaks.

The idea comes from Tesseract's three array encodings (JSON lists, base64 in JSON, references
to binary files): JSON lists for small arrays a person reads, packed bytes for the rest.

### Non-goals for v1

- `.athdf` transport. It is already base64; its next step is writing straight to disk without
  passing through `page.evaluate`, which is a different mechanism (section 7).
- `stats()`, `probe()` and `diagnostics()`: their results are small.
- New kinds of read-back (a whole 3D variable, a projection). The envelope in section 3 is
  written so that they can use it later.
- Changing the default encoding. Section 7 says when that could be revisited.

## 2. Where things stand

- `slice()` ([src/ui/api.ts:1004](../src/ui/api.ts)) samples `res` x `res` points, `res` up to
  2048, into a `new Array<number>(res * res)` and returns it as `values`.
- The result crosses `page.evaluate` ([tools/headless/session.mjs:120](../tools/headless/session.mjs)),
  which serializes it as JSON text in the page and parses it in Node.
- Every value starts as a 32-bit float on the GPU. Widened to a double, it prints with up to
  17 significant digits.

| `res` | Values | As a JSON list (estimate) | As base64 of float32 |
|---|---|---|---|
| 64 | 4,096 | 80 kB | 22 kB |
| 256 (the default on most grids) | 65,536 | 1.3 MB | 350 kB |
| 1024 | 1.05 million | 21 MB | 5.6 MB |
| 2048 | 4.19 million | 84 MB | 22 MB |

Who reads `values` today:

| Reader | Where | What it does |
|---|---|---|
| Runner, `--slice VAR` | [cli.mjs:211](../tools/headless/cli.mjs) | Writes the whole result as `NNNN-tT-VAR.json` |
| MCP `slice` tool | [mcp.mjs:121](../tools/headless/mcp.mjs) | Takes the mean, reshapes into rows, returns or saves JSON |
| Self-test | [selftest.mjs:48](../tools/headless/selftest.mjs) | Checks the row shape |
| GPU tests | [api.spec.ts:42, 73, 184](../tests/gpu/api.spec.ts) | Length, refinement levels, bit-equality during a recording |
| Docs | [api.md](../docs/api.md), [headless.md](../docs/headless.md), the `shaderous-sim` skill | Describe the result |

## 3. Design

### The API

One new option:

```js
shaderous.slice({ variable: 'rho', res: 1024, encoding: 'base64' })
```

| `encoding` | Result |
|---|---|
| `'json'` (default) | As today: `{ t, variable, axis, pos, res, axes, min, max, values }` |
| `'base64'` | The same without `values`, plus `array: { dtype: 'float32', shape: [res, res], order: 'C', endian: 'little', base64 }` |

- `shape` is `[j, i]`, so the bytes are `values[j * res + i]` in order: the same layout as
  today, and what numpy calls C order.
- `endian` is stated, not assumed. Typed arrays use the machine's byte order; every platform
  WebGPU runs on is little-endian, and the page refuses to encode on one that is not.
- An unknown `encoding` rejects with a message listing the two, as the API's other bad options
  do.
- `min` and `max` stay in the result either way, so a caller can scale a picture without
  decoding.

### The envelope is its own type

`array` is an `EncodedArray`, built by one function `encodeArray(typed, shape)` in a new
`src/io/arrays.ts`, with `decodeArray` beside it. Both are pure, so they are unit-tested in
Node. A later read-back (a 3D variable, a projection) returns the same shape of object.

### Inside `slice()`

Fill a `Float32Array(res * res)` in both modes; for `'json'`, finish with `Array.from`. One
loop serves both, and the JSON result is unchanged as long as the sampled values are already
single precision. That is a thing to check, not to assume: `variableOf` may compute a derived
variable in doubles. If it does, the JSON path keeps its own `Array<number>` so that its
output does not change by a bit, and `'base64'` documents that it rounds to single precision.

`toBase64` ([api.ts:201](../src/ui/api.ts)) already encodes bytes without building one giant
string of character codes; reuse it.

### The Node side

- `session.mjs` gets `sliceArray(result)`: a `Float32Array` from `result.array`, copied out of
  the `Buffer` so that it is aligned.
- A small `.npy` writer (`tools/headless/npy.mjs`): the magic bytes, a version 1.0 header
  padded to 64 bytes, then the data. No dependency. `np.load` reads it with the right shape and
  dtype.

### The runner

`--slice-format json|npy`, default `json`.

- `json`: as today.
- `npy`: asks the page for `'base64'` and writes `NNNN-tT-VAR.npy`. The slice's other fields
  (`t`, `variable`, `axis`, `pos`, `res`, `axes`, `min`, `max`) go into that output's entry in
  `manifest.json`, under `slices`, so nothing is lost by dropping the JSON wrapper.

### The MCP tool

- Always fetch as `'base64'`: this is transport between the page and the server, and the agent
  never sees it.
- What the tool returns is unchanged: the summary, and rows of values when `res` is at most 64
  or `include_values` is set.
- `save_path` ending in `.npy` writes a `.npy` file; any other name writes JSON as today.
- The mean is summed in doubles from the `Float32Array`.

## 4. What this buys, and what it may not

Size falls by about four times at every resolution. Time is less certain. The sampling loop
calls a closure per point, on an adaptive mesh through the block tree, and at 2048 squared
that is 4.2 million calls; it may cost more than serializing does. Step 1 measures the two
apart. If sampling dominates, this plan still cuts the bytes and the files but not the wait,
and the wait wants a different fix (sampling on the GPU, or row by row from the block data).
The plan goes ahead either way only if step 1 shows serialization is at least a quarter of
the call at `res` 1024; otherwise stop after step 1 and write down what was found.

## 5. Steps

1. **Measure.** A script that times `slice()` at `res` 256, 1024 and 2048 on a uniform 2D
   grid and a 3D adaptive one, splitting the page's time (sampling, building the result) from
   the bridge's (the call's whole time less the page's). Run at normal priority, on a static
   build. Record the table in this file.
2. **`src/io/arrays.ts`** with `encodeArray` and `decodeArray`, and unit tests: a round trip,
   the shape and dtype fields, NaN and infinities surviving, a refusal on a wrong byte length.
3. **`slice({ encoding })`** in `api.ts`, with the check on double-precision variables from
   section 3.
4. **GPU test** in `api.spec.ts`: the decoded `'base64'` slice equals the `'json'` one value
   for value after `Math.fround`, for `rho`, `press` and `level`, on a uniform and an adaptive
   run; an unknown encoding rejects.
5. **`npy.mjs` and `sliceArray`**, with a unit test of the header bytes against a file written
   by numpy (kept as a fixture of a few hundred bytes).
6. **Runner**: `--slice-format`, the manifest's `slices` entry, the help text.
7. **MCP**: base64 transport and `.npy` for `save_path`.
8. **Self-test**: the runner with `--slice-format npy` writes a file of `80 + 4 * res * res`
   bytes with the right header; the MCP tool's answers are what they were.
9. **Docs**: `slice` in `api.md`, the outputs table and the tool list in `headless.md`, the
   `shaderous-sim` skill, and a recipe reading a `.npy` slice with numpy.
10. **Measure again** with the script of step 1 and add the after column.

Steps 2 to 4 ship alone (the API gains an option nobody is made to use). Steps 5 to 8 ship
together. Each is a few hours; the whole is about a day.

## 6. Risks

| Risk | Answer |
|---|---|
| A derived variable is computed in doubles, so `'base64'` differs from `'json'` in the last bits | Checked in step 3; documented as rounding to single precision; the JSON path is left exact |
| A script reads `values` from a base64 result and finds nothing | `values` is absent, not empty, so the failure is loud; the docs show `array` beside it |
| `.npy` loses the slice's metadata | It is in `manifest.json`; the recipe in the docs reads both |
| The bridge is not where the time goes | Section 4: measured first, with a stopping rule |

## 7. Later, not now

- **Default encoding.** If nothing outside the repository turns out to depend on JSON slices,
  the runner could write `.npy` by default. Not before a release note says so.
- **`.athdf` straight to disk.** A 512 cubed state is gigabytes, and base64 through
  `page.evaluate` builds it as one string on both sides. The fix is a download the browser
  streams to a file (Playwright's download event, as the page's own Export button triggers),
  which is binary all the way. It shares nothing with this plan but the motive.
- **A 3D read-back** of one variable on a uniform resampling, returned as an `EncodedArray`.

## 8. What was measured, and what was built

### Step 1: before (a prototype of the base64 path, in the page)

`tools/headless/bench_slice.mjs` on a static build, normal priority, median of 3. "Page" is
the call's time inside the page (snapshot capture plus sampling); "bridge" is the rest of the
call as Node sees it. Sedov at t = 0.02. Times in ms; bytes as `JSON.stringify` of the result.

| Setup | `res` | JSON: page | JSON: bridge | JSON: total | JSON MB | base64: total | base64 MB |
|---|---|---|---|---|---|---|---|
| 2D uniform, n = 1024 | 256 | 45 | 182 | 227 | 0.2 | 51 | 0.3 |
| | 1024 | 127 | 2891 | 3018 | 3.3 | 269 | 5.6 |
| | 2048 | 414 | 12400 | 12814 | 13.2 | 902 | 22.4 |
| 3D adaptive, n = 64, amr = 2 | 256 | 33 | 209 | 242 | 0.4 | 42 | 0.3 |
| | 1024 | 113 | 3329 | 3442 | 6.2 | 245 | 5.6 |
| | 2048 | 404 | 13019 | 13424 | 25.0 | 802 | 22.4 |

Snapshot capture alone (`probe([])`): 43 ms (2D), 27 ms (3D).

Two things the estimates got wrong:

- **The time is all in the bridge**, 96% of the call at `res` 1024 (the stopping rule asked
  for a quarter). Sampling is cheap: 0.1 s at 1024, 0.4 s at 2048. Playwright's
  `page.evaluate` serializes a list of numbers element by element, far slower than JSON text.
- **The bytes barely shrink.** A Sedov slice is mostly the ambient value, which prints as
  `1`, so the JSON is 3 to 6 bytes a number, not 20. Base64 of float32 is a flat 5.3 bytes a
  number: larger than the JSON on the 2D run, 10% smaller on the 3D one. The `.npy` file (4
  bytes a number) is about as large as the JSON file. The gain is time, not size.

### Step 10: after (`encoding: 'base64'` built in)

| Setup | `res` | JSON: total | base64: page | base64: bridge | base64: total | Faster |
|---|---|---|---|---|---|---|
| 2D uniform, n = 1024 | 256 | 253 | 62 | 22 | 84 | 3x |
| | 1024 | 3031 | 80 | 68 | 148 | 20x |
| | 2048 | 12903 | 226 | 252 | 478 | 27x |
| 3D adaptive, n = 64, amr = 2 | 256 | 259 | 32 | 8 | 41 | 6x |
| | 1024 | 6289 | 117 | 112 | 229 | 27x |
| | 2048 | 17527 | 371 | 409 | 779 | 22x |

(The 3D JSON column ran slower than in step 1, 6.3 s against 3.4 s at 1024: JSON timings
through the bridge are noisy on this machine; the base64 ones agree within 10%.)

### Changes from the plan

- **The runner fetches base64 for both formats**, not only for `npy`: the JSON file it writes
  is built in Node from the Float32Array and is the same as before (every variable is read from
  a float32 array, and `level` is a small integer, so the values are exact; the GPU test checks
  it bit for bit). So `--slice` is 20 times faster at large `res` without changing a file.
- **No double-precision path.** `variableOf` reads every value from a float32 array, so step
  3's check found nothing to keep apart: one Float32Array loop serves both encodings.
- **`encodeArray` has its own synchronous base64** (`Uint8Array.prototype.toBase64` where the
  engine has it, else `btoa` in chunks of 48 kB) instead of the page's FileReader `toBase64`,
  so that it is pure and tested in Node. It is faster than the FileReader prototype of step 1
  (page time at 2048: 226 ms against 483).
- **The `.npy` header is 128 bytes** for a 2D shape (version 1.0 pads to 64, and the header
  dict is longer than 54 characters), not 80; the self-test checks `128 + 4 * res * res`.
- The `.npy` writer has a reader beside it (`readNpy`), used by the tests and the self-test.

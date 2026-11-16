# Plan: one schema, generated everywhere

Status: proposed 2026-11-16, not built. The lists of hand-kept copies below come from reading
the code, not from comparing every copy value by value; phase 1 does that comparison and will
say how much has actually drifted.

## 1. Goal

Every fact about what a setup, a view or an API call accepts is written once, and each place
that repeats it today (the document codec, `describe()`, the MCP tool schemas, the runner's
help, the tables in the docs, the agent skill) is derived from that one place or checked
against it by a test.

The idea comes from Tesseract, where one typed schema per component produces its command
line, its HTTP interface and its reference docs.

### Non-goals

- A schema library (zod, TypeBox). The tables are plain data, as `options.ts` already is.
- Generating prose. Only tables and lists of keys are generated; the text around them stays
  written by hand.
- Changing any link key, document path, tool name or flag.
- Typed descriptions of results (`diagnostics()`, `state()`). Worth doing later for units and
  meanings; left out here to keep this to inputs.

## 2. Where things stand

Most of the way there for setup options. [src/setup/options.ts](../src/setup/options.ts) is
"the option table: every setting of a setup, listed once", and the dialog, the conflict
checks, the defaults, the link keys and `controls()` read it. `describe().setup` publishes it.

What is still kept by hand, in a second or third copy:

| Fact | Source of truth today | Hand-kept copies |
|---|---|---|
| Choices of `riemann`, `limiter` | The `select` choices in `options.ts` | Type unions in [sim.ts:27](../src/sim.ts) and [spec.ts:73](../src/setup/spec.ts); enums in [codec.ts:76](../src/setup/codec.ts); literal arrays in `describe()` ([api.ts:885](../src/ui/api.ts)) |
| Choices of `eos`, `gravity`, `cooling`, `frame` | Label maps and `options.ts` | Enums in [codec.ts:72-78](../src/setup/codec.ts) |
| Which setup keys exist | `linkKey` in `options.ts`, plus the grid's keys in [link.ts](../src/ui/link.ts) | The `setupKeys` sentence in [mcp.mjs:67](../tools/headless/mcp.mjs); the key table in [api.md](../docs/api.md#setups); the skill |
| View settings: names, link keys, ranges | None. `ViewState` in the renderer, `VIEW_NUMBERS` and `VIEW_KEYS` in [link.ts:84](../src/ui/link.ts), clamps where each is applied | The `screenshot` tool's description in [mcp.mjs:144](../tools/headless/mcp.mjs); the `view()` table in [api.md](../docs/api.md); `--view` in the runner's help |
| Slice variables | `VARIABLES` in `api.ts` | The `slice` tool's description in `mcp.mjs` |
| What each API method takes | TypeScript signatures in `api.ts` | Every `inputSchema` in `mcp.mjs`; the reference headings in `api.md`; the skill |
| The runner's flags | `parseArgs` options in [cli.mjs](../tools/headless/cli.mjs) | The `HELP` text in the same file; [headless.md](../docs/headless.md) |
| The setup document's shape | `DOCUMENT` in `codec.ts`, partly from `options.ts` | The tables in [setups.md](../docs/setups.md) |

## 3. Design

### Three tables

| Table | State | Holds |
|---|---|---|
| `OPTIONS`, `PART_OPTIONS` | Exists | Every setup option: path, link key, control, range, choices, default, rules |
| `VIEW_OPTIONS` | New, `src/render/view_options.ts` | Every view setting: `ViewState` key, link key, kind, range or choices, which modes it applies to, a one-line meaning |
| `API_METHODS` | New, `src/ui/api_schema.ts` | Every method of `window.shaderous`: name, one-line summary, parameters as JSON Schema, and whether it changes the run |

`VIEW_OPTIONS` is the one with real work in it, because the view has no table at all today.
`encodeLink` and `decodeLink` loop over it in place of `VIEW_NUMBERS` and `VIEW_KEYS`,
`view()` validates and clamps from it, and a `satisfies` check ties its keys to `ViewState`
so that a new view setting without an entry fails to compile.

`API_METHODS` is deliberately thin: parameter schemas and summaries only. The MCP tools are
not one-to-one with the methods (`save_path`, `include_values` and the rounding are the
server's own), so each tool takes its method's parameter schema and adds its own keys on top.
A tool with no method behind it keeps its hand-written schema.

### Choices as constants

Each list of choices becomes one `as const` array beside its meaning (`RIEMANN_KINDS` in
`sim.ts`, and so on). The type union is `(typeof RIEMANN_KINDS)[number]`, the option's
`select` is built from it with labels, the codec's enum reads it, and `describe()` returns
it. The codec's `DOCUMENT` keeps its hand-written shape for the nesting but takes every enum
and range from the option at that path.

### One generated file for Node

`mcp.mjs` and `cli.mjs` are plain Node and cannot import the TypeScript tables, and the MCP
server should be able to list its tools before a browser is up. So a script writes
`tools/headless/schema.json`: setup keys with kinds, ranges and choices; view keys likewise;
slice variables; each method's parameter schema.

- The file is committed, so the tools need no build step.
- `npm run gen` rewrites it (run through Vitest's own runner, which already loads the
  TypeScript; no new dependency).
- A unit test builds it in memory and fails if the committed file differs, naming
  `npm run gen`. `npm test` gates the Pages deploy, so a stale file cannot ship.

With it, the runner can refuse an unknown `--set` or `--view` key before launching Chrome,
naming the nearest valid one, and its help can list the keys.

### Generated tables in the docs

Tables that are pure listings sit between markers:

```md
<!-- gen:view-keys -->
| Key | Meaning |
...
<!-- /gen -->
```

`npm run gen` rewrites what is between them in `docs/api.md`, `docs/setups.md`,
`docs/headless.md` and the `shaderous-sim` skill; the same unit test checks them. The
`view()` table and the setup key table go first. A table whose cells are long explanations
(the parts table in `setups.md`) stays by hand, and is checked instead: a test asserts that
every key the table names exists and that every option appears in it.

### The runner's flags

One array of `{ flag, type, arg, help, group }` in `cli.mjs` builds both the `parseArgs`
options and the `HELP` text. This is local to that file and independent of the rest.

### A JSON Schema for setup documents

From `DOCUMENT` and the option tables, generate `public/shaderous-setup.schema.json`, served
with the site. A `.shaderous.json` file that names it under `$schema` gets completion and
range checking in an editor. The codec learns to accept `$schema` without a warning. Files
saved from the Setup dialog include the line.

## 4. Phases

Each phase lands by itself and leaves the tree working.

1. **Audit, as tests.** A unit test per row of the table in section 2 that can be checked
   mechanically: the codec's enums equal the options' choices; `describe()`'s lists equal
   them; every key in the docs' setup and view tables is a real key, and every real key is in
   the table; every `linkKey` is unique; the MCP's `setupKeys` sentence names only real keys.
   No behavior changes. Whatever fails is the drift, fixed by hand in this phase and listed
   here. This phase is worth having even if the rest is never built.
2. **Choices as constants** (`sim.ts`, `spec.ts`, `options.ts`, `codec.ts`, `api.ts`). The
   phase 1 tests for enums become trivially true and are kept.
3. **`VIEW_OPTIONS`**, with `link.ts` and `view()` reading it. Guarded by the existing link
   round-trip tests in `link.test.ts` and the GPU tests of `view()`; links written before must
   decode to the same view.
4. **`schema.json` and `npm run gen`**, with the staleness test. `mcp.mjs` reads the setup
   keys, view keys and variables from it; the hand-written sentences go.
5. **`API_METHODS`**, and the MCP `inputSchema`s derived from it. `selftest.mjs` already
   drives every tool, and a new check compares `tools/list` before and after: same names, same
   required keys, same types.
6. **Generated doc tables** and the marker check.
7. **The runner's flag table.**
8. **The setup document's JSON Schema.**

Phases 1 and 2 are a few hours each. Phase 3 is the largest, about a day, because the link
code is rewritten around the table. Phases 4 to 8 are half a day each. Phases 7 and 8 depend
on nothing before them and can be done at any point.

## 5. Risks

| Risk | Answer |
|---|---|
| A link written by an older build decodes differently after phase 3 | The round-trip tests, plus a fixture of real links from the README and the teaching guide decoded before and after |
| `describe()` changes shape and breaks a script | It does not change: the same fields, now filled from the constants. A snapshot test of its keys |
| The generated file causes merge noise between sessions sharing the checkout | It is small and deterministic (sorted keys); a conflict is resolved by running `npm run gen` |
| `options.ts`, `api.ts` and `link.ts` are busy files | Small commits, one phase each; phase 1 touches only tests |
| `API_METHODS` grows into a second copy of the TypeScript signatures | A `satisfies` check that its method names equal the API object's keys; parameter types are not duplicated beyond what the MCP needs |
| Generated docs read worse than the hand-written tables | Only listings are generated; a table that reads worse is moved back to hand-kept with a check |

## 6. Decisions to make before starting

| Decision | Proposal | Alternative |
|---|---|---|
| Is `schema.json` committed? | Yes, with a staleness test | Built at server start by asking the page's `describe()`: no file, but no tool list until a browser is up |
| How far to take `API_METHODS` | Parameters and summaries only | Leave the MCP schemas by hand and only check their key names against the methods (cheaper, less complete) |
| Generated doc tables | The setup key and view tables only, at first | Check-only for all tables, generate none |

## 7. Relation to other plans

Independent of [binary-slice-transport.md](binary-slice-transport.md). Both touch the `slice`
tool's schema in `mcp.mjs` and `slice` in `api.md`; whichever lands second carries the other's
change across.

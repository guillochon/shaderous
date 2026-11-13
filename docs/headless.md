# Headless runs: command line and MCP server

There are two tools for running Shaderous without anyone watching the page. Both are in
[tools/headless](../tools/headless), and both are built on the page's
[scripting API](api.md):

- **The command-line runner** ([cli.mjs](../tools/headless/cli.mjs)) runs a setup or a
  parameter sweep to completion. It writes diagnostics, images, slices and `.athdf` files to a
  folder. Use it for batch work: production runs, convergence studies, figures, CI.
- **The MCP server** ([mcp.mjs](../tools/headless/mcp.mjs)) gives an AI agent the simulator
  as a set of tools. The agent loads a setup, steps it, reads numbers back and looks at
  screenshots, all within one conversation. Use it for exploring.

Both work the same way. They serve this checkout with Vite, or use a URL you give. They open
the page in Chrome with WebGPU, headless by default, and call `window.shaderous`. Nothing has to
be running beforehand.

**Contents**

1. [Requirements](#requirements)
2. [Command-line runner](#command-line-runner)
3. [MCP server](#mcp-server)
4. [Troubleshooting](#troubleshooting)

---

## Requirements

- Node 20 or later, and `npm install` in the checkout.
- Google Chrome. Set `PW_CHANNEL=msedge` to use Edge. Playwright drives the installed browser,
  so it doesn't need to download one.
- A GPU that Chrome will use for WebGPU. On a machine without one, Chrome may fall back to a
  software adapter, which works but is very slow. `state` and the run manifest name the
  adapter, so you can tell which one you got.

`npm run test:headless` checks the whole chain on your machine. It drives the MCP server
through a short conversation, then runs the command-line runner, and reports each check.

## Command-line runner

```bash
npm run headless -- <command> [setup] [options]
```

`node tools/headless/cli.mjs <command> ...` does the same. `--help` lists every option.

### Commands

- **`run`** runs one setup and writes its outputs to a folder.
- **`sweep`** runs a setup once for every combination of `--vary` values.
- **`describe [problem]`** prints the problems, fields and choices as JSON, or one problem's
  full entry.

### The setup

A setup uses the keys of a shareable link (see [api.md](api.md#setups)). It can be given as a
query string or link, as flags, or both:

```bash
npm run headless -- run "problem=sedov&dim=3&n=128&amr=2" --until 0.05
```

```bash
npm run headless -- run --problem sedov --dim 3 --n 128 --amr 2 --set gamma=1.4 --until 0.05
```

`--set KEY=VALUE` takes any setup key and can be repeated. That includes a problem's own
parameters, as `p.<id>`.

### When to stop

- `--until T` runs to time T. The last step is shortened so that the run lands on T exactly.
- `--steps N` takes at most N steps.
- With neither, the run goes to the problem's end time.
- `--every DT` writes outputs at t = 0 and every DT after it. Output times are exact multiples
  of DT, so outputs line up across runs with different timesteps.
- `--timeout SECONDS` stops a run that is taking too long.

### Outputs

Each run writes a folder, `--out DIR`, which defaults to `runs/<problem>-<date>`.

| File | Contents |
| --- | --- |
| `manifest.json` | The setup in force (`setup`, a query string that reproduces the run), view settings, build, GPU adapter, the command line, start and end time, why the run stopped, the files of each output, and any errors the page reported |
| `diag.jsonl` | One line of diagnostics per output: time, steps, wall time, mass, momentum, energies, Mach number, point masses, field ranges |
| `NNNN-tT.png` | A screenshot at each output, with `--png`. Set the size with `--width` and `--height`, and what is drawn with `--view KEY=VALUE` |
| `NNNN-tT-VAR.json` | A slice of variable VAR at each output, with `--slice VAR` (repeatable). Set it with `--slice-axis`, `--slice-pos` and `--slice-res`; the format is that of `slice()` in [api.md](api.md#reference) |
| `NNNN-tT.athdf` | The state of the gas at each output, with `--athdf`. yt and Athena++'s tools read these, and the page or `load_athdf` can continue the run from them |
| `NNNN-tT-debris.json` | Debris energies and the return rate dM/dt at each output, with `--debris`, for problems that track them (the tidal disruption). The format is that of `debris()` in [api.md](api.md#reference) |

A `--view` setting is any key of `view()` in [api.md](api.md#reference), such as
`--view mode=slice`, `--view field=pressure` (a field by name or index), `--view colormap=viridis`
or `--view log=false`.

The process exits with 0 on success, 1 on an error, and 2 if the page reported an error during
the run. In that last case the run's outputs are still written, and the errors are listed in
`manifest.json`.

### Examples

A 2D Kelvin-Helmholtz run, with a picture of the passive scalar every 0.25 time units:

```bash
npm run headless -- run --problem kelvin-helmholtz --dim 2 --n 1024 --until 2 --every 0.25 --png --view field=2 --out runs/kh
```

A convergence study of the Sedov blast, keeping density slices to compare:

```bash
npm run headless -- sweep --problem sedov --dim 2 --vary n=128,256,512,1024 --until 0.05 --slice rho --slice-res 512 --out runs/sedov-convergence
```

A sweep writes one folder per combination, named after its values (`n=256/`, or
`gamma=1.4,cfl=0.3/` for two keys). It also writes `sweep.jsonl`, with one line per run: the
values, the setup in force, the final time and steps, why the run stopped, and the wall time.

A long 3D run, at low priority so the machine stays responsive, with checkpoints:

```bash
npm run headless -- run --problem tidal-disruption --dim 3 --n 256 --amr 3 --every 0.5 --athdf --low-priority --out runs/tde
```

A tidal disruption's return rate, the classic dM/dt diagram. Run a few dynamical times past
pericenter with `--debris`, then plot the last output:

```bash
npm run headless -- run --problem tidal-disruption-full --dim 3 --n 128 --amr 2 --until 20 --every 1 --debris --low-priority --out runs/tde
```

```bash
python tools/headless/plot_dmdt.py runs/tde
```

The plot shows dM/dε, split into what has left the box and what is still in it, and dM/dt
against time since pericenter. It uses solar masses per year and years where the problem has
units. Give it several outputs (`runs/tde/*-debris.json`) to see whether dM/dt has stopped
changing. It needs numpy and matplotlib.

### Options

| Option | Meaning |
| --- | --- |
| `--url URL` | Use the page served at URL (for example the live site) instead of serving this checkout |
| `--headed` | Show the browser window, for when headless Chrome finds no GPU adapter |
| `--low-priority` | Run at below-normal CPU priority. On Windows the browser inherits it |
| `--quiet` | No progress lines on stderr |

## MCP server

The server speaks the [Model Context Protocol](https://modelcontextprotocol.io) over stdio. It
uses no libraries beyond those the project already has. It opens one simulator page on the first
tool call and keeps it until the client disconnects.

### Connecting

[.mcp.json](../.mcp.json) at the repository root registers the server for Claude Code in this
project. Claude Code asks you to approve it the first time. For another client, or from
elsewhere, give the command with an absolute path:

```json
{
  "mcpServers": {
    "shaderous": {
      "command": "node",
      "args": ["/path/to/shaderous/tools/headless/mcp.mjs"]
    }
  }
}
```

With Claude Code you can also run
`claude mcp add shaderous -- node /path/to/shaderous/tools/headless/mcp.mjs`.

The server accepts `--url URL`, `--headed` and `--low-priority`, which mean the same as for the
runner.

### Tools

| Tool | What it does |
| --- | --- |
| `describe` | Lists problems (id, dimensions, end time, parameters), fields, variables and choices. Give `problem` for one problem's full entry |
| `load` | Builds a run from a setup (query string or object), paused at t = 0 |
| `run` | Advances to a time (`until`, landed on exactly), by a number of `steps`, or whichever comes first, with an optional `timeout_seconds` |
| `state` | Returns the problem, grid, time, steps, end time, GPU adapter, and `query`, the setup in force |
| `diagnostics` | Reads back totals and extremes: mass, momentum, energies, Mach number, point masses, field ranges |
| `stats` | Gives each variable's min, max and volume-weighted mean over the box |
| `slice` | Samples a variable on a plane. Returns min, max and mean, with the values as rows when `res` is at most 64; `save_path` writes the full slice to a file instead |
| `probe` | Gives the gas at given points |
| `screenshot` | Returns a PNG image of the view, optionally changing the `view` first and saving to `save_path` |
| `view` | Changes what screenshots show, leaving the run alone |
| `controls`, `control` | List the panel's controls, or set one mid-run as a person would |
| `restart` | Starts again from the initial conditions |
| `save_athdf`, `load_athdf` | Write the state to an `.athdf` file, or start a run from one |
| `debris` | Gives debris energies and the return rate dM/dt, for problems that track them: totals, the surviving core, the first return and the peak. `save_path` writes the full report |
| `errors` | Lists what the page has reported going wrong |

File paths are relative to the server's working directory, which for Claude Code is the
project root. A tool that fails returns its error message as a tool error, with `isError` set,
so the model can correct its call.

## Troubleshooting

- **"the simulator did not start: ... WebGPU"**: Chrome found no WebGPU adapter. Try
  `--headed`, and check `chrome://gpu` in a normal window. On Windows the runner already passes
  `--use-angle=d3d11`, which headless Chrome needs there.
- **"could not start Chrome"**: Chrome isn't installed where Playwright looks. Install it, or
  use Edge with `PW_CHANNEL=msedge`.
- **It runs but is very slow**: check `adapter` in `state` or `manifest.json`. A software
  fallback adapter is 100 or more times slower than a GPU.
- **A command line with `&` in the setup misbehaves on Windows**: `cmd` splits commands on `&`.
  Quote the setup, or give it as flags (`--problem sedov --dim 2`).
- **Outputs past the end time are missing**: a run stops at the problem's end time. Pass a later
  one with `--set tend=10`, or `--set tend=inf` for none.

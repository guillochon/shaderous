# Paper

The journal class is set in `paper.toml` and described in `.claude/skills/journal/SKILL.md`.

One source, `main.tex`, typeset two ways by three-line wrappers that set `\papermode` before `\input{main}`:

| Wrapper | Output | Layout |
|---|---|---|
| `submit.tex` | `submit.pdf` | the layout the journal wants at submission |
| `preprint.tex` | `preprint.pdf` | close to the published article; this is what goes to arXiv |

Compiling `main.tex` directly gives the submission layout. For classes that have only one layout (MNRAS, A&A, amsart) the two PDFs are the same.

## Build

```bash
cd paper
latexmk -pdf
```

or `python util/build.py` from the repository root, which also reports what the logs say.

## arXiv

```bash
python util/arxiv_zip.py
```

writes `<slug>-arxiv.zip` in the repository root and test-compiles it in a clean directory.

## Figures

Every figure is drawn by a script in `figures/`; the PDFs are committed and the scripts are the record of how they were made.

| Script | Output | Data |
|---|---|---|
| `make_hydro.py` | `hydro.pdf` | `data/hydro/`: Sod, advection and Sedov runs (`collect.mjs`) |
| `make_balance.py` | `balance.pdf` | `data/balance/`: the stratified column and the white dwarf (`collect.mjs`) |
| `make_amr.py` | `amr.pdf` | `data/amr/`: images from the renderer (`shots.mjs`) |
| `make_gravity.py` | `gravity.pdf` | `data/gravity/`: Evrard runs (the command-line runner) and the sphere's potential (`collect.mjs`) |
| `make_tde.py` | `tde.pdf` | `data/tde/`: two tidal disruptions with debris energies (the command-line runner) |
| `make_gallery.py` | `gallery.pdf` | `data/gallery/`: images from the renderer and the interface (`shots.mjs`) |
| `make_performance.py` | `performance.pdf` | `data/performance/`: throughput on this machine's GPU (`collect.mjs`) |

Each script plots from the data committed beside it, with numpy and matplotlib:

```bash
uv run python paper/figures/make_hydro.py
```

With `--run`, a script first makes its data again by running the simulator. That needs the code's repository around this one (`npm install` there), Node, Chrome and a GPU, and takes minutes. The runs use the code's headless tools (`docs/headless.md` in the code's repository): the command-line runner, `shots.mjs` (images drawn by the code's renderer, or of the whole page) and `collect.mjs` (the validation suite's own setups, from `src/validation/figure_data.ts`). Each script's docstring says what it ran; the images and the runner's outputs also record the setup as the link that reproduces it.

## References

`references.bib` holds hand-written entries in the ADS style. Check them with `python util/check_citations.py` after every change.

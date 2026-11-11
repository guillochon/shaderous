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

| Script | Output | Needs |
|---|---|---|
| `make_example.py` | `example.pdf` | numpy, matplotlib |

```bash
uv run python paper/figures/make_example.py
```

## References

`references.bib` holds hand-written entries in the ADS style. Check them with `python util/check_citations.py` after every change.

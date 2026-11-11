# Shaderous paper

The paper that describes Shaderous, the WebGPU hydrodynamics code in the parent directory. It targets ApJ (AASTeX 7, `aastex701`) and was started from the [back-in-the-fold](https://github.com/guillochon/back-in-the-fold) paper template, whose rules, scripts and journal skill live here unchanged.

Run every command below from this directory (`paper/`), not from the repository root: the scripts in `util/` find `paper.toml` relative to themselves.

| Path | Contents |
|---|---|
| `paper/main.tex` | The manuscript |
| `paper/references.bib` | Bibliography: ADS exports, plus hand-written entries for papers that ADS does not index |
| `research/prior-art.md` | The literature search on hydrodynamics in graphics shaders, and what is new in Shaderous |
| `research/measurements.md` | Where every number in the manuscript comes from |
| `research/<citekey>/` | Downloaded sources used to check a claim |
| `.cursor/rules/*.mdc` | Writing rules (style, readability, abstract, citations, figures, numbers) |
| `.claude/skills/journal/` | How to format and submit for the AAS journals |
| `util/` | Build and check scripts |

## Build and check

```bash
python util/build.py      # paper/submit.pdf (one column) and paper/preprint.pdf (two columns)
python util/check.py      # journal agreement, abstract, citation keys and order, style
python util/check_floats.py
python util/check_citations.py   # needs ADS_TOKEN (SciX token) in the environment or .env
```

`check_citations.py` cannot verify ten entries against ADS because ADS does not index them (SIAM, ACM, Eurographics and Springer titles). Checked against Crossref instead (`--source open`), they pass except for known differences: Crossref truncates the titles of `Bolz2003` and `Lefohn2006` after the colon, `Harris2003` and `Goodnight2003` carry DataCite DOIs from the Eurographics library, and `Koren1993` is a book chapter with no DOI (checked against the CWI repository record).

## Still to do

- Measure throughput on named GPUs and browsers (`\todo` in the Performance section).
- Save a run of the validation suite and add the measured values to Table 1.
- Figures: a shock tube against the exact solution, the Sedov blast on the adaptive mesh, the well-balanced column, the Evrard energy curves, and a diagram of the flux recomputation across refinement and time-bin boundaries. Each needs a `paper/figures/make_<name>.py` script (see `.cursor/rules/figure-design.mdc`).
- Acknowledgments, ORCID in `paper.toml`, and a Zenodo DOI for the code.

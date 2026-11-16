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

- Acknowledgments, ORCID in `paper.toml`, and a Zenodo DOI for the code.
- Burning, once the network is retrained on the tabulated equation of state: the nickel yield of the collision and the limiter again, the measured speed of the carbon-oxygen detonation, and the network against the reference over a whole detonation.
- A figure of the carbon-oxygen detonation or the burning collision.
- The Apple M4 Pro throughput predates the longer timing of small grids (`throughput` in `src/validation/figure_data.ts`); measure it again when that machine is at hand.

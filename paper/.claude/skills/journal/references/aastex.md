# AAS Journals: `aastex` (AASTeX 7.0.1)

ApJ, ApJL, ApJS, AJ, PSJ and RNAAS share one class, `aastex701.cls`, which ships with TeX Live
and MiKTeX and is on arXiv. Author guide: https://journals.aas.org/aastexguide/ and
https://journals.aas.org/manuscript-preparation/. Check it for the current length limits
of the specific journal (Letters and RNAAS have hard ones).

## Layouts

| Wrapper | Class options | Use |
|---|---|---|
| `submit.tex` | `aastex701` (default) | single column; what you send to the journal and co-authors |
| `preprint.tex` | `[twocolumn]` | close to the published look; arXiv gets this |

AAS accepts either layout at submission. `[manuscript]` (double spaced) is only for referees
who ask for it. `[linenumbers]` helps during review.

## Front matter

- `\shorttitle{}` and `\shortauthors{}` set the running heads.
- Authors: `\author[ORCID]{Name}` then `\affiliation{...}` and `\email{...}` per author.
  Fill `orcid` in `paper.toml`; AAS shows ORCID links.
- Abstract: one paragraph, at most 250 words (and the 1920-character arXiv limit applies too).
- Keywords: AAS uses concepts from the Unified Astronomy Thesaurus (https://astrothesaurus.org),
  separated by ` --- `. Use the exact UAT concept names; AASTeX 7 can attach their IDs
  with `\uat{Concept}{ID}`.

## Body

- Cite with `\citet` / `\citep`; the class loads natbib. The style is `aasjournalv7.bst`.
- The class defines the ADS journal macros (`\apj`, `\mnras`, ...), so an ADS export works
  directly. Keep the hand-written bib anyway (see `.cursor/rules/verify-citations.mdc`).
- Tables: plain `table` + `tabular` with `booktabs` is fine. Use `deluxetable` only for tables
  that run over several pages; long data tables go to the journal as machine-readable tables.
- `figure*` for full-width figures in two-column mode. AAS also takes figure sets, animations
  and interactive figures; ask the journal before relying on one.
- Back matter, in order: `acknowledgments` environment, `\facilities{}` (with
  `\facility{HST(STIS)}` style names from the AAS facility list), `\software{}`, `\appendix`,
  then the bibliography. The template body already does this.
- Data: AAS asks for data behind figures and a DOI (Zenodo or similar) for software and data.

## arXiv

`aastex701.cls` is on arXiv, so `util/arxiv_zip.py` does not bundle it.
`aasjournalv7.bst` is not needed there because the zip carries the `.bbl`.

## After switching to `aastex`

- Replace the keyword list in `paper.toml` with UAT concepts and rerun the switch, or edit
  `\keywords{}` directly.
- Remove anything that another class required and AAS does not (highlights, MNRAS data
  availability heading, MSC codes).
- Put `\facilities{}` and `\software{}` before `\appendix` if they are missing.

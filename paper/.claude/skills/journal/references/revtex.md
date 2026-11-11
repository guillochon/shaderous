# APS journals: `revtex` (REVTeX 4.2)

`revtex4-2.cls` and `apsrev4-2.bst` ship with TeX Live and MiKTeX and are on arXiv. The class
option `prd` picks Physical Review D; change it to `prl`, `prx`, `pre`, ... for another APS
journal. Author information: https://journals.aps.org/authors. Check the length limits
there (PRL's is strict and counts figures and equations as words).

## Layouts

| Wrapper | Class options | Use |
|---|---|---|
| `submit.tex` | `[aps,prd,preprint]` | single column, double spaced, for referees |
| `preprint.tex` | `[aps,prd,twocolumn]` | close to the published article; arXiv gets this |

## Front matter

- `\author{}`, `\email{}`, `\affiliation{}` per author; `superscriptaddress` numbers shared
  affiliations. `util/new_paper.py` builds them.
- The abstract goes before `\maketitle`. APS no longer uses PACS codes, and the template
  prints no keywords.

## Body

- APS house style: "Fig. 1" and "Eq. (1)" in running text ("Figure" and "Equation" at the
  start of a sentence), "Table I" with Roman numerals (the class numbers tables that way).
  `util/check_floats.py` cannot match Roman table numbers; check table placement by eye.
- References: numbered, `apsrev4-2`, which omits article titles unless the class option
  `longbibliography` is set. Most authors set it for arXiv.
- `nofootinbib` keeps footnotes out of the reference list.
- `acknowledgments` is a REVTeX environment, so the portable form in the body works as is.
- Supplemental Material is a separate file referenced from the text.

## arXiv

REVTeX is on arXiv; the zip carries only the source, figures and `.bbl`.

## After switching to `revtex`

- Set the journal option (`prd`, `prl`, ...).
- Change figure and equation references to APS style.
- Decide on `longbibliography`.
- Check the length limit for Letters.

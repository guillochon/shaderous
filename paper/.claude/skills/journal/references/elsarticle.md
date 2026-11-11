# Elsevier journals: `elsarticle`

`elsarticle.cls`, `elsarticle-num-names.bst` and `elsarticle-harv.bst` ship with TeX Live and
MiKTeX and are on arXiv. Covers Icarus, Acta Astronautica, New Astronomy, Astronomy and
Computing, Journal of Computational Physics and most other Elsevier titles. Each journal's
"Guide for Authors" page sets its reference style and required statements; read it.

Set `\journal{}` in `main.tex` to the journal name.

## Layouts

| Wrapper | Class options | Use |
|---|---|---|
| `submit.tex` | `[preprint,12pt]` | single column; what Elsevier asks for at submission |
| `preprint.tex` | `[final,5p]` + newtx Times, text width narrowed to 486 pt | two columns, close to the article; arXiv gets this |

The 486 pt text width keeps arXiv's identifier stamp out of the text.

## Front matter (inside `frontmatter`)

- `\author[affN]{Name\corref{cor1}}`, `\ead{email}`, `\cortext[cor1]{...}`,
  `\affiliation[affN]{organization={...}}`. `util/new_paper.py` builds them.
- Abstract: one paragraph; most Elsevier journals cap it at 250 words.
- **Highlights**: 3 to 5 bullets, each at most 85 characters including spaces. The preprint
  layout swallows them because the published article does not print them.
- Keywords: one to seven, separated by `\sep`.
- Some journals want a graphical abstract (a separate file at submission).

## Body

- References: the template uses numbered `elsarticle-num-names` with `sort&compress`. Journals
  that use author-year (Icarus, for one) want `elsarticle-harv`: change
  `\bibliographystyle` and drop `\biboptions{sort&compress}`.
- `elsarticle` does not define the ADS journal macros; `macros.tex` provides them.
- Required statements vary by journal: declaration of competing interest, CRediT author
  contributions, data availability, and a declaration of generative AI use. Add them as
  unnumbered sections before the references.

## arXiv

Everything is on arXiv; the zip carries only the source, figures and `.bbl`.

## After switching to `elsarticle`

- Set `\journal{}` and the reference style for that journal.
- Write the highlights (each under 85 characters).
- Add the required declarations.

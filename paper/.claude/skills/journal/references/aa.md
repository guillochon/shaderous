# Astronomy & Astrophysics: `aa`

A&A distributes its own class, `aa.cls`, and style, `aa.bst`
(https://www.aanda.org/for-authors/latex-issues/). It is on CTAN as `aa`. If `kpsewhich aa.cls`
finds nothing, MiKTeX usually installs it on first use; otherwise download the package and
put `aa.cls`, `aa.bst` and `natbib`-compatible files in `paper/`. `util/arxiv_zip.py` adds any
`.cls` or `.sty` in `paper/` that the source loads, so arXiv gets them.

## Layouts

A&A is two columns at submission and in print; `submit.tex` and `preprint.tex` give the same
layout. Use the `referee` class option only if the editor asks for it.

## Front matter

- `\title{}`, `\titlerunning{}` (from `short_title`), `\author{... \inst{1} \and ...}`,
  `\institute{... \and ...}`, `\authorrunning{}`. `util/new_paper.py` builds them.
- **Structured abstract**, five parts:
  `\abstract{context}{aims}{methods}{results}{conclusions}`. After a switch, the whole old
  abstract sits in the first part; split it. `util/check_abstract.py` counts all five parts
  together.
- Keywords from the A&A keyword list, separated by ` -- `.

## Body

- A&A style in running text: "Fig. 1", "Figs. 1 and 2", "Sect. 2", "Eq. (1)", "Table 1";
  write "Figure", "Section" and "Equation" in full only at the start of a sentence.
  `\citet` / `\citep` with `aa.bst`.
- Acknowledgements use the class environment `acknowledgements` (British spelling); the
  portable `acknowledgments` in `macros.tex` also works.
- Appendices come after the references in A&A; move `\appendix` and what follows to after
  `\bibliography{}`.
- A&A has language editing; still follow the rules in `.cursor/rules/`.

## After switching to `aa`

- Split the abstract into its five parts.
- Replace the keywords with entries from the A&A list.
- Change in-text references to "Fig.", "Sect." and "Eq." style.
- Move the appendix after the bibliography.

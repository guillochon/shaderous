# Mathematics: `amsart`

`amsart.cls` (AMS) ships with every TeX distribution and is on arXiv. It is the usual
layout for arXiv `math.*` preprints and for AMS journals; many other mathematics journals
accept an `amsart` manuscript and reformat it.

## Layouts

`amsart` is single column in both layouts; `submit.tex` and `preprint.tex` are the same.
The template adds 1.1 in margins with `geometry`, Latin Modern fonts and `microtype`.

## Front matter

- `\title[running head]{Title}`, then per author `\author{}`, `\address{}`, `\email{}`.
  `util/new_paper.py` builds them.
- `\subjclass[2020]{Primary 00A00; Secondary 00A00}` with Mathematics Subject Classification
  2020 codes (https://mathscinet.ams.org/msc/msc2020.html), and `\keywords{a, b}`.
- The abstract goes before `\maketitle` in `amsart`.

## Body

- Theorem environments (`theorem`, `lemma`, `proposition`, `corollary`, `definition`,
  `example`, `remark`) are defined in `macros.tex` and numbered within sections, sharing one
  counter. Follow `.cursor/rules/math-notation.mdc`.
- Refer to equations with `\eqref{}`, which prints the parentheses.
- Citations: natbib with `numbers,sort&compress` and `plainnat`, so `\citet` gives
  "Hogg et al. [1]" and `\citep` gives "[1]", consistent with the astronomy classes. AMS
  journals prefer `amsplain` or `amsalpha` with plain `\cite`; switch only at submission,
  because `\citet` does not exist without natbib.
- Add MathSciNet MR numbers to the bib (`mrnumber = {...}`) when the journal wants them.

## arXiv

`amsart` is on arXiv; the zip carries only the source, figures and `.bbl`. Pick the primary
`math.*` category to match the primary MSC code.

## After switching to `amsart`

- Add MSC codes to `\subjclass`.
- Convert `Equation~(\ref{...})` to `\eqref{...}` if you prefer the math convention.
- Delete astronomy-only back matter (`\facilities`) if it is empty.

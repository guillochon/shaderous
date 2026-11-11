# MNRAS: `mnras`

`mnras.cls` and `mnras.bst` ship with TeX Live and MiKTeX (package `mnras`) and are on arXiv.
Author guide: https://academic.oup.com/mnras/pages/general_instructions. Check it for the
current limits; the journal changes them.

## Layouts

MNRAS is two columns at submission and in print, so `submit.tex` and `preprint.tex` give the
same layout. Both exist so that every class builds the same two files.
The template uses `[fleqn,usenatbib]` and newtx Times fonts.

## Front matter

- `\title[running head]{Title}`: the running head comes from `short_title` in `paper.toml`.
- One `\author[short list]{...}` block: names with superscript affiliation numbers, a
  `\thanks{E-mail: ...}` on the corresponding author, then the numbered affiliations after
  `\\`. `util/new_paper.py` builds it from `paper.toml`.
- Abstract: a single paragraph, at most 250 words.
- Keywords: one to six, taken **only** from the MNRAS keyword list (linked from the
  author guide above), written
  `keyword one -- keyword two` in the `keywords` environment. Use the list's exact wording
  including its "stars: individual: ..." style prefixes.

## Body

- British spelling and MNRAS house style (for example "per cent", "colour",
  "modelling"). The copy editors convert, but write it this way to avoid surprises.
- Units as `km\,s$^{-1}$`, not `km/s`.
- Cite with `\citet` / `\citep` (the `usenatbib` option); style `mnras.bst`.
- A **Data Availability** section is mandatory; the template puts it after the body, outside
  the markers, because no other class needs it.
- Appendices conventionally follow the references in MNRAS. The template body puts
  `\appendix` before the bibliography, which MNRAS accepts; move it after `\bibliography{}`
  if the editor asks.
- End with `\bsp` and `\label{lastpage}` (in the template).

## arXiv

`mnras.cls` is on arXiv; nothing is bundled besides the `.bbl`.

## After switching to `mnras`

- Replace the keywords with entries from the MNRAS list.
- Fill in the Data Availability section.
- Convert spelling to British and "%" in prose to "per cent".
- Check that figures fit a column (`\columnwidth`) or use `figure*`.

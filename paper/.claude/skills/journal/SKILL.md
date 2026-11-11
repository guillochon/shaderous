---
name: journal
description: The journal class this paper is written for, and how to format, build and submit for it. Use whenever editing the front matter, title, abstract, author list, keywords, bibliography style, figures or tables of paper/main.tex; when compiling the paper or making the arXiv zip; when the user names a journal (ApJ, AJ, MNRAS, A&A, Icarus, PRD, an AMS journal, ...) or asks to switch, retarget or reformat the paper for one; and before submission.
---

# Target journal

Selected journal class: <!-- selected-journal -->`aastex`<!-- /selected-journal -->

This line, `journal` in `paper.toml`, and the class in `paper/main.tex` must agree.
`python util/new_paper.py` keeps all three in step; never edit one by hand without the others.
`python util/check.py` reports a mismatch.

Read the reference file for the selected class before changing anything journal-specific:

| Class | Journals | Reference |
|---|---|---|
| `aastex` | ApJ, ApJL, ApJS, AJ, PSJ, RNAAS | [references/aastex.md](references/aastex.md) |
| `mnras` | MNRAS, RASTI | [references/mnras.md](references/mnras.md) |
| `aa` | Astronomy & Astrophysics | [references/aa.md](references/aa.md) |
| `elsarticle` | Icarus, Acta Astronautica, New Astronomy, Astronomy and Computing, JCP | [references/elsarticle.md](references/elsarticle.md) |
| `revtex` | PRD, PRL, PRX, PRE and the other APS journals | [references/revtex.md](references/revtex.md) |
| `amsart` | AMS journals and arXiv math.* preprints | [references/amsart.md](references/amsart.md) |

## How the paper is built, whatever the class

- `paper/main.tex` is the only manuscript source. It reads `\papermode` to pick a layout.
  `paper/submit.tex` and `paper/preprint.tex` are three-line wrappers that set it and
  `\input{main}`; `cd paper && latexmk -pdf` builds `submit.pdf` and `preprint.pdf`.
- Everything between `%% ---- BODY ----` and `%% ---- END BODY ----` is journal-independent.
  Keep class-specific markup (author blocks, keyword syntax, highlights, data-availability
  statements) outside the markers so a switch carries the body over intact.
- `paper/macros.tex` holds the paper's own macros, the ADS journal abbreviations, theorem
  environments, and portable `acknowledgments` / `\software` / `\facilities`. It works under
  every class. Use those portable forms in the body rather than a class-specific spelling.
- Citations are natbib `\citet` / `\citep` in every class.

## Switching journal

1. `python util/new_paper.py --switch --journal <class>`: rebuilds the front matter from
   `paper.toml` (authors, keywords, short title), keeps the title, abstract and body, saves the
   old source as `paper/main.tex.bak`, and updates this file and `paper.toml`.
2. Read the new class's reference file and do its "after switching" list by hand.
3. Build both layouts and run `python util/check.py`. Fix every new warning.

## Adding a journal class

Add `templates/journals/<class>/main.tex` with the same placeholders as the others
(`<<TITLE>>`, `<<AUTHORS>>`, `<<ABSTRACT>>`, `<<KEYWORDS>>`, `<<BODY>>`, `<<BIBS>>`, ...), an
author renderer and a keyword separator in `util/new_paper.py`, a row in the table above, and
`references/<class>.md` covering the points the other reference files cover.

# Agent instructions

This repository is one scientific paper. The manuscript is `paper/main.tex`; the settings that every script reads are in `paper.toml`.

The manuscript conventions live in `.cursor/rules/*.mdc` and apply to every agent, not only Cursor. Read the ones that bear on a task before editing the manuscript, the bibliography or the figure scripts:

| Rule | Covers |
|---|---|
| `writing-style.mdc` | voice, tense, pronouns, banned words, section titles, ties, `\todo` |
| `readability.mdc` | sentence-level rules for every prose edit (below) |
| `em-dashes.mdc` | em-dashes, sparingly |
| `abstract.mdc` | 1920-character limit, no macros or citation commands in the title or abstract |
| `verify-citations.mdc` | never cite from memory; check every bib change |
| `figure-design.mdc` | one script per figure, shared style, placement |
| `math-notation.mdc` | equations, notation, theorems and proofs |
| `numbers-and-units.mdc` | traceable numbers, precision, units |
| `research-sources.mdc` | where downloaded papers and notes go |
| `compile.mdc` | how to build and what a clean build is |
| `proposed-edits.mdc` | large revisions as numbered, reviewable edits |

The target journal and how to format for it are in the `journal` skill (`.claude/skills/journal/SKILL.md`). Use it for anything journal-specific: front matter, keywords, reference style, switching class, submission.

Every prose edit follows these readability rules:

@.cursor/rules/readability.mdc

## Before finishing a change

```bash
python util/check.py            # journal agreement, abstract, citation keys and order, style
python util/build.py            # both layouts compile cleanly
```

After a bibliography change, also `python util/check_citations.py`. Before arXiv, `python util/arxiv_zip.py`.

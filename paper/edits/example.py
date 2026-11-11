# ruff: noqa: E501  (the edits are exact LaTeX passages)
"""Example proposal file for util/edits.py. Copy it to edits/<topic>.py and replace ITEMS.

    python util/edits.py render edits/example.py          # edits/example.html, one numbered item per change
    python util/edits.py apply edits/example.py 1,2       # apply only the approved items

Rule numbers in a title refer to .cursor/rules/readability.mdc:
  1 human verbs for people   2 keep the optional "that"   3 subject, verb and object close
  4 at most two stacked nouns   5 the common phrase   6 no ad hoc noun-to-verb coinages
"""

# The file to edit, relative to the repo root. Default: paper/main.tex.
TARGET = "paper/main.tex"

ITEMS = [
    # 1
    (
        "Methods: make 'we' the subject of the sampling sentence [1]",
        [
            (
                "sub",
                r"and we sample the posterior with an affine-invariant ensemble sampler",
                r"and we draw samples from the posterior with an affine-invariant ensemble sampler",
            ),
        ],
    ),
    # 2
    (
        "Methods: one claim per sentence; split at the semicolon [3]",
        [
            (
                "sub",
                r"assumes that the uncertainties are known; Section~\ref{sec:discussion} discusses",
                r"assumes that the uncertainties are known. Section~\ref{sec:discussion} discusses",
            ),
        ],
    ),
]

# Item numbers that must run before the others (lower runs first; the default is 1).
PRIORITY = {}

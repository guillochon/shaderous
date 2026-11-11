#!/usr/bin/env python3
"""Check the title and abstract of paper/main.tex for arXiv and journal forms.

Counts the collapsed abstract (TeX commands included, whitespace squeezed to
single spaces) against abstract_limit in paper.toml, 1920 characters for arXiv.
Exits nonzero if the abstract is over the limit, if the abstract or the title
uses a macro defined in the preamble or paper/macros.tex or a citation command,
or if the abstract opens with a citation command.

    python util/check_abstract.py
"""

from __future__ import annotations

import re
import sys

from paperconf import abstract, flatten, load, preamble_macros, strip_comments, title, utf8_stdout

CITE_COMMANDS = {
    "cite",
    "citealp",
    "citealt",
    "citeauthor",
    "citep",
    "citet",
    "citeyear",
    "citeyearpar",
    "nocite",
}
COMMAND = re.compile(r"\\([A-Za-z@]+)")


def main() -> int:
    utf8_stdout()
    limit = load()["paper"].get("abstract_limit", 1920)
    tex = strip_comments(flatten())
    banned = preamble_macros(tex) | CITE_COMMANDS
    body = abstract(tex)
    words = len(re.sub(r"\\[A-Za-z]+|[{}$]", " ", body).split())
    forbidden = sorted(set(COMMAND.findall(body)) & banned)
    in_title = sorted(set(COMMAND.findall(title(tex))) & banned)
    leads = body.startswith("\\cite")
    print(f"abstract characters: {len(body)} / {limit}")
    print(f"abstract words: {words}")
    print("forbidden macros: " + (", ".join(forbidden) if forbidden else "none"))
    print("forbidden macros in the title: " + (", ".join(in_title) if in_title else "none"))
    print("opens with a citation command: " + ("yes" if leads else "no"))
    if len(body) <= limit and not forbidden and not in_title and not leads:
        return 0
    if len(body) > limit:
        print(f"abstract is {len(body) - limit} characters over the limit")
    if forbidden:
        print("abstract uses forbidden macros")
    if in_title:
        print("title uses forbidden macros")
    if leads:
        print("abstract opens with a citation command")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Lint the manuscript prose for the mechanical parts of .cursor/rules/.

Reads the body of paper/main.tex and every paper/ file it \\inputs, with comments
removed. Fails on:
  - gendered pronouns (writing-style.mdc: other authors are they/them),
  - words listed in paper.toml [style] banned.
Warns on:
  - em-dashes in prose above [style] max_em_dashes_per_1000 (em-dashes.mdc),
  - a doubled word ("the the"),
  - a common British spelling (writing-style.mdc: American spelling),
  - "Figure \\ref" without a tie (writing-style.mdc),
  - every \\todo left in the text (writing-style.mdc). --submission makes these fail.

A source line that ends in "% style-ok" is skipped (for "Her X-1", "He II", ...).

    python util/check_style.py
    python util/check_style.py --submission
"""

from __future__ import annotations

import argparse
import re
import sys

from paperconf import MAIN, ROOT, load, sources, split_comment, utf8_stdout

PRONOUN_ANY = re.compile(r"\b(he|she|his|her|hers|him|himself|herself)\b")
PRONOUN_CAP = re.compile(r"\b(She|His|Hers|Him|Himself|Herself)\b")
# Capitalized He/Her only where a sentence starts, and not as helium (He II) or Hercules (Her X-1).
PRONOUN_START = re.compile(r"(?:^|[.!?]\s+)(He|Her)\b(?![~\s]*(?:I{1,3}V?|IV|V|X-\d)\b|-)")
DOUBLED = re.compile(r"\b([A-Za-z]+)\s+\1\b", re.I)
UNTIED = re.compile(r"\b(Figures?|Figs?\.|Tables?|Sections?|Sects?\.|Equations?|Eqs?\.|Appendix|Theorem|Lemma|Proposition)\s+\\(?:ref|eqref)\b")
BRITISH = re.compile(
    r"\b("
    r"(?:col|behavi|fav|flav|hon|harb|lab|neighb|vap|rum|hum)our(?:s|ed|ing|ful|able)?"
    r"|(?:normal|parametr|parameter|minim|maxim|optim|character|organ|real|recogn|util|standard|general|summar"
    r"|emphas|categor|local|special|stabil|initial|visual|polar|synchron|random|regular|linear|marginal|discret"
    r"|quant|vectori|ional|symmetr|homogen|central|neutral|ion|digit|penal|capital|criti|theor|apolog)is(?:e|ed|es|ing|ation|ations)"
    r"|analys(?:e|ed|ing)|paralys(?:e|ed|ing)|catalys(?:e|ed|ing)"
    r"|(?:cent|met|fib|lit|theat|calib)re[sd]?"
    r"|(?:model|label|travel|signal|cancel|fuel|tunnel|channel|total|level)l(?:ed|ing|er|ers)"
    r"|grey|aluminium|sulphur\w*|programme[sd]?|whilst|amongst|focuss(?:ed|ing|es)|defence|offence|licence|judgement"
    r")\b",
    re.I,
)
TODO = re.compile(r"\\todo\{")
EM_DASH = re.compile(r"(?<!-)---(?!-)|\u2014")
JOURNAL_MARKUP = re.compile(r"\\keywords|Note---|\\begin\{keywords\}")


def prose_lines():
    """(path, line number, code) for every body line that is not skipped."""
    for path in sources():
        in_body = path != MAIN
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not in_body:
                in_body = r"\begin{document}" in line
                continue
            if line.rstrip().endswith("% style-ok"):
                continue
            code = split_comment(line)[0]
            if code.strip():
                yield path.relative_to(ROOT).as_posix(), n, code


def main() -> int:
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--submission", action="store_true", help="fail on any \\todo left in the text")
    args = parser.parse_args()
    style = load().get("style", {})
    banned = [w.lower() for w in style.get("banned", [])]
    banned_re = re.compile(r"\b(" + "|".join(map(re.escape, banned)) + r")\b", re.I) if banned else None
    max_rate = float(style.get("max_em_dashes_per_1000", 2.0))

    failures, warnings, todos = [], [], []
    words = dashes = 0
    for where, n, code in prose_lines():
        loc = f"{where}:{n}"
        # \cite keys, \ref labels and command names are not prose.
        text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{[^{}]*\}", lambda m: m.group(0) if m.group(0).startswith(("\\emph", "\\textit", "\\textbf", "\\todo")) else " ", code)
        text = re.sub(r"\$[^$]*\$", " ", text)
        for m in [*PRONOUN_ANY.finditer(text), *PRONOUN_CAP.finditer(text), *PRONOUN_START.finditer(text)]:
            failures.append(f"{loc}: gendered pronoun '{m.group(1)}' (use they/them; '% style-ok' if it is not a pronoun)")
        if banned_re:
            for m in banned_re.finditer(text):
                failures.append(f"{loc}: banned word '{m.group(1)}'")
        for m in DOUBLED.finditer(text):
            warnings.append(f"{loc}: doubled word '{m.group(0)}'")
        for m in BRITISH.finditer(text):
            warnings.append(f"{loc}: British spelling '{m.group(1)}' (use American; '% style-ok' if it is a quotation or name)")
        for m in UNTIED.finditer(code):
            warnings.append(f"{loc}: '{m.group(0)}' needs a tie: {m.group(1)}~\\ref")
        if TODO.search(code):
            todos.append(f"{loc}: {code.strip()[:90]}")
        if not JOURNAL_MARKUP.search(code):
            dashes += len(EM_DASH.findall(code))
        words += len(text.split())

    rate = 1000 * dashes / max(words, 1)
    print(f"prose words: {words}; em-dashes: {dashes} ({rate:.1f} per 1000, limit {max_rate})")
    if rate > max_rate:
        warnings.append(f"em-dashes: {rate:.1f} per 1000 words is over {max_rate}; prefer commas, parentheses or colons")
    for line in failures:
        print("FAIL " + line)
    for line in warnings:
        print("warn " + line)
    if todos:
        print(f"{len(todos)} \\todo left:")
        for line in todos:
            print("  " + line)
    fail = bool(failures) or (args.submission and bool(todos))
    print("style: " + ("problems" if fail else "OK"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

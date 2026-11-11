#!/usr/bin/env python3
"""Check where figures and tables land in the compiled PDF.

For each figure and table, finds the page of its first mention in the text and
the page of its caption, and flags a float that lands more than one page after
its first reference, one that comes before it, and two figures on one page.
Needs pdftotext (poppler; in Git for Windows, TeX Live and most Linux systems).

    python util/check_floats.py                     # paper/<arxiv_mode>.pdf
    python util/check_floats.py paper/submit.pdf

Captions are matched as "Figure 1." / "Figure 1:" / "Fig. 1." / "FIG. 1." and
"Table 1" / "TABLE 1". REVTeX numbers tables in Roman numerals; those are not
matched, so check REVTeX tables by eye.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from paperconf import PAPER, load, utf8_stdout

KINDS = {
    "Figure": (r"(?:Figure|Fig\.|FIG\.)", r"(?:Figures?|Figs?\.|FIGS?\.)"),
    "Table": (r"(?:Table|TABLE)", r"(?:Tables?|TABLES?)"),
}


def pages(pdf: Path) -> list[str]:
    try:
        out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", check=True).stdout
    except FileNotFoundError:
        raise SystemExit("pdftotext is not on the PATH (install poppler)") from None
    return out.split("\f")


def main() -> int:
    utf8_stdout()
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER / f"{load()['paper'].get('arxiv_mode', 'preprint')}.pdf"
    if not pdf.is_file():
        raise SystemExit(f"no {pdf}; build first (python util/build.py)")
    pg = pages(pdf)
    ok = True
    fig_pages: dict[int, list[int]] = {}
    ref_list = next((i + 1 for i, t in enumerate(pg) if re.search(r"^\s*REFERENCES|^\s*References\s*$", t, re.M)), len(pg))
    for kind, (cap_word, ref_word) in KINDS.items():
        n = 1
        while True:
            cap = re.compile(rf"^\s*{cap_word}\s*{n}[.:]", re.M)
            cap_p = next((i + 1 for i, t in enumerate(pg) if cap.search(t)), None)
            if cap_p is None:
                break
            ref = re.compile(rf"{ref_word}\s*(?:\d+[a-z]?(?:,\s*|\s+and\s+|--|–))*{n}(?![0-9])")

            def mentions(text: str, ref: re.Pattern = ref, cap: re.Pattern = cap) -> bool:
                for m in ref.finditer(text):
                    line_start = text.rfind("\n", 0, m.start()) + 1
                    if cap.match(text, line_start):
                        continue  # the caption itself
                    before = text[max(0, m.start() - 6) : m.start()]
                    if before.endswith("their ") or re.match(r"\s+of\s", text[m.end() : m.end() + 6]):
                        continue  # another paper's figure
                    return True
                return False

            ref_p = next((i + 1 for i, t in enumerate(pg) if mentions(t)), None)
            flag = ""
            if ref_p is None:
                flag = "  <-- never referenced in the text"
                ok = False
            elif kind == "Table" and cap_p > ref_list:
                flag = "  (appendix)"
            elif cap_p > ref_p + 1:
                flag = f"  <-- {cap_p - ref_p} pages after first reference"
                ok = False
            elif cap_p < ref_p:
                flag = "  <-- appears before first reference"
                ok = False
            print(f"{kind} {n}: first reference p.{ref_p}, caption p.{cap_p}{flag}")
            if kind == "Figure":
                fig_pages.setdefault(cap_p, []).append(n)
            n += 1
    for p, figs in fig_pages.items():
        if len(figs) > 1:
            print(f"two figures on page {p}: {figs}")
            ok = False
    print("OK" if ok else "PLACEMENT PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

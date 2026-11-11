#!/usr/bin/env python3
"""Put the keys inside every multi-key \\cite command in year order.

Years come from the bibliography files in paper.toml; ties keep the order the
author wrote. Rewrites paper/main.tex and every paper/ file it \\inputs.

    python util/sort_citations.py            # rewrite in place
    python util/sort_citations.py --check    # exit 1 if anything is out of order
"""

from __future__ import annotations

import re
import sys

from paperconf import ROOT, bibs, sources, utf8_stdout

ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", re.S)
YEAR = re.compile(r"\byear\s*=\s*[{\"]?(\d{4})", re.I)
CITE = re.compile(r"(\\cite(?:t|p|alt|alp)?\*?(?:\[[^\]]*\]){0,2}\{)([^}]*)(\})")


def years() -> dict[str, int]:
    out: dict[str, int] = {}
    for bib in bibs():
        for m in ENTRY.finditer(bib.read_text(encoding="utf-8")):
            y = YEAR.search(m.group(2))
            if y:
                out[m.group(1)] = int(y.group(1))
    return out


def main() -> int:
    utf8_stdout()
    check = "--check" in sys.argv
    yr = years()
    changed = 0
    missing: set[str] = set()

    def fix(m: re.Match) -> str:
        nonlocal changed
        keys = [k.strip() for k in m.group(2).split(",")]
        if len(keys) < 2:
            return m.group(0)
        missing.update(k for k in keys if k not in yr)
        ordered = sorted(keys, key=lambda k: yr.get(k, 9999))  # stable: ties keep author order
        if ordered != keys:
            changed += 1
            if check:
                print(f"  {m.group(0)}")
        return m.group(1) + ", ".join(ordered) + m.group(3)

    for path in sources():
        raw = path.read_text(encoding="utf-8")
        new = CITE.sub(fix, raw)
        if new != raw and not check:
            path.write_text(new, encoding="utf-8")
            print(f"rewrote {path.relative_to(ROOT).as_posix()}")
    if missing:
        print("no year found for:", ", ".join(sorted(missing)))
    if check:
        print(f"{changed} citation(s) out of chronological order")
        return 1 if changed else 0
    print(f"reordered {changed} citation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Review and apply a list of proposed edits to the manuscript, one numbered item at a time.

A proposal file (see edits/example.py and .cursor/rules/proposed-edits.mdc)
defines ITEMS, a list of (title, [edit, ...]), and optionally PRIORITY and
TARGET. An edit is one of
  ("sub", old, new)            exact substring; must occur exactly once
  ("para", prefix, new)        the whole source line that starts with prefix; new=None deletes it
  ("span", first, last, new)   every line from the one starting with first through the
                               one starting with last (paragraph breaks included)
Items run in PRIORITY order and then by number, so a rewrite of a whole
paragraph can run before an item that edits a clause inside it.

    python util/edits.py render edits/cuts.py           # writes edits/cuts.html for review
    python util/edits.py apply edits/cuts.py 1,3,5-9    # applies only the approved items
"""

from __future__ import annotations

import argparse
import difflib
import html
import importlib.util
import re
import sys
from pathlib import Path

from paperconf import MAIN, ROOT, utf8_stdout


def load_proposal(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = getattr(module, "TARGET", None)
    module.TARGET = (ROOT / target) if target else MAIN
    module.PRIORITY = getattr(module, "PRIORITY", {})
    return module


def find_line(lines, prefix):
    hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"line prefix matched {len(hits)} times: {prefix[:60]!r}")
    return hits[0]


def apply_edit(text, edit):
    """Return (new text, before snippet, after snippet)."""
    kind = edit[0]
    if kind == "sub":
        _, old, new = edit
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"substring matched {n} times: {old[:60]!r}")
        return text.replace(old, new), old, new
    lines = text.split("\n")
    if kind == "para":
        _, prefix, new = edit
        i = find_line(lines, prefix)
        before = lines[i]
        if new is None:
            # drop the line and the blank line after it
            end = i + 2 if i + 1 < len(lines) and not lines[i + 1].strip() else i + 1
            del lines[i:end]
            return "\n".join(lines), before, ""
        lines[i] = new
        return "\n".join(lines), before, new
    if kind == "span":
        _, first, last, new = edit
        i, j = find_line(lines, first), find_line(lines, last)
        before = "\n".join(lines[i : j + 1])
        lines[i : j + 1] = [new]
        return "\n".join(lines), before, new
    raise ValueError(kind)


def parse_numbers(spec, count):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [n for n in out if not 1 <= n <= count]
    if bad:
        raise SystemExit(f"no such item: {bad}")
    return out


def words(s):
    return len(re.sub(r"\\[A-Za-z]+|[{}$\\]", " ", s).split())


def diff_html(before, after):
    a, b = before.split(" "), after.split(" ")
    left, right = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        old = html.escape(" ".join(a[i1:i2]))
        new = html.escape(" ".join(b[j1:j2]))
        if op == "equal":
            left.append(old)
            right.append(new)
        else:
            if old:
                left.append(f"<del>{old}</del>")
            if new:
                right.append(f"<ins>{new}</ins>")
    return " ".join(left), " ".join(right)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Proposed edits</title><style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#666; --line:#ddd; --del:#fde2e2; --delfg:#9b1c1c; --ins:#dcf5e3; --insfg:#14632f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17181a; --fg:#e8e8e8; --muted:#9a9a9a; --line:#333; --del:#4a1f22; --delfg:#ffb3b3; --ins:#173a24; --insfg:#a8e6b8; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1200px; padding:16px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:17px; margin:0 0 4px; }} h4 {{ margin:0 0 4px; font-size:12px; text-transform:uppercase; color:var(--muted); }}
section {{ border-top:1px solid var(--line); padding:16px 0; }}
.saved {{ color:var(--muted); font-size:13px; margin-bottom:8px; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:10px; }}
@media (max-width:700px) {{ .pair {{ grid-template-columns:1fr; }} }}
p {{ font-family:ui-monospace, Consolas, monospace; font-size:12.5px; white-space:pre-wrap; overflow-wrap:anywhere; margin:0; }}
p.lede {{ font-family:inherit; font-size:14px; }}
del {{ background:var(--del); color:var(--delfg); text-decoration:none; }}
ins {{ background:var(--ins); color:var(--insfg); text-decoration:none; }}
</style></head><body>
<h1>Proposed edits to {target}</h1>
<p class=lede>{count} items, about {total} words saved in all. Each item is shown against the current source. Red is removed, green is added. Reply with the numbers to apply.</p>
{blocks}
</body></html>"""


def render(proposal, out: Path):
    text = proposal.TARGET.read_text(encoding="utf-8").replace("\r\n", "\n")
    blocks, total = [], 0
    for n, (title, edits) in enumerate(proposal.ITEMS, 1):
        saved, rows = 0, []
        for edit in edits:
            # each item is shown against the current source, independent of the others
            _, before, after = apply_edit(text, edit)
            saved += words(before) - words(after)
            left, right = diff_html(before, after)
            rows.append(f"<div class=pair><div><h4>Before</h4><p>{left}</p></div>"
                        f"<div><h4>After</h4><p>{right or '<em>(deleted)</em>'}</p></div></div>")
        total += saved
        blocks.append(f"<section><h2>{n}. {html.escape(title)}</h2>"
                      f"<div class=saved>saves about {saved} words</div>{''.join(rows)}</section>")
    target = proposal.TARGET.relative_to(ROOT).as_posix()
    out.write_text(PAGE.format(target=target, count=len(proposal.ITEMS), total=total, blocks="".join(blocks)),
                   encoding="utf-8")
    print(f"wrote {out}: {len(proposal.ITEMS)} items, about {total} words")


def apply(proposal, numbers):
    raw = proposal.TARGET.read_text(encoding="utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for n in sorted(numbers, key=lambda n: (proposal.PRIORITY.get(n, 1), n)):
        for edit in proposal.ITEMS[n - 1][1]:
            text, _, _ = apply_edit(text, edit)
        print(f"applied {n}. {proposal.ITEMS[n - 1][0]}")
    proposal.TARGET.write_text(text.replace("\n", "\r\n") if crlf else text, encoding="utf-8", newline="")


def main():
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("render", help="write an HTML review page next to the proposal")
    r.add_argument("proposal", type=Path)
    r.add_argument("--out", type=Path, help="HTML path (default: the proposal with .html)")
    a = sub.add_parser("apply", help="apply the numbered items to the manuscript")
    a.add_argument("proposal", type=Path)
    a.add_argument("numbers", help="e.g. 1,3,5-9")
    args = parser.parse_args()
    proposal = load_proposal(args.proposal)
    if args.command == "render":
        render(proposal, args.out or args.proposal.with_suffix(".html"))
    else:
        apply(proposal, parse_numbers(args.numbers, len(proposal.ITEMS)))


if __name__ == "__main__":
    sys.exit(main())

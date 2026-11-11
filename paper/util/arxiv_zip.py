#!/usr/bin/env python3
"""Zip the files arXiv needs to compile the paper, and test-compile the zip.

arXiv compiles a single top-level file, so the zip's main.tex is paper/main.tex
with every \\input of a paper/ file (macros.tex, section files) spliced in and
\\def\\papermode{<arxiv_mode>} prepended: the same source, typeset as the layout
that paper.toml names (preprint by default). Comment text is removed, because
arXiv publishes the source; each comment's % is kept so line ends still behave.
The bibliography goes in as main.bbl from a fresh build in a temporary
directory, so arXiv needs neither BibTeX nor the .bib files. Every figure named
by \\includegraphics keeps its path. A .cls or .sty in paper/ that the source
loads goes in too; classes in arXiv's TeX Live do not.

    python util/arxiv_zip.py              # build, zip to <slug>-arxiv.zip, test-compile the zip
    python util/arxiv_zip.py --out PATH   # choose the zip path
    python util/arxiv_zip.py --keep-comments
    python util/arxiv_zip.py --no-verify  # skip the test compile of the zip
    python util/arxiv_zip.py --dry-run    # list files, do not write
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from build import compile_job, report
from paperconf import PAPER, ROOT, flatten, load, split_comment, strip_comments, utf8_stdout

# Optional [...] before the brace, so \includegraphics[width=...]{...} matches.
GRAPHICS_CMD = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
PACKAGE_CMD = re.compile(r"\\(?:documentclass|usepackage|RequirePackage)\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
GRAPHIC_EXTS = ["", ".pdf", ".png", ".jpg", ".jpeg", ".eps"]


def drop_comment_text(tex: str) -> str:
    """Remove comment text. A line that is only a comment goes; otherwise the % stays."""
    out = []
    for line in tex.splitlines():
        code, comment = split_comment(line)
        if not comment:
            out.append(line)
        elif code.strip():
            out.append(code + "%")
    return "\n".join(out) + "\n"


def figures(tex: str) -> list[str]:
    """Figure files relative to paper/, in first-use order, with their extensions."""
    ordered, missing = [], []
    for match in GRAPHICS_CMD.finditer(strip_comments(tex)):
        name = match.group(1).strip()
        found = next((name + ext for ext in GRAPHIC_EXTS if (PAPER / (name + ext)).is_file()), None)
        if found is None:
            missing.append(name)
        elif found not in ordered:
            ordered.append(found)
    if missing:
        print("Missing from paper/:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)
    return ordered


def local_packages(tex: str) -> list[str]:
    """Class and style files that live in paper/ and that the source loads."""
    out = []
    for match in PACKAGE_CMD.finditer(strip_comments(tex)):
        for name in match.group(1).split(","):
            for ext in (".cls", ".sty"):
                fname = name.strip() + ext
                if (PAPER / fname).is_file() and fname not in out:
                    out.append(fname)
    return out


def collect(bbl: Path, mode: str, keep_comments: bool) -> list[tuple[str, bytes]]:
    """(zip name, bytes) pairs, main.tex first."""
    tex = flatten()
    names = figures(tex)
    body = tex if keep_comments else drop_comment_text(tex)
    preamble = f"% arXiv build of the {mode} layout.\n\\def\\papermode{{{mode}}}\n"
    items = [("main.tex", (preamble + body).encode("utf-8"))]
    bbl_bytes = bbl.read_bytes()
    if b"\\bibitem" not in bbl_bytes:
        print(f"{bbl} has no \\bibitem entries; BibTeX did not find the references", file=sys.stderr)
        sys.exit(1)
    items.append(("main.bbl", bbl_bytes))
    items += [(name, (PAPER / name).read_bytes()) for name in local_packages(tex)]
    items += [(name, (PAPER / name).read_bytes()) for name in names]
    return items


def verify(zip_path: Path) -> bool:
    """Unpack the zip in a clean directory and compile it the way arXiv does: no BibTeX."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # A clean environment, so nothing outside the zip (paper/ on a search path) can be found.
        env = {k: v for k, v in os.environ.items() if k not in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS")}
        cmd = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
        for _ in range(3):
            subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
        problems, warnings = report(Path(tmp) / "main.log")
        pdf = Path(tmp) / "main.pdf"
        if not pdf.is_file():
            problems.append("no main.pdf")
        for line in problems:
            print(f"  zip test compile: {line}")
        for line in warnings:
            print(f"  zip test compile: warning: {line}")
        if not problems:
            print(f"  zip test compile: clean ({pdf.stat().st_size / 1024:.0f} KB PDF)")
        return not problems


def main() -> int:
    utf8_stdout()
    conf = load()["paper"]
    default_zip = ROOT / f"{conf.get('slug', 'paper')}-arxiv.zip"
    mode = conf.get("arxiv_mode", "preprint")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=default_zip,
                        help=f"zip path (default: {default_zip.name} in the repository root)")
    parser.add_argument("--keep-comments", action="store_true", help="keep TeX comments in main.tex")
    parser.add_argument("--no-verify", action="store_true", help="do not test-compile the zip")
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes, do not write")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        print(f"building {mode}.tex in a temporary directory")
        result = compile_job(mode, Path(tmp))
        problems, _ = report(Path(tmp) / f"{mode}.log")
        if result.returncode != 0 or problems:
            print((result.stdout or "")[-3000:], file=sys.stderr)
            for line in problems:
                print(f"  {line}", file=sys.stderr)
            print("the build failed; fix it before zipping (python util/build.py)", file=sys.stderr)
            return 1
        items = collect(Path(tmp) / f"{mode}.bbl", mode, args.keep_comments)

    width = max(len(name) for name, _ in items)
    total = 0
    for name, data in items:
        total += len(data)
        print(f"  {name:{width}s} {len(data) / 1024:8.1f} KB")
    print(f"  {'total':{width}s} {total / 1024:8.1f} KB")
    if args.dry_run:
        return 0

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in items:
            zf.writestr(name, data)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    if not args.no_verify and not verify(out):
        print("the zip does not compile cleanly on its own; do not upload it", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

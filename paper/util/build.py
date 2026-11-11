#!/usr/bin/env python3
"""Compile the paper and report what the logs say.

    python util/build.py                 # build submit.pdf and preprint.pdf in paper/
    python util/build.py preprint        # build one layout
    python util/build.py --report-only   # read the existing logs, do not compile

Uses latexmk when it runs (MiKTeX's latexmk needs Perl) and the
pdflatex / bibtex / pdflatex / pdflatex cycle otherwise. Reports errors,
undefined references and citations, multiply defined labels, overfull boxes
wider than 1 pt and floats too large for the page. Exits nonzero on an error or
on anything undefined.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

from paperconf import PAPER, utf8_stdout

JOBS = ["submit", "preprint"]
OVERFULL = re.compile(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)[^\n]*lines? (\d+)")


@cache
def latexmk_works() -> bool:
    try:
        probe = subprocess.run(["latexmk", "-v"], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return probe.returncode == 0


def tex_env(src: Path = PAPER) -> dict:
    """BibTeX runs in the output directory, so tell it where src keeps the .bib and .bst files."""
    env = dict(os.environ)
    for var in ("BIBINPUTS", "BSTINPUTS"):
        env[var] = f"{src}{os.pathsep}{env.get(var, '')}"
    return env


def compile_job(job: str, outdir: Path | None = None, src: Path = PAPER, quiet: bool = True) -> subprocess.CompletedProcess:
    """Typeset src/<job>.tex into outdir (default: src). Returns the last process run."""
    outdir = outdir or src
    outdir.mkdir(parents=True, exist_ok=True)
    kwargs = {"cwd": src, "env": tex_env(src), "text": True, "capture_output": quiet,
              "encoding": "utf-8", "errors": "replace"}
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={outdir}", f"{job}.tex"]
    if latexmk_works():
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={outdir}", f"{job}.tex"]
        result = subprocess.run(cmd, **kwargs)
        aux, bbl = outdir / f"{job}.aux", outdir / f"{job}.bbl"
        # latexmk on Windows now and then skips BibTeX in a fresh output directory.
        if result.returncode == 0 and not bbl.exists() and aux.is_file() and "\\bibdata" in aux.read_text(errors="replace"):
            subprocess.run(["bibtex", job], **{**kwargs, "cwd": outdir})
            subprocess.run(latex, **kwargs)
            result = subprocess.run(latex, **kwargs)
        return result
    result = subprocess.run(latex, **kwargs)
    if result.returncode != 0:
        return result
    subprocess.run(["bibtex", job], **{**kwargs, "cwd": outdir})
    subprocess.run(latex, **kwargs)
    return subprocess.run(latex, **kwargs)


def report(log_path: Path) -> tuple[list[str], list[str]]:
    """(problems that fail the build, warnings) read from a LaTeX log."""
    if not log_path.is_file():
        return [f"no log at {log_path}"], []
    log = log_path.read_text(encoding="utf-8", errors="replace")
    problems, warnings = [], []
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("! "):
            # TeX prints the offending source line as "l.<n> ..." a few lines below the error.
            where = next((s.strip() for s in lines[i + 1 : i + 8] if s.startswith("l.")), "")
            problems.append("error: " + line[2:] + (f" [{where}]" if where else ""))
    undefined_refs = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
    undefined_cites = sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", log)))
    if undefined_refs:
        problems.append("undefined references: " + ", ".join(undefined_refs))
    if undefined_cites:
        problems.append("undefined citations: " + ", ".join(undefined_cites))
    multiply = sorted(set(re.findall(r"Label `([^']+)' multiply defined", log)))
    if multiply:
        warnings.append("multiply defined labels: " + ", ".join(multiply))
    for width, line in OVERFULL.findall(log):
        if float(width) > 1.0:
            warnings.append(f"overfull hbox {width}pt at source line {line}")
    if "Float too large" in log:
        warnings.append("a float is too large for the page")
    if "Label(s) may have changed" in log:
        warnings.append("labels may have changed; run the build again")
    return problems, warnings


def main() -> int:
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("jobs", nargs="*", default=JOBS, help="layouts to build (default: submit preprint)")
    parser.add_argument("--report-only", action="store_true", help="read existing logs; do not compile")
    args = parser.parse_args()
    failed = False
    for job in args.jobs:
        if not args.report_only:
            print(f"building {job}.pdf with {'latexmk' if latexmk_works() else 'pdflatex + bibtex'}")
            result = compile_job(job)
            if result.returncode != 0:
                print(f"  the build of {job} exited with {result.returncode}")
        problems, warnings = report(PAPER / f"{job}.log")
        for line in problems:
            print(f"  {job}: {line}")
        for line in warnings:
            print(f"  {job}: warning: {line}")
        if not problems and not warnings:
            print(f"  {job}: clean")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

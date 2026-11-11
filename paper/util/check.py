#!/usr/bin/env python3
"""Run every offline check in one go: the pre-commit and CI entry point.

    python util/check.py               # offline checks
    python util/check.py --build       # also compile both layouts and read the logs
    python util/check.py --submission  # also fail on any \\todo left in the text

Checks, in order: the journal named in paper.toml, the journal skill and
paper/main.tex agree; the title and abstract (check_abstract.py); citation keys
(check_citations.py --keys); citation order (sort_citations.py --check); prose
style (check_style.py). The online citation check is not run here because it
needs the network; run util/check_citations.py after any bib change.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from paperconf import JOURNALS, MAIN, ROOT, SKILL, load, utf8_stdout

# The \documentclass each template uses, to recognise the class in paper/main.tex.
CLASS_OF = {}
for template in JOURNALS.glob("*/main.tex"):
    match = re.search(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", template.read_text(encoding="utf-8"))
    if match:
        CLASS_OF[template.parent.name] = match.group(1)


def journal_agrees() -> bool:
    journal = load()["paper"]["journal"]
    skill = re.search(r"<!-- selected-journal -->`([^`]*)`", SKILL.read_text(encoding="utf-8"))
    used = re.search(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", MAIN.read_text(encoding="utf-8"))
    problems = []
    if not skill or skill.group(1) != journal:
        problems.append(f"the journal skill names {skill.group(1) if skill else 'nothing'}")
    if not used or used.group(1) != CLASS_OF.get(journal):
        problems.append(f"paper/main.tex uses {used.group(1) if used else 'no class'}, not {CLASS_OF.get(journal)}")
    if problems:
        print(f"paper.toml says {journal}, but " + "; ".join(problems))
        print("run: python util/new_paper.py --switch --journal " + journal)
        return False
    print(f"journal: {journal} ({CLASS_OF[journal]}) in paper.toml, the skill and main.tex")
    return True


def run(*args: str) -> bool:
    print(f"\n$ python {' '.join(args)}", flush=True)
    return subprocess.run([sys.executable, *args], cwd=ROOT).returncode == 0


def main() -> int:
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", action="store_true", help="compile both layouts too")
    parser.add_argument("--submission", action="store_true", help="fail on \\todo")
    args = parser.parse_args()
    results = {"journal": journal_agrees()}
    results["abstract"] = run("util/check_abstract.py")
    results["citation keys"] = run("util/check_citations.py", "--keys")
    results["citation order"] = run("util/sort_citations.py", "--check")
    results["style"] = run("util/check_style.py", *(["--submission"] if args.submission else []))
    if args.build:
        results["build"] = run("util/build.py")
    print("\n" + "\n".join(f"{'ok  ' if ok else 'FAIL'} {name}" for name, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

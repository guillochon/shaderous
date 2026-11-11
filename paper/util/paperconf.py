"""Shared settings and TeX helpers for the scripts in util/.

paper.toml at the repository root holds the settings. Every script reads it
through load(), so a new paper changes one file instead of every script.
The TeX helpers are deliberately small: they find the title, the abstract,
the preamble macros and the \\input tree of paper/main.tex, which is all the
checks need. Python 3.11+ and the standard library only.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
CONFIG = ROOT / "paper.toml"
MAIN = PAPER / "main.tex"
JOURNALS = ROOT / "templates" / "journals"
SKILL = ROOT / ".claude" / "skills" / "journal" / "SKILL.md"

INPUT = re.compile(r"\\(input|include)\s*\{([^}]+)\}")
MACRO_DEF = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator|def)\s*\*?\s*\{?\\([A-Za-z@]+)"
)
TITLE = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{")


def utf8_stdout() -> None:
    """Windows consoles default to cp1252; the reports print TeX and Unicode."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def load() -> dict:
    with CONFIG.open("rb") as fh:
        return tomllib.load(fh)


def bibs(conf: dict | None = None) -> list[Path]:
    conf = conf or load()
    return [PAPER / name for name in conf["paper"].get("bibs", ["references.bib"])]


def split_comment(line: str) -> tuple[str, str]:
    """(code, comment) for one line. A % after an odd run of backslashes is literal."""
    i = 0
    while True:
        j = line.find("%", i)
        if j < 0:
            return line, ""
        k = j - 1
        while k >= 0 and line[k] == "\\":
            k -= 1
        if (j - 1 - k) % 2 == 1:
            i = j + 1
            continue
        return line[:j], line[j:]


def strip_comments(text: str) -> str:
    """Drop comment text, keeping newlines so line numbers survive."""
    return "".join(
        split_comment(line.rstrip("\n"))[0] + ("\n" if line.endswith("\n") else "")
        for line in text.splitlines(keepends=True)
    )


def resolve_input(name: str) -> Path | None:
    """The paper/ file an \\input names, or None for a file the TeX install supplies."""
    name = name.strip()
    path = PAPER / (name if name.endswith(".tex") else name + ".tex")
    return path if path.is_file() else None


def sources(path: Path = MAIN) -> list[Path]:
    """main.tex and every paper/ file it \\inputs, depth first, each once."""
    out: list[Path] = []

    def walk(p: Path) -> None:
        if p in out:
            return
        out.append(p)
        for match in INPUT.finditer(strip_comments(p.read_text(encoding="utf-8"))):
            child = resolve_input(match.group(2))
            if child:
                walk(child)

    walk(path)
    return out


def flatten(path: Path = MAIN, _depth: int = 0) -> str:
    """The source with every \\input or \\include of a paper/ file spliced in.

    Comments are kept, and an \\input inside a comment is left alone.
    """
    if _depth > 20:
        raise SystemExit(f"\\input nesting deeper than 20 at {path}")
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        code, comment = split_comment(line.rstrip("\r\n"))
        newline = "\n" if line.endswith("\n") else ""

        def splice(match: re.Match) -> str:
            child = resolve_input(match.group(2))
            if child is None:
                return match.group(0)
            body = flatten(child, _depth + 1).rstrip("\n")
            if match.group(1) == "include":
                return "\\clearpage\n" + body + "\n\\clearpage "
            return body + "\n"

        lines.append(INPUT.sub(splice, code) + comment + newline)
    return "".join(lines)


def balanced(text: str, start: int) -> tuple[str, int]:
    """Body of the brace group whose opening brace is just before start, and the index after it."""
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start : i - 1], i


def squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def title(tex: str) -> str:
    match = TITLE.search(tex)
    if not match:
        raise SystemExit("no \\title{...} in paper/main.tex")
    return squeeze(balanced(tex, match.end())[0])


def abstract(tex: str) -> str:
    """The abstract as written, whitespace squeezed.

    Handles the abstract environment and A&A's structured
    \\abstract{context}{aims}{methods}{results}{conclusions}, whose parts are
    joined with spaces.
    """
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if match:
        return squeeze(match.group(1))
    match = re.search(r"\\abstract\s*\{", tex)
    if not match:
        raise SystemExit("no abstract in paper/main.tex")
    parts, i = [], match.end()
    while True:
        body, i = balanced(tex, i)
        parts.append(body)
        nxt = re.match(r"\s*\{", tex[i:])
        if not nxt:
            break
        i += nxt.end()
    return squeeze(" ".join(p for p in parts if p.strip()))


def preamble_macros(tex: str) -> set[str]:
    """Names of commands that the preamble defines (macros.tex included once flattened)."""
    preamble = tex.split(r"\begin{document}", 1)[0]
    return set(MACRO_DEF.findall(strip_comments(preamble)))


def body(tex: str) -> str:
    """Text between \\begin{document} and \\end{document}."""
    after = tex.split(r"\begin{document}", 1)[-1]
    return after.split(r"\end{document}", 1)[0]

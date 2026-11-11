#!/usr/bin/env python3
"""Start a paper from the template, or move an existing one to another journal class.

Reads paper.toml (title, authors, keywords, journal) and writes paper/main.tex from
templates/journals/<journal>/main.tex, plus the wrappers submit.tex and preprint.tex
and the latexmkrc. paper/macros.tex and paper/references.bib are copied from
templates/shared/ only if they do not exist yet, so they are never overwritten.

    python util/new_paper.py --title "..." --slug my-paper --journal aastex   # new paper
    python util/new_paper.py --switch --journal mnras                         # change journal
    python util/new_paper.py --list                                           # journals available

--switch keeps the current title, abstract and everything between the BODY markers
of paper/main.tex, and rebuilds the front matter for the new class. Without
--switch the script refuses to overwrite an existing paper/main.tex unless --force
is given. Either way it sets `journal` in paper.toml and the selected journal in
.claude/skills/journal/SKILL.md, so the two stay in step.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys

from paperconf import (
    CONFIG,
    JOURNALS,
    MAIN,
    PAPER,
    ROOT,
    SKILL,
    abstract,
    load,
    strip_comments,
    title,
    utf8_stdout,
)

SHARED = ROOT / "templates" / "shared"
BODY_START = "%% ---- BODY ----"
BODY_END = "%% ---- END BODY ----"
BUILD_EXTS = [".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".toc", ".spl", ".synctex.gz"]

WRAPPER = """\
% {what} Builds {job}.pdf. See the header of main.tex.
\\def\\papermode{{{job}}}
\\input{{main}}
"""
WRAPPERS = {
    "submit": "Submission typesetting.",
    "preprint": "Typesetting close to the published article; arXiv receives this one.",
}
LATEXMKRC = """\
# A bare `latexmk -pdf` in this directory builds both typesettings of main.tex:
#   submit.pdf and preprint.pdf (see the header of main.tex for what each one is).
@default_files = ('submit.tex', 'preprint.tex');
"""


def surname(name: str) -> str:
    return name.split()[-1]


def initials(name: str) -> str:
    parts = name.split()
    return " ".join(p[0] + "." for p in parts[:-1]) + " " + parts[-1]


def short_authors(authors: list[dict], style=surname) -> str:
    names = [style(a["name"]) for a in authors]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al."


def affiliation_numbers(authors: list[dict]) -> tuple[list[str], list[int]]:
    """Unique affiliations in order of first use, and each author's 1-based index into them."""
    unique: list[str] = []
    index = []
    for a in authors:
        aff = a.get("affiliation", "")
        if aff not in unique:
            unique.append(aff)
        index.append(unique.index(aff) + 1)
    return unique, index


def authors_aastex(authors):
    out = []
    for a in authors:
        orcid = f"[{a['orcid']}]" if a.get("orcid") else ""
        out.append(f"\\author{orcid}{{{a['name']}}}")
        if a.get("affiliation"):
            out.append(f"\\affiliation{{{a['affiliation']}}}")
        if a.get("email"):
            out.append(f"\\email{{{a['email']}}}")
        out.append("")
    return "\n".join(out).rstrip()


def authors_mnras(authors):
    unique, index = affiliation_numbers(authors)
    names = []
    for a, n in zip(authors, index, strict=True):
        thanks = f"\\thanks{{E-mail: {a['email']}}}" if a.get("corresponding") and a.get("email") else ""
        names.append(f"{a['name']},$^{{{n}}}${thanks}")
    affs = "\\\\\n".join(f"$^{{{n}}}${aff}" for n, aff in enumerate(unique, 1))
    short = short_authors(authors, initials)
    return f"\\author[{short}]{{\n" + "\n".join(names) + "\n\\\\\n" + affs + "\n}"


def authors_aa(authors):
    unique, index = affiliation_numbers(authors)
    names = []
    for a, n in zip(authors, index, strict=True):
        thanks = (
            f"\\fnmsep\\thanks{{Corresponding author: \\email{{{a['email']}}}}}"
            if a.get("corresponding") and a.get("email")
            else ""
        )
        names.append(f"{a['name']}\\inst{{{n}}}{thanks}")
    return (
        "\\author{" + "\n  \\and ".join(names) + "}\n"
        "\\institute{" + "\n  \\and ".join(unique) + "}\n"
        f"\\authorrunning{{{short_authors(authors)}}}"
    )


def authors_elsarticle(authors):
    unique, index = affiliation_numbers(authors)
    out = []
    for a, n in zip(authors, index, strict=True):
        cor = "\\corref{cor1}" if a.get("corresponding") else ""
        out.append(f"\\author[aff{n}]{{{a['name']}{cor}}}")
        if a.get("email"):
            out.append(f"\\ead{{{a['email']}}}")
    if any(a.get("corresponding") for a in authors):
        out.append("\\cortext[cor1]{Corresponding author}")
    out += [f"\\affiliation[aff{n}]{{organization={{{aff}}}}}" for n, aff in enumerate(unique, 1)]
    return "\n".join(out)


def authors_revtex(authors):
    out = []
    for a in authors:
        out.append(f"\\author{{{a['name']}}}")
        if a.get("email"):
            out.append(f"\\email{{{a['email']}}}")
        if a.get("affiliation"):
            out.append(f"\\affiliation{{{a['affiliation']}}}")
    return "\n".join(out)


def authors_amsart(authors):
    out = []
    for a in authors:
        out.append(f"\\author{{{a['name']}}}")
        if a.get("affiliation"):
            out.append(f"\\address{{{a['affiliation']}}}")
        if a.get("email"):
            out.append(f"\\email{{{a['email']}}}")
    return "\n".join(out)


AUTHORS = {
    "aastex": authors_aastex,
    "mnras": authors_mnras,
    "aa": authors_aa,
    "elsarticle": authors_elsarticle,
    "revtex": authors_revtex,
    "amsart": authors_amsart,
}
KEYWORD_SEP = {
    "aastex": " --- ",
    "mnras": " -- ",
    "aa": " -- ",
    "elsarticle": " \\sep ",
    "revtex": ", ",
    "amsart": ", ",
}


def journals() -> list[str]:
    return sorted(p.name for p in JOURNALS.iterdir() if (p / "main.tex").is_file())


def set_toml(key: str, value: str) -> None:
    """Rewrite one `key = "..."` line of the [paper] table in paper.toml, keeping comments."""
    text = CONFIG.read_text(encoding="utf-8")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    new, n = re.subn(
        rf'^({re.escape(key)}\s*=\s*)"[^"\n]*"',
        lambda m: m.group(1) + f'"{escaped}"',
        text,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise SystemExit(f'paper.toml has no line `{key} = "..."` to update')
    CONFIG.write_text(new, encoding="utf-8")


def set_skill(journal: str) -> None:
    text = SKILL.read_text(encoding="utf-8")
    new, n = re.subn(
        r"(<!-- selected-journal -->).*?(<!-- /selected-journal -->)",
        lambda m: f"{m.group(1)}`{journal}`{m.group(2)}",
        text,
        flags=re.S,
    )
    if n != 1:
        raise SystemExit(f"{SKILL.relative_to(ROOT)} has no selected-journal marker")
    SKILL.write_text(new, encoding="utf-8")


def current_parts() -> tuple[str, str, str]:
    """Title, abstract and body of the existing paper/main.tex."""
    raw = MAIN.read_text(encoding="utf-8")
    if BODY_START not in raw or BODY_END not in raw:
        raise SystemExit(f"paper/main.tex lacks the {BODY_START!r} and {BODY_END!r} markers")
    body = raw.split(BODY_START, 1)[1].split(BODY_END, 1)[0].strip("\n")
    stripped = strip_comments(raw)
    return title(stripped), abstract(stripped), body


def render(journal: str, conf: dict, title_text: str, abstract_text: str, body: str) -> str:
    paper = conf["paper"]
    authors = conf.get("authors") or [{"name": "Anonymous"}]
    template = (JOURNALS / journal / "main.tex").read_text(encoding="utf-8")
    bibs = ",".join(re.sub(r"\.bib$", "", b) for b in paper.get("bibs", ["references.bib"]))
    fields = {
        "TITLE_COMMENT": f"{paper.get('slug', 'paper')}: {title_text}",
        "TITLE": title_text,
        "SHORT_TITLE": paper.get("short_title") or title_text,
        "SHORT_AUTHORS": short_authors(authors),
        "AUTHORS": AUTHORS[journal](authors),
        "ABSTRACT": abstract_text.strip(),
        "KEYWORDS": KEYWORD_SEP[journal].join(paper.get("keywords", [])),
        "BODY": body.strip("\n"),
        "BIBS": bibs,
    }
    out = template
    for key, value in fields.items():
        out = out.replace(f"<<{key}>>", value)
    left = re.findall(r"<<[A-Z_]+>>", out)
    if left:
        raise SystemExit(f"template {journal} has unfilled fields: {', '.join(sorted(set(left)))}")
    return out


def main() -> int:
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--journal", help="journal class (see --list); default: paper.toml")
    parser.add_argument("--title", help="paper title; stored in paper.toml")
    parser.add_argument("--slug", help="short machine name; stored in paper.toml")
    parser.add_argument("--switch", action="store_true", help="keep title, abstract and body of the current paper")
    parser.add_argument("--force", action="store_true", help="overwrite paper/main.tex with a fresh skeleton")
    parser.add_argument("--list", action="store_true", help="list the journal classes and exit")
    args = parser.parse_args()

    if args.list:
        print("\n".join(journals()))
        return 0
    if args.title:
        set_toml("title", args.title)
    if args.slug:
        set_toml("slug", args.slug)
    if args.journal:
        if args.journal not in journals():
            raise SystemExit(f"no journal {args.journal!r}; choose from: {', '.join(journals())}")
        set_toml("journal", args.journal)

    conf = load()
    journal = conf["paper"]["journal"]
    if journal not in AUTHORS:
        raise SystemExit(f"util/new_paper.py has no author block for {journal!r}; add one to AUTHORS")

    if args.switch:
        title_text, abstract_text, body = current_parts()
        if args.title:
            title_text = args.title
        shutil.copy2(MAIN, PAPER / "main.tex.bak")
        print("saved the previous source as paper/main.tex.bak")
    else:
        if MAIN.exists() and not args.force:
            raise SystemExit(
                "paper/main.tex exists. Use --switch to change journal and keep the text, "
                "or --force to start over."
            )
        title_text = conf["paper"]["title"]
        abstract_text = (SHARED / "abstract.tex").read_text(encoding="utf-8")
        body = (SHARED / "body.tex").read_text(encoding="utf-8")

    PAPER.mkdir(exist_ok=True)
    # An .aux or .bbl written under the old class breaks the first build under the new one.
    for job in ("main", *WRAPPERS):
        for ext in BUILD_EXTS:
            (PAPER / f"{job}{ext}").unlink(missing_ok=True)
    MAIN.write_text(render(journal, conf, title_text, abstract_text, body), encoding="utf-8")
    for job, what in WRAPPERS.items():
        (PAPER / f"{job}.tex").write_text(WRAPPER.format(what=what, job=job), encoding="utf-8")
    (PAPER / "latexmkrc").write_text(LATEXMKRC, encoding="utf-8")
    for name in ("macros.tex", "references.bib"):
        if not (PAPER / name).exists():
            shutil.copy2(SHARED / name, PAPER / name)
            print(f"copied templates/shared/{name} to paper/")
    set_skill(journal)
    print(f"wrote paper/main.tex for {journal}")
    if journal == "aa" and args.switch:
        print("A&A wants a structured abstract: split \\abstract{...}{}{}{}{} into its five parts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

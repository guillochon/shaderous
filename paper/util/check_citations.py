#!/usr/bin/env python3
"""Check the hand-written bibliography against the citation keys and against real indexes.

The bibliography files stay as written: this script never rewrites them.

    python util/check_citations.py --keys          # keys resolve, no duplicates; no network
    python util/check_citations.py                 # also check every entry online
    python util/check_citations.py --source open   # Crossref and arXiv even with an ADS token

Online, each entry is checked against ADS when ADS_TOKEN is set (in the
environment or in .env at the repo root): by DOI, then arXiv id, then first
author, year and title. That run writes paper/bibcodes.json and a raw ADS export
to paper/ads_export_raw.bib for comparison (gitignored; not the bibliography).
Without a token, or with --source open, entries with a DOI are checked against
Crossref and entries with only an arXiv id against the arXiv API. Entries with
neither are reported as unverified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from paperconf import PAPER, ROOT, bibs, flatten, utf8_stdout

BIBCODES_PATH = PAPER / "bibcodes.json"
EXPORT_PATH = PAPER / "ads_export_raw.bib"
ADS_API = "https://api.adsabs.harvard.edu/v1"
USER_AGENT = "paper-template-check-citations/1.0"

CITE = re.compile(r"\\(?:no)?cite[a-zA-Z]*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}")
COMMENT = re.compile(r"(?<!\\)%.*")
ENTRY_START = re.compile(r"@(\w+)\s*\{")
SKIP_KINDS = {"comment", "string", "preamble"}


# ---------------------------------------------------------------- reading the bib


def load_env(path: Path) -> None:
    """Set variables from a .env file without overriding the environment."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        while i < n and body[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        name_match = re.match(r"(\w+)\s*=\s*", body[i:])
        if not name_match:
            i += 1
            continue
        name = name_match.group(1).lower()
        i += name_match.end()
        if i >= n:
            break
        if body[i] == "{":
            depth, start = 0, i + 1
            while i < n:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            fields[name] = body[start : i - 1].strip()
        elif body[i] == '"':
            i += 1
            start = i
            while i < n and body[i] != '"':
                i += 1
            fields[name] = body[start:i].strip()
            i += 1
        else:
            start = i
            while i < n and body[i] not in ",\n":
                i += 1
            fields[name] = body[start:i].strip()
    return fields


def parse_bib(path: Path) -> dict[str, tuple[str, dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, tuple[str, dict[str, str]]] = {}
    i = 0
    while True:
        at = text.find("@", i)
        if at < 0:
            break
        match = ENTRY_START.match(text, at)
        if not match:
            i = at + 1
            continue
        kind = match.group(1).lower()
        body_start = match.end()
        if kind in SKIP_KINDS:
            i = body_start
            continue
        comma = text.find(",", body_start)
        if comma < 0:
            break
        key = text[body_start:comma].strip()
        depth, j = 1, comma + 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        entries[key] = (kind, parse_fields(text[comma + 1 : j - 1]))
        i = j
    return entries


def load_bibs() -> tuple[dict[str, tuple[str, dict[str, str], str]], list[str]]:
    entries: dict[str, tuple[str, dict[str, str], str]] = {}
    duplicates: list[str] = []
    for path in bibs():
        if not path.is_file():
            raise SystemExit(f"missing {path.relative_to(ROOT)}")
        for key, (kind, fields) in parse_bib(path).items():
            if key in entries:
                duplicates.append(key)
            entries[key] = (kind, fields, path.name)
    return entries, duplicates


def cite_keys() -> list[str]:
    text = COMMENT.sub("", flatten())
    keys: list[str] = []
    for group in CITE.findall(text):
        for key in group.split(","):
            key = key.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def norm(value: str) -> str:
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = re.sub(r"[{}$~\\]", "", value).lower()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", value).split())


def doi_of(fields: dict[str, str]) -> str:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", fields.get("doi", ""), flags=re.I)


def eprint_of(fields: dict[str, str]) -> str:
    return re.sub(r"(?i)^arxiv:", "", fields.get("eprint", "")).strip()


def first_author(fields: dict[str, str]) -> str:
    author = re.sub(r"[{}]", "", fields.get("author", "")).split(" and ")[0]
    return author.split(",")[0].strip()


def compare(fields: dict[str, str], found: dict) -> list[str]:
    """Differences between a bib entry and a record normalized to
    {title, year, first_author, volume, page, doi}."""
    issues = []
    if fields.get("year") and found.get("year") and str(found["year"]) != fields["year"]:
        issues.append(f"year {fields['year']}->{found['year']}")
    if fields.get("volume") and found.get("volume") and str(found["volume"]) != fields["volume"]:
        issues.append(f"vol {fields['volume']}->{found['volume']}")
    page = found.get("page") or ""
    if fields.get("pages") and page and norm(fields["pages"].split("-")[0]) != norm(page.split("-")[0]):
        issues.append(f"page {fields['pages']}->{page}")
    if norm(fields.get("title", ""))[:40] != norm(found.get("title", ""))[:40]:
        issues.append(f'TITLE: "{found.get("title", "")}"')
    first = norm(first_author(fields))
    if first and first not in norm(found.get("first_author", "")):
        issues.append(f"AUTHOR {first} -> {found.get('first_author', '')}")
    if found.get("doi") and doi_of(fields) and found["doi"].lower() != doi_of(fields).lower():
        issues.append(f"doi -> {found['doi']}")
    return issues


# ---------------------------------------------------------------- ADS


def ads_request(method: str, path: str, payload: dict | None = None, params: dict | None = None):
    url = ADS_API + path
    headers = {"Authorization": "Bearer " + os.environ["ADS_TOKEN"]}
    data = None
    if params:
        url += "?" + urllib.parse.urlencode(params)
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"ADS returned {exc.code} for {path}: {detail}") from exc


def ads_search(query: str) -> list[dict]:
    fl = "bibcode,title,author,year,pub,volume,page,doi,identifier,doctype"
    return ads_request("GET", "/search/query", params={"q": query, "fl": fl, "rows": 5})["response"]["docs"]


def ads_find(fields: dict[str, str]) -> tuple[list[dict], str]:
    doi = doi_of(fields)
    if doi:
        docs = ads_search(f'doi:"{doi}"')
        if docs:
            return docs, "doi"
    eprint = eprint_of(fields)
    if eprint:
        docs = ads_search(f"arXiv:{eprint}")
        if docs:
            return docs, "arxiv"
    author, year = first_author(fields), fields.get("year", "")
    title = norm(fields.get("title", ""))
    words = " ".join(title.split()[:8])
    if author and year and words:
        docs = ads_search(f'author:"^{author}" year:{year} title:"{words}"')
        if docs:
            return docs, "title"
    if author and year.isdigit():
        loose = " ".join(title.split()[:4])
        docs = ads_search(f'author:"^{author}" year:{int(year) - 1}-{int(year) + 1} abs:"{loose}"')
        return docs, "loose"
    return [], "unsearchable"


def ads_record(doc: dict) -> dict:
    return {
        "title": doc["title"][0] if doc.get("title") else "",
        "year": doc.get("year"),
        "first_author": doc["author"][0] if doc.get("author") else "",
        "volume": doc.get("volume"),
        "page": (doc.get("page") or [""])[0],
        "doi": (doc.get("doi") or [""])[0],
    }


def validate_ads(entries: dict) -> list[str]:
    bibcodes: dict[str, str] = {}
    problems = []
    for key, (_kind, fields, _source) in entries.items():
        docs, how = ads_find(fields)
        time.sleep(0.1)
        if not docs:
            print(f"{key:<28} {'NOT FOUND':<22} {how:<12} {fields.get('title', '')}")
            problems.append(key)
            continue
        docs.sort(key=lambda doc: (doc.get("doctype") == "eprint",))
        doc = docs[0]
        issues = compare(fields, ads_record(doc))
        print(f"{key:<28} {doc['bibcode']:<22} {how:<12} {'; '.join(issues) if issues else 'OK'}")
        bibcodes[key] = doc["bibcode"]
        if issues:
            problems.append(key)
    BIBCODES_PATH.write_text(json.dumps(dict(sorted(bibcodes.items())), indent=2) + "\n", encoding="utf-8")
    if bibcodes:
        exported = ads_request(
            "POST",
            "/export/bibtex",
            payload={"bibcode": list(bibcodes.values()), "keyformat": "%1H:%Y", "maxauthor": 0, "journalformat": 1},
        )
        header = (
            "% Raw ADS BibTeX for comparison only. Do not replace the bibliography with\n"
            "% this file: the hand-written entries are the source of truth.\n\n"
        )
        EXPORT_PATH.write_text(header + exported["export"], encoding="utf-8")
    print(f"checked {len(entries)} against ADS; problems {len(problems)}")
    return problems


# ---------------------------------------------------------------- Crossref and arXiv


class Unreachable(Exception):
    pass


def get(url: str) -> bytes | None:
    """The response body, None for a 404, Unreachable after two failed attempts."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = str(exc) or type(exc).__name__
        time.sleep(3 * (attempt + 1))
    raise Unreachable(f"{url.split('?')[0]}: {error}")


def crossref_record(doi: str) -> dict | None:
    raw = get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"))
    if raw is None:
        return None
    msg = json.loads(raw)["message"]
    issued = (msg.get("published-print") or msg.get("published-online") or msg.get("issued") or {})
    parts = issued.get("date-parts") or [[None]]
    authors = msg.get("author") or [{}]
    first = authors[0].get("family") or authors[0].get("name") or ""
    return {
        "title": " ".join(re.sub(r"<[^>]+>", "", (msg.get("title") or [""])[0]).split()),
        "year": parts[0][0],
        "first_author": first,
        "volume": msg.get("volume"),
        "page": msg.get("page") or msg.get("article-number") or "",
        "doi": msg.get("DOI", ""),
    }


def arxiv_record(eprint: str) -> dict | None:
    raw = get("https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(eprint))
    if raw is None:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(raw).find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        return None
    name = entry.findtext("a:author/a:name", default="", namespaces=ns)
    return {
        "title": " ".join(entry.findtext("a:title", default="", namespaces=ns).split()),
        "year": entry.findtext("a:published", default="", namespaces=ns)[:4],
        "first_author": name.split()[-1] if name else "",
    }


def validate_open(entries: dict) -> list[str]:
    problems, unverified = [], []
    for key, (_kind, fields, _source) in entries.items():
        doi, eprint = doi_of(fields), eprint_of(fields)
        try:
            if doi:
                record, how = crossref_record(doi), "crossref"
            elif eprint:
                record, how = arxiv_record(eprint), "arxiv"
                # arXiv gives the posting year, which can precede the journal year of a published paper.
                if record and not fields.get("journal", "").lower().startswith("arxiv"):
                    record.pop("year", None)
                time.sleep(3)  # arXiv asks for one request every three seconds
            else:
                print(f"{key:<28} {'UNVERIFIED':<12} no DOI or arXiv id; check by hand")
                unverified.append(key)
                continue
        except Unreachable as exc:
            print(f"{key:<28} {'UNREACHABLE':<12} {exc}; run again later")
            problems.append(key)
            continue
        if record is None:
            print(f"{key:<28} {'NOT FOUND':<12} {how}: {doi or eprint}")
            problems.append(key)
            continue
        issues = compare(fields, record)
        print(f"{key:<28} {how:<12} {'; '.join(issues) if issues else 'OK'}")
        if issues:
            problems.append(key)
    print(f"checked {len(entries) - len(unverified)} against Crossref/arXiv; "
          f"problems {len(problems)}; unverified {len(unverified)}")
    return problems


# ---------------------------------------------------------------- main


def check_keys(entries: dict, duplicates: list[str]) -> list[str]:
    problems = []
    keys = cite_keys()
    if duplicates:
        problems.append("duplicate keys: " + ", ".join(duplicates))
    missing = [key for key in keys if key not in entries]
    if missing:
        problems.append("cited but absent from the bibs: " + ", ".join(missing))
    print(f"bib entries: {len(entries)}")
    print(f"citation keys: {len(keys)}")
    uncited = [key for key in entries if key not in keys]
    if uncited:
        print("in the bibs but not cited (fine, but consider deleting): " + ", ".join(uncited))
    for key, (_kind, fields, source) in entries.items():
        lacking = [f for f in ("author", "title", "year") if not fields.get(f)]
        if lacking:
            problems.append(f"{key} ({source}) lacks {', '.join(lacking)}")
    for problem in problems:
        print(problem)
    return problems


def main() -> int:
    utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keys", action="store_true", help="check citation keys only; no network")
    parser.add_argument("--source", choices=["auto", "ads", "open"], default="auto",
                        help="auto: ADS if ADS_TOKEN is set, else Crossref and arXiv")
    args = parser.parse_args()
    load_env(ROOT / ".env")
    entries, duplicates = load_bibs()
    problems = check_keys(entries, duplicates)
    if args.keys:
        return 1 if problems else 0
    use_ads = args.source == "ads" or (args.source == "auto" and os.environ.get("ADS_TOKEN"))
    if args.source == "ads" and not os.environ.get("ADS_TOKEN"):
        raise SystemExit("ADS_TOKEN is not set. Put it in .env (https://ui.adsabs.harvard.edu/user/settings/token).")
    problems.extend(validate_ads(entries) if use_ads else validate_open(entries))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

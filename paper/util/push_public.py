#!/usr/bin/env python3
"""Push a public snapshot of the paper to its own GitHub repository.

The working repository stays private (referee reports, downloaded papers, notes in
progress). This copies the paths listed under [public] include in paper.toml into
a clone of the public repo named by [public] repo, test-compiles the snapshot, and
pushes one commit. It never copies research/review*.md, research/ subfolders
(downloaded papers), .env, or build products, whatever the globs say.

    python util/push_public.py --dry-run               # local copy in .public-snapshot/, no push
    python util/push_public.py --dry-run --out DIR
    python util/push_public.py                         # create the repo if needed, sync and push
    python util/push_public.py --skip-build            # do not test-compile

The public repo is created public on first use, so a first push publishes the
snapshot. Run --dry-run and read what it wrote before the first real push.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from build import compile_job, report
from paperconf import ROOT, load, utf8_stdout

NEVER = ("research/review", ".env")
BUILD_EXTS = [".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pdf", ".spl", ".synctex.gz", ".toc"]
GITIGNORE = """\
.env
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc
*.spl
paper/main.pdf
paper/submit.pdf
paper/preprint.pdf
*-arxiv.zip
paper/figures/*.png
paper/ads_export_raw.bib
__pycache__/
.venv/
"""


def run(cmd, cwd=None, check=True, capture=False):
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=capture)


def repo_info(repo: str):
    """gh metadata for the repo, or None. Follows GitHub rename redirects."""
    result = run(["gh", "repo", "view", repo, "--json", "nameWithOwner,url,isPrivate"], check=False, capture=True)
    return json.loads(result.stdout) if result.returncode == 0 else None


def ensure_public_repo(repo: str, description: str) -> None:
    info = repo_info(repo)
    if info and info.get("nameWithOwner") == repo:
        if info.get("isPrivate"):
            raise SystemExit(f"{repo} is private; refusing to push a public snapshot into it")
        return
    run(["gh", "repo", "create", repo, "--public", "--description", description or "Paper source"])
    info = repo_info(repo)
    if not info or info.get("nameWithOwner") != repo or info.get("isPrivate"):
        raise SystemExit(f"could not create the public repo {repo}")


def files_to_copy(include: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in include:
        for path in sorted(ROOT.glob(pattern)):
            rel = path.relative_to(ROOT).as_posix()
            if not path.is_file() or rel in out or rel.startswith(NEVER):
                continue
            if rel.startswith("research/") and rel.count("/") > 1:
                continue  # downloaded third-party sources
            out.append(rel)
    if not out:
        raise SystemExit("[public] include matched no files")
    return out


def populate(dest: Path, include: list[str], title: str) -> None:
    for rel in files_to_copy(include):
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out)
    (dest / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    if not (dest / "README.md").exists():
        (dest / "README.md").write_text(
            f"# {title}\n\nManuscript source, bibliography and figure scripts.\n\n"
            "Build with `cd paper && latexmk -pdf`.\n",
            encoding="utf-8",
        )


def test_build(dest: Path) -> None:
    """Compile the snapshot so a missing figure or bib entry is caught before it is published."""
    paper = dest / "paper"
    for job in ("submit", "preprint"):
        compile_job(job, src=paper)
        problems, _ = report(paper / f"{job}.log")
        if problems:
            raise SystemExit(f"snapshot {job} does not compile cleanly: {problems}; not pushing")
    for path in paper.iterdir():
        if any(path.name.endswith(ext) for ext in BUILD_EXTS) and path.stem in ("submit", "preprint", "main"):
            path.unlink()
    print("snapshot compiles cleanly")


def wipe_worktree(dest: Path) -> None:
    for path in dest.iterdir():
        if path.name == ".git":
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def main() -> None:
    utf8_stdout()
    conf = load()
    public = conf.get("public", {})
    repo = public.get("repo", "")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="write a local copy; do not create or push a repo")
    parser.add_argument("--out", type=Path, help="directory for --dry-run (default: .public-snapshot/)")
    parser.add_argument("--skip-build", action="store_true", help="do not test-compile the snapshot")
    args = parser.parse_args()
    include = public.get("include", [])
    title = conf["paper"]["title"]

    if args.dry_run:
        dest = args.out or (ROOT / ".public-snapshot")
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        populate(dest, include, title)
        if not args.skip_build:
            test_build(dest)
        print(f"wrote dry-run copy to {dest}")
        return

    if not repo:
        raise SystemExit("set [public] repo = \"owner/name\" in paper.toml first")
    ensure_public_repo(repo, public.get("description", ""))
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "snapshot"
        url = f"https://github.com/{repo}.git"
        if run(["git", "clone", url, str(dest)], check=False, capture=True).returncode != 0:
            dest.mkdir(parents=True)
            run(["git", "init", "-b", "main"], cwd=dest)
            run(["git", "remote", "add", "origin", url], cwd=dest)
        wipe_worktree(dest)
        populate(dest, include, title)
        if not args.skip_build:
            test_build(dest)
        run(["git", "add", "-A"], cwd=dest)
        if not run(["git", "status", "--porcelain"], cwd=dest, capture=True).stdout.strip():
            print("public repo already up to date")
            return
        sha = run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture=True).stdout.strip()
        run(["git", "commit", "-m", f"Update paper snapshot ({sha})"], cwd=dest)
        run(["git", "push", "-u", "origin", "HEAD"], cwd=dest)
        print(f"pushed {url}")


if __name__ == "__main__":
    main()

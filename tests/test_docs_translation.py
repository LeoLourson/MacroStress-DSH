"""The English documentation stays a twin of the Russian one: every document has a translation,
and a translation links to files that exist, in English wherever a translation exists."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(([^)\s]+)\)")
TREES = ("docs", "dsh", "src", "tests", "corpus")
SKIPPED = ("docs/local/", "src/mst/templates/")
"""A private folder and the templates a new plugin is generated from."""


def documents() -> list[Path]:
    """Every folder's README and all of ``docs/``; the skills and corpus templates are data."""
    found = [ROOT / "CHANGELOG.md"]
    for tree in TREES:
        for path in (ROOT / tree).rglob("*.md"):
            relative = path.relative_to(ROOT).as_posix()
            if path.name.endswith(".en.md") or relative.startswith(SKIPPED):
                continue
            if tree == "docs" or path.name == "README.md":
                found.append(path)
    return found


def translations() -> list[Path]:
    found = [ROOT / "CHANGELOG.en.md", ROOT / "README.md"]
    for tree in TREES:
        found += [
            path
            for path in (ROOT / tree).rglob("*.en.md")
            if not path.relative_to(ROOT).as_posix().startswith(SKIPPED)
        ]
    return found


def link_problems(path: Path) -> list[str]:
    """Links of a translation that lead nowhere or to Russian text that has a translation.

    The file's own original is allowed: that link is the language switch.
    """
    problems = []
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = (path.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            problems.append(f"{path.name}: {target} does not exist")
        elif (
            resolved.suffix == ".md"
            and not resolved.name.endswith(".en.md")
            and resolved.with_name(resolved.stem + ".en.md").is_file()
            and path.name != resolved.stem + ".en.md"
            and path != resolved
        ):
            problems.append(f"{path.name}: {target} is the Russian text")
    return problems


def test_every_document_has_an_english_twin():
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in documents()
        if not path.with_name(path.stem + ".en.md").is_file()
    ]
    assert not missing, missing


def test_every_english_document_has_a_russian_original():
    orphans = [
        path.relative_to(ROOT).as_posix()
        for path in translations()
        if path.name.endswith(".en.md")
        and not path.with_name(path.name[: -len(".en.md")] + ".md").is_file()
    ]
    assert not orphans, orphans


def test_english_links_resolve_and_stay_english():
    problems = [problem for path in translations() for problem in link_problems(path)]
    assert not problems, problems


def test_the_guard_catches_a_broken_and_a_russian_link(tmp_path):
    (tmp_path / "other.md").write_text("x", encoding="utf-8")
    (tmp_path / "other.en.md").write_text("x", encoding="utf-8")
    (tmp_path / "own.md").write_text("x", encoding="utf-8")
    page = tmp_path / "own.en.md"
    page.write_text(
        "[a](nowhere.en.md) [b](other.md) [c](other.en.md) [d](own.md) [e](https://a.b)",
        encoding="utf-8",
    )
    found = link_problems(page)
    assert len(found) == 2
    assert "nowhere" in found[0] and "other.md" in found[1]

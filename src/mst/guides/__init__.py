"""The working guides of the agent: every client reads them from the server.

A guide is a Markdown file with a frontmatter of ``name``, ``description`` and ``whenToUse``.
The server gives them to any client through its instructions, the ``guide`` tool and a prompt.
``dsh/skills`` holds the same files, because the DSH plugin is installed without this package
and registers them as skills; a test keeps the two copies equal.
"""

from pathlib import Path

ROOT = Path(__file__).parent


def parse(text: str) -> dict:
    """The frontmatter fields of one guide with its body under ``content``."""
    _, front, body = text.split("---\n", 2)
    fields = dict(line.split(": ", 1) for line in front.splitlines() if line.strip())
    return {**fields, "content": body.strip()}


def catalogue() -> list[dict]:
    """Every guide without its text, in alphabetical order."""
    return [
        {key: value for key, value in parse(_read(path)).items() if key != "content"}
        for path in sorted(ROOT.glob("*.md"))
    ]


def get(name: str) -> dict:
    """One guide with its text; an unknown name is refused with the names that exist."""
    names = [path.stem for path in sorted(ROOT.glob("*.md"))]
    if name not in names:
        raise ValueError(f"no guide {name!r}, try {', '.join(names)}")
    return parse(_read(ROOT / f"{name}.md"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")

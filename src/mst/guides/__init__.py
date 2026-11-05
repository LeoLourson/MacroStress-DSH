"""The working guides of the agent: every client reads them from the server.

A guide is a Markdown file with a frontmatter of ``name``, ``description`` and ``whenToUse``.
The server gives them to any client through its instructions, the ``guide`` tool and a prompt.
The guides of this package belong to the platform; a plugin adds its own through
``Plugin.guides``, so the instruction of a feature lives with the feature. ``dsh/skills``
holds copies of both sets, because the DSH plugin is installed without this package and
registers them as skills; a test keeps every copy equal to its source.
"""

from pathlib import Path

ROOT = Path(__file__).parent


def parse(text: str) -> dict:
    """The frontmatter fields of one guide with its body under ``content``."""
    _, front, body = text.split("---\n", 2)
    fields = dict(line.split(": ", 1) for line in front.splitlines() if line.strip())
    return {**fields, "content": body.strip()}


def _files() -> dict[str, Path]:
    """Every guide by name: the platform's, then the guides the installed plugins declare."""
    found = {path.stem: path for path in sorted(ROOT.glob("*.md"))}
    from mst import plugins

    for plugin in plugins.registry().plugins.values():
        for path in plugin.guides:
            found[Path(path).stem] = Path(path)
    return found


def catalogue() -> list[dict]:
    """Every guide without its text, in alphabetical order."""
    files = _files()
    return [
        {key: value for key, value in parse(_read(files[name])).items() if key != "content"}
        for name in sorted(files)
    ]


def get(name: str) -> dict:
    """One guide with its text; an unknown name is refused with the names that exist."""
    files = _files()
    if name not in files:
        raise ValueError(f"no guide {name!r}, try {', '.join(sorted(files))}")
    return parse(_read(files[name]))


def source(name: str) -> Path:
    """The file a guide is read from, for the checks that keep the DSH copies equal."""
    files = _files()
    if name not in files:
        raise ValueError(f"no guide {name!r}, try {', '.join(sorted(files))}")
    return files[name]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")

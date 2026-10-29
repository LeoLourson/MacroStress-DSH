"""One workspace: its directories, canonical hashes, atomic JSON and its state."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_of(obj: Any) -> str:
    """The hash of an object: sha256 of its canonical JSON.

    Canonical means sorted keys, no ASCII escapes, no spaces: two dicts with the same
    content hash the same, and a changed value changes the hash.
    """
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as JSON atomically: a temporary file, then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, draft = tempfile.mkstemp(dir=path.parent, prefix=".draft-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(obj, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(draft, path)
    except BaseException:
        Path(draft).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any | None:
    """The JSON at ``path``, or None when there is no file there."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class Workspace:
    """The directories of one workspace. Nothing is created until something is written."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @property
    def state_file(self) -> Path:
        return self.root / "workspace.json"

    @property
    def series(self) -> Path:
        return self.root / "series"

    @property
    def corpus(self) -> Path:
        return self.root / "corpus"

    @property
    def macro_models(self) -> Path:
        return self.root / "models" / "macro"

    @property
    def satellite_models(self) -> Path:
        return self.root / "models" / "satellites"

    @property
    def portfolios(self) -> Path:
        return self.root / "portfolios"

    @property
    def scenarios(self) -> Path:
        return self.root / "scenarios"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"


def state(ws: Workspace) -> dict[str, Any]:
    """What the workspace holds: the model versions, the corpus index, the counts."""
    return {
        "workspace": str(ws.root),
        "macro": _latest(ws.macro_models),
        "satellites": _latest(ws.satellite_models),
        "corpus_index": read_json(ws.corpus / "index" / "state.json"),
        "portfolios": _count(ws.portfolios),
        "scenarios": _count(ws.scenarios, suffix=".json"),
        "runs": _count(ws.runs),
        "jobs": _unfinished(ws.jobs),
    }


def write_state(ws: Workspace) -> dict[str, Any]:
    """Write ``workspace.json`` for the prompt section and return what was written."""
    snapshot = state(ws)
    write_json(ws.state_file, snapshot)
    return snapshot


def _latest(directory: Path) -> str | None:
    """The newest entry of ``directory`` by modification time, or None when it is empty."""
    if not directory.is_dir():
        return None
    entries = sorted(directory.iterdir(), key=lambda entry: entry.stat().st_mtime)
    return entries[-1].name if entries else None


def _count(directory: Path, suffix: str = "") -> int:
    """How many entries of ``directory`` end with ``suffix``; zero when it is not there."""
    if not directory.is_dir():
        return 0
    return sum(1 for entry in directory.iterdir() if entry.name.endswith(suffix))


def _unfinished(directory: Path) -> int:
    """How many job records carry no finish time yet."""
    records = directory.glob("*/job.json")
    return sum(1 for path in records if not (read_json(path) or {}).get("finished"))

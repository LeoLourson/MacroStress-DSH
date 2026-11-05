"""The published answers an interview replaced: kept by hash, readable later.

A reopened answer is not lost: the normal form that was published is copied here under its
own hash before the answer stops being current. The hash is the one the estimate recorded, so
a version of the model can still be traced to the answer it read.
"""

from datetime import datetime
from pathlib import Path
from threading import RLock

from mst import store

FOLDER = "_versions"
LOCK = RLock()


def folder(interview, key=None) -> Path:
    root = Path(interview) / FOLDER
    return root / key if key else root


def keep(interview, key, digest: str, answer: dict) -> Path:
    """Keep the normal form under its hash; the same answer is not written twice."""
    with LOCK:
        path = folder(interview, key) / f"{digest}.json"
        if not path.is_file():
            store.write_json(
                path,
                {
                    "hash": digest,
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "answer": answer,
                },
            )
        return path


def history(interview, key) -> list[dict]:
    """The kept versions of one answer, oldest first, without their bodies."""
    found = folder(interview, key)
    if not found.is_dir():
        return []
    rows = []
    for file in sorted(found.glob("*.json")):
        record = store.read_json(file) or {}
        rows.append({"hash": record.get("hash", file.stem), "at": record.get("at")})
    return rows

"""The elicitation answers of the workspace: one folder per date, one file per key.

A model whose estimate reads answers the series store does not hold declares them through
``MacroModel.input_keys``; the agent answers through the ``model`` tool of the server, and the
answers live in ``workspace/elicitation/<date>/<key>.json``. The file keeps the answer in its
normal form under ``answer`` with ``written_at`` beside it; the hash the write returns covers
the normal form alone, so re-writing the same answer changes nothing the estimate reads and
the hashes a model version records stay comparable across dates. No human approves the
answers: the server checks their form (``MacroModel.check_input``) at the moment of writing.

Nothing here knows a model: the folder is the workspace's, the keys are the model's own, and
a model that declares no inputs is never seen by this module.
"""

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

from . import store

ANSWER = "answer"
WRITTEN_AT = "written_at"


def hash_of(normal: dict) -> str:
    """The hash of a normal form: what a write returns and a model version records."""
    return store.sha256_of(normal)


def path(root: Path, as_of: date, key: str) -> Path:
    """``<root>/<as_of>/<key>.json``: where the answer of one key of one date lives."""
    return Path(root) / as_of.isoformat() / f"{key}.json"


def write(root: Path, as_of: date, key: str, normal: dict) -> str:
    """Write the normal form of an answer atomically; returns the hash of the normal form.

    A temporary file and ``os.replace``, so an interrupted write leaves the previous answer
    in place and no half file behind.
    """
    target = path(root, as_of, key)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {WRITTEN_AT: datetime.now().isoformat(timespec="seconds"), ANSWER: normal}
    descriptor, draft = tempfile.mkstemp(dir=target.parent, prefix=".draft-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(record, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        store.replace(Path(draft), target)
    except BaseException:
        Path(draft).unlink(missing_ok=True)
        raise
    return store.sha256_of(normal)


def read(root: Path, as_of: date) -> dict[str, dict]:
    """Every answer written for ``as_of``, by key, in its normal form.

    A temporary file an interrupted write left behind is not an answer.
    """
    folder = Path(root) / as_of.isoformat()
    if not folder.is_dir():
        return {}
    found = {}
    for file in sorted(folder.glob("*.json")):
        if file.name.startswith("."):
            continue
        record = json.loads(file.read_text(encoding="utf-8"))
        found[file.stem] = record[ANSWER]
    return found


def status(root: Path, model, as_of: date) -> dict:
    """Every key of the model for ``as_of``, written or missing, with the hash and the time.

    The missing keys come first, so the agent reads what keeps the estimate from starting
    before anything else.
    """
    folder = Path(root) / as_of.isoformat()
    records = {}
    if folder.is_dir():
        for file in sorted(folder.glob("*.json")):
            if file.name.startswith("."):
                continue
            record = json.loads(file.read_text(encoding="utf-8"))
            records[file.stem] = record
    rows = []
    for key in model.input_keys(as_of):
        record = records.get(key)
        rows.append(
            {
                "key": key,
                "written": record is not None,
                "hash": store.sha256_of(record[ANSWER]) if record else None,
                WRITTEN_AT: record[WRITTEN_AT] if record else None,
            }
        )
    rows.sort(key=lambda row: (row["written"], row["key"]))
    return {"keys": rows}


def missing(root: Path, model, as_of: date) -> list[str]:
    """The keys of the model no answer is written for, in the order they are declared."""
    return [key for key in model.input_keys(as_of) if key not in read(root, as_of)]

"""Model versions on disk: the immutable files of one estimation, named by their hash.

A version is a folder ``<directory>/<sha256>/`` with ``meta.json`` and ``arrays.npz``. The hash
covers the JSON of the meta (sorted keys) and every array's name, type, shape and bytes, so
the same estimation gives the same name and a changed file no longer loads. The macro model
(``mst.macro.version``) and the satellites (``mst.drivers.bundle``) keep their versions so.

The hash is not ``store.sha256_of``: it also covers arrays, and it stays as it is so that the
versions already written keep their names.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

NAME = re.compile(r"^[0-9a-f]{64}$")


def digest(meta: dict, arrays: dict[str, np.ndarray]) -> str:
    """The name of a version with this meta and these arrays."""
    hashed = hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        hashed.update(f"{name}|{array.dtype.str}|{array.shape}".encode())
        hashed.update(array.tobytes())
    return hashed.hexdigest()


def save(directory: Path, meta: dict, arrays: dict[str, np.ndarray]) -> str:
    """Writes the version under ``directory`` unless it exists; returns its name.

    The files go to a ``.draft-`` folder first, which is renamed when complete.
    """
    name = digest(meta, arrays)
    target = directory / name
    if target.exists():
        return name
    directory.mkdir(parents=True, exist_ok=True)
    draft = Path(tempfile.mkdtemp(dir=directory, prefix=".draft-"))
    try:
        (draft / "meta.json").write_text(
            json.dumps(meta, sort_keys=True, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        np.savez(draft / "arrays.npz", **arrays)
        os.replace(draft, target)
    except BaseException:
        shutil.rmtree(draft, ignore_errors=True)
        raise
    return name


def read(directory: Path, name: str, what: str) -> tuple[dict, dict[str, np.ndarray]]:
    """The meta and arrays of a version; refuses a name that is not a hash and a changed version.

    ``what`` names the kind of version in the errors, e.g. ``model version``.
    """
    if not NAME.fullmatch(name):
        raise ValueError(f"not a {what}: {name!r}")
    folder = directory / name
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    with np.load(folder / "arrays.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}
    if digest(meta, arrays) != name:
        raise ValueError(f"{what} {name} does not match its hash")
    return meta, arrays


def folders(directory: Path) -> list[Path]:
    """The finished versions under ``directory``, newest first; drafts are left out."""
    if not directory.is_dir():
        return []
    found = [
        path
        for path in directory.iterdir()
        if not path.name.startswith(".") and (path / "meta.json").is_file()
    ]
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)

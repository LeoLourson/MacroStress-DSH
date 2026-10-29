"""A model version: the immutable files of one estimation, named by their hash.

``<root>/macro/<sha256>/`` holds ``meta.json`` (spec, calculation date, variables, months,
shock labels, diagnostics) and ``arrays.npz`` (the identified posterior and the panel it was
estimated on). The hash covers the canonical JSON and every array's name, type, shape and
bytes, so the same estimation gives the same name and a changed file no longer loads. Versions
stay out of git (plan, section 4, decision 5).
"""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from .estimate import Posterior
from .panel import Panel
from .spec import Dummy, Spec

ARRAYS = ("const", "lags", "dummy", "chol", "logvar", "volvol", "dof", "history", "rotation")


def _digest(meta: dict, arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}|{array.dtype.str}|{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save(model: Posterior, spec: Spec, panel: Panel, as_of: date, root: Path) -> str:
    """Writes the version unless it exists; returns its hash."""
    if model.rotation is None:
        raise ValueError("only an identified posterior is a model version")
    meta = {
        "spec": spec.model_dump(mode="json"),
        "as_of": as_of.isoformat(),
        "ids": model.ids,
        "n_global": model.n_global,
        "months": [m.isoformat() for m in model.months],
        "shocks": model.shocks,
        "diagnostics": model.diagnostics,
    }
    arrays = {name: getattr(model, name) for name in ARRAYS}
    arrays |= {"panel_y": panel.y, "panel_approximate": panel.approximate}
    name = _digest(meta, arrays)
    target = root / "macro" / name
    if target.exists():
        return name
    target.parent.mkdir(parents=True, exist_ok=True)
    draft = Path(tempfile.mkdtemp(dir=target.parent, prefix=".draft-"))
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


def catalogue(root: Path) -> list[dict]:
    """Every saved version, newest first, as its meta records it."""
    directory = Path(root) / "macro"
    if not directory.is_dir():
        return []
    folders = [path for path in directory.iterdir() if (path / "meta.json").is_file()]
    folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [_summary(path) for path in folders]


def describe(root: Path, name: str) -> dict:
    """One version: the meta it stores and the share of its draws the signs identified."""
    _, meta = load(root, name)
    diagnostics = meta["diagnostics"]
    identified, estimated = diagnostics.get("identified"), diagnostics.get("estimated")
    return {
        "version": name,
        "as_of": meta["as_of"],
        "ids": meta["ids"],
        "months": meta["months"],
        "shocks": meta["shocks"],
        "diagnostics": diagnostics,
        "identification_share": identified / estimated if identified and estimated else None,
    }


def _summary(folder: Path) -> dict:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    diagnostics = meta["diagnostics"]
    return {
        "version": folder.name,
        "as_of": meta["as_of"],
        "months": len(meta["months"]),
        "shocks": meta["shocks"],
        "identified": diagnostics.get("identified"),
        "estimated": diagnostics.get("estimated"),
    }


def load(root: Path, name: str) -> tuple[Posterior, dict]:
    """The identified posterior and the meta of a version; refuses a changed version."""
    folder = root / "macro" / name
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    with np.load(folder / "arrays.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}
    if _digest(meta, arrays) != name:
        raise ValueError(f"model version {name} does not match its hash")
    spec = Spec.model_validate(meta["spec"])
    model = Posterior(
        ids=meta["ids"],
        n_global=meta["n_global"],
        months=[date.fromisoformat(m) for m in meta["months"]],
        lags_count=spec.lags,
        dummies=[Dummy.model_validate(d) for d in meta["spec"]["dummies"]],
        diagnostics=meta["diagnostics"],
        shocks=meta["shocks"],
        **{key: arrays[key] for key in ARRAYS},
    )
    return model, meta

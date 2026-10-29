"""A macro model version: ``<root>/macro/<sha256>/``, written and read by ``mst.versions``.

``meta.json`` holds the spec, calculation date, variables, months, shock labels and
diagnostics; ``arrays.npz`` the identified posterior and the panel it was estimated on.
Versions live in the workspace, out of git.
"""

import json
from datetime import date
from pathlib import Path

from mst import versions

from .estimate import Posterior
from .panel import Panel
from .spec import Dummy, Spec

ARRAYS = ("const", "lags", "dummy", "chol", "logvar", "volvol", "dof", "history", "rotation")


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
    return versions.save(root / "macro", meta, arrays)


def catalogue(root: Path) -> list[dict]:
    """Every saved version, newest first, as its meta records it."""
    return [_summary(path) for path in versions.folders(Path(root) / "macro")]


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
    meta, arrays = versions.read(root / "macro", name, "model version")
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

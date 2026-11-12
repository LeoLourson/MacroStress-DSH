"""Convert a directory of legacy scenarios after validating every file."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .legacy import read_legacy
from .translate import QuantityData, TranslationData


def _drivers(raw):
    names = {row["target"] for row in raw.get("impacts", [])}
    for family in ("constraints", "expectations", "probabilities"):
        for row in raw.get(family, []):
            names.update(term["variable"] for term in row["terms"])
    return {name for name in names if ":" in name}


def _history(name, values, history, last):
    values = np.asarray(values, dtype=float)
    history[name] = values
    finite = values[np.isfinite(values)]
    if finite.size:
        last[name] = float(finite[-1])


def context_of(models: Path, raw: dict) -> TranslationData:
    """Read the original version's panel and, when needed, satellite histories."""
    from mst.macro import registry, version

    panel, _ = version.panel_of(models, raw["macro"])
    known = {q.id: q for q in registry.load_registry().variables}
    quantities = {
        name: QuantityData(name, known[name].transform) for name in panel.ids if name in known
    }
    history, last = {}, {}
    for index, name in enumerate(panel.ids):
        _history(name, panel.y[:, index], history, last)
    drivers = _drivers(raw)
    if drivers:
        from mst import plugins
        from mst.drivers import bundle

        satellite = bundle.load(models, raw["satellites"], plugins.registry().satellites)
        if satellite.macro != raw["macro"]:
            raise ValueError("satellite bundle belongs to a different macro version")
        for name in sorted(drivers):
            quantity = name.split(":", 1)[0]
            part = satellite.parts[quantity]
            index = part["driver"].index(name)
            quantities[name] = QuantityData(name, satellite.registry.quantity(quantity).transform)
            key = f"{quantity}.d"
            if key in satellite.arrays:
                _history(name, satellite.arrays[key][:, index], history, last)
    return TranslationData(panel.months[-1].strftime("%Y-%m"), quantities, last, history)


def _stage(path: Path, data: bytes) -> Path:
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".migration-", delete=False
        ) as file:
            temp = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        if temp is not None:
            temp.unlink(missing_ok=True)
        raise
    return temp


def _commit(changes: dict[Path, bytes]) -> None:
    staged, backups, replaced = {}, {}, []
    try:
        for path, data in changes.items():
            staged[path] = _stage(path, data)
            backups[path] = _stage(path, path.read_bytes()) if path.exists() else None
        for path, temp in staged.items():
            os.replace(temp, path)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            backup = backups[path]
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(backup, path)
        raise
    finally:
        for path in (*staged.values(), *backups.values()):
            if path is not None:
                path.unlink(missing_ok=True)


def migrate(directory: Path, models: Path) -> int:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")
    changes, causes, errors = {}, [], []
    for path in sorted(directory.glob("*.json")):
        try:
            original = path.read_bytes()
            raw = json.loads(original)
            if "interventions" in raw or "observations" in raw:
                continue
            scenario = read_legacy(raw, resolver=lambda item: context_of(models, item))
            if raw.get("causes") or raw.get("regime_event") is not None:
                causes.append(path.name)
            changes[path] = (
                json.dumps(scenario.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
        except Exception as error:
            errors.append(f"{path}: {error}")
    if errors:
        raise ValueError("\n".join(errors))
    if causes:
        manifest = directory.parent / f"{directory.name}-migration-causes.txt"
        existing = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
        changes[manifest] = ("\n".join(sorted(set(existing + causes))) + "\n").encode("utf-8")
    _commit(changes)
    return len(changes) - bool(causes)

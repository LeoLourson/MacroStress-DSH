"""The operations of the ``scenario`` tool: list, get, save and validate.

A scenario is ``scenarios/<id>.json`` of the workspace: the event, its sources, the interventions
and the observations, with the name of the human who approved it. ``list`` and ``get`` read the
files, ``save`` checks the contract and writes one atomically, and ``validate`` reports every
problem that keeps the scenario from being computed. The model versions are not part of the file:
``validate`` loads the newest finished pair on disk, or the pair it is given. Nothing here touches
the network: ``validate`` runs on the index and the versions already on disk.
"""

import re
from pathlib import Path
from typing import Any

from .. import store, versions
from ..drivers import bundle as satellite_bundle
from ..macro import version as macro_version
from .build import validate as check
from .spec import Spec

ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
"""The id of a scenario is also the name of its file."""


def catalogue(root: Path) -> dict:
    """Every scenario file with the counts of its event and the name of its approver."""
    found, broken = [], []
    for file in sorted(root.glob("*.json")):
        try:
            item = Spec.model_validate(store.read_json(file))
        except Exception as error:  # noqa: BLE001 - one broken file does not hide the rest
            broken.append({"id": file.stem, "error": f"{type(error).__name__}: {error}"})
            continue
        found.append(
            {
                "id": file.stem,
                "narrative": item.narrative,
                "sources": len(item.sources),
                "observations": len(item.observations),
                "interventions": len(item.interventions),
                "approved_by": item.approved_by,
            }
        )
    return {"scenarios": found, "broken": broken}


def get(root: Path, id: str) -> dict:
    """One scenario file as it is stored."""
    return {"id": id, "spec": load(root, id).model_dump(mode="json")}


def save(root: Path, id: str, raw: dict[str, Any]) -> dict:
    """Check the contract of ``raw`` and write it as ``<id>.json``."""
    spec = Spec.model_validate(raw)
    store.write_json(path(root, id), spec.model_dump(mode="json"))
    return {"id": id, "spec": spec.model_dump(mode="json")}


def validate(
    root: Path,
    id: str,
    models: Path,
    corpus: Path | None = None,
    *,
    macro: str | None = None,
    satellites: str | None = None,
    horizon: int = 12,
    inner: int = 1,
) -> dict:
    """Every problem of the scenario under the chosen versions, or that all is well.

    The versions are the named ones, otherwise the newest finished macro version and the newest
    bundle estimated on it. ``horizon`` and ``inner`` are accepted as the calculation request
    states them; the builder's check does not take them yet and runs its twelve months on one
    inner path.
    """
    spec = load(root, id)
    chosen, bundle = versions_of(models, macro=macro, satellites=satellites)
    report = check(spec, chosen, bundle, as_of=chosen.as_of, corpus=corpus)
    return {"id": id, **report}


def versions_of(
    models: Path, *, macro: str | None = None, satellites: str | None = None
) -> tuple[macro_version.MacroVersion, satellite_bundle.SatelliteBundle]:
    """The macro version and the satellites bundle a calculation runs on.

    An explicit name is loaded as it is. Without one the newest finished macro version is taken,
    and the newest bundle estimated on that version — a bundle of another version is never
    chosen. A missing version is named, and a bundle of another macro version is refused.
    """
    from mst import plugins

    installed = plugins.registry()
    name = macro or _newest(models / "macro", "macro version")
    loaded = macro_version.load(models, name, installed.macro)
    bundle_name = satellites or _newest_satellites(models / "satellites", loaded.name)
    bundle = satellite_bundle.load(models, bundle_name, installed.satellites)
    if bundle.macro != loaded.name:
        raise ValueError(
            f"the satellites {bundle_name} were estimated on macro version {bundle.macro}, "
            f"not {loaded.name}"
        )
    return loaded, bundle


def _newest(directory: Path, what: str) -> str:
    """The newest finished version under ``directory``; a missing one is named."""
    found = versions.folders(directory)
    if not found:
        raise ValueError(
            f"there is no finished {what} under {directory}: estimate one before a calculation"
        )
    return found[0].name


def _newest_satellites(directory: Path, macro: str) -> str:
    """The newest bundle under ``directory`` estimated on ``macro``; a missing one is named."""
    for folder in versions.folders(directory):
        if (store.read_json(folder / "meta.json") or {}).get("macro") == macro:
            return folder.name
    raise ValueError(
        f"there is no satellite bundle under {directory} estimated on macro version {macro}: "
        "estimate the satellites before a calculation"
    )


def load(root: Path, id: str) -> Spec:
    """The scenario ``id`` of ``root``."""
    data = store.read_json(path(root, id))
    if data is None:
        raise FileNotFoundError(f"no scenario {id!r} in {root}")
    return Spec.model_validate(data)


def path(root: Path, id: str) -> Path:
    """The file of scenario ``id``; an id that is not a file name is refused."""
    if not ID.fullmatch(id or ""):
        raise ValueError(f"{id!r} is not a scenario id: use lowercase letters, digits and dashes")
    return root / f"{id}.json"

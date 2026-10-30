"""The operations of the ``scenario`` tool: list, get, save and validate.

A scenario is ``scenarios/<id>.json`` of the workspace. ``list`` and ``get`` read the files,
``save`` checks the contract and writes one atomically, and ``validate`` loads the two model
versions the file names and reports every problem that keeps the scenario from being computed.
Nothing here touches the network: ``validate`` runs on the index and the versions already on
disk.
"""

import re
from pathlib import Path
from typing import Any

from .. import store
from ..drivers import bundle as satellite_bundle
from ..macro import version as macro_version
from .build import validate as check
from .spec import Spec

ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
"""The id of a scenario is also the name of its file."""


def catalogue(root: Path) -> dict:
    """Every scenario file with the versions, horizon and counts it names."""
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
                "macro": item.macro,
                "satellites": item.satellites,
                "horizon": item.horizon,
                "constraints": len(item.constraints),
                "driving_shocks": list(item.driving_shocks),
                "overlays": len(item.overlays),
                "sources": len(item.sources),
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


def validate(root: Path, id: str, models: Path, corpus: Path | None = None) -> dict:
    """Every problem of the scenario with the versions it names, or that all is well."""
    from mst import plugins

    spec = load(root, id)
    installed = plugins.registry()
    macro = macro_version.load(models, spec.macro, installed.macro)
    satellites = satellite_bundle.load(models, spec.satellites, installed.satellites)
    report = check(
        spec, macro, satellites, as_of=macro.as_of, corpus=corpus
    )
    return {"id": id, **report}


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

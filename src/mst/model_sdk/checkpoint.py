"""Checkpoints and progress of a long estimate.

A satellite model may keep each finished quantity in ``context.checkpoints``: ``save`` writes
the part and its residuals under a key the model builds from everything the result depends
on (``key``), and ``load`` returns them when the estimate is started again with the same
inputs. A changed spec, history or model version gives another key, so a stale checkpoint is
never read. The platform removes the folder once the bundle is saved.

``report`` prints a timestamped line to stderr, which a job writes to its log.
"""

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from mst import versions

from .contracts import SatellitePart

_PART = "part."
_RESIDUAL = "residual."


def report(text: str) -> None:
    """One line of progress on stderr, with the time."""
    print(f"{datetime.now():%H:%M:%S} {text}", file=sys.stderr, flush=True)


def key(meta: dict, arrays: dict[str, np.ndarray]) -> str:
    """The name of a checkpoint whose result depends on this meta and these arrays."""
    return versions.digest(meta, arrays)


def load(
    directory: Path | None, name: str
) -> tuple[SatellitePart, dict[str, np.ndarray]] | None:
    """The part and residuals kept under ``name``, or ``None`` when there is none."""
    if directory is None or not (directory / name / "meta.json").is_file():
        return None
    folder = directory / name
    stored = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    with np.load(folder / "arrays.npz") as found:
        arrays = {item: found[item] for item in found.files}
    part = SatellitePart(
        meta=stored["meta"],
        arrays={k[len(_PART) :]: v for k, v in arrays.items() if k.startswith(_PART)},
    )
    residuals = {
        driver: arrays[f"{_RESIDUAL}{i}"] for i, driver in enumerate(stored["residuals"])
    }
    return part, residuals


def save(
    directory: Path | None,
    name: str,
    part: SatellitePart,
    residuals: dict[str, np.ndarray],
) -> None:
    """Keep a finished part under ``name``; written whole or not at all."""
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    drivers = list(residuals)
    arrays = {f"{_PART}{k}": v for k, v in part.arrays.items()}
    arrays.update({f"{_RESIDUAL}{i}": residuals[d] for i, d in enumerate(drivers)})
    draft = Path(tempfile.mkdtemp(dir=directory, prefix=".draft-"))
    try:
        (draft / "meta.json").write_text(
            json.dumps({"meta": part.meta, "residuals": drivers}, ensure_ascii=False),
            encoding="utf-8",
        )
        np.savez(draft / "arrays.npz", **arrays)
        versions._publish(draft, directory / name)
    except BaseException:
        shutil.rmtree(draft, ignore_errors=True)
        raise

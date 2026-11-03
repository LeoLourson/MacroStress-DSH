"""The gates G2–G5 of ``plans/macro-models.md`` from saved runs: ``scripts/macro_gates.py``.

The script reads a list of run ids of a workspace and prints, as JSON, what the runs say
about the macro part of the plan's gates:

- G2 and G3, on the base run, and G4, on every scenario run: the pooled shares of macro
  cells whose band leaves the admissible domain or grows wider than the history, from
  ``checks["paths"]`` of the run's ``meta.json``; the thresholds are the plan's, five
  percent, and the base and the scenarios are told apart by their run id, not by the script;
- G5, on every scenario run: the largest portfolio gain as a multiple of the notional, from
  ``changes.npz`` and the current values of ``results.json``; the threshold is ten times the
  notional;
- the absurd-value shares of the paths: USD/RUB above 300, the key rate, the Fed rate and
  the BBB spread below zero.

Every number is measured on the saved files of the run; a run without a ``meta.json`` or
whose status says it did not finish is reported as broken, and nothing is estimated again.
The floors of the registry make the below-zero shares of the floored variables zero by
construction, so their pressure on the floor is read from ``bound_touch``.
"""

import json
from pathlib import Path

import numpy as np

from mst.metrics.baselines import DOMAIN, WIDTH
from mst.store import Workspace

#: The plan's thresholds, fixed in advance.
SHARE_LIMIT = 0.05
GAIN_LIMIT = 10.0

#: The statuses of a run that wrote no results to read.
UNFINISHED = ("requested", "running", "failed")

#: The absurd values of the plan: a variable, the direction and the boundary.
ABSURD = {
    "usdrub_above_300": ("ru.usdrub", "above", 300.0),
    "key_rate_below_0": ("ru.key_rate", "below", 0.0),
    "fed_rate_below_0": ("gl.fed_rate", "below", 0.0),
    "spread_bbb_below_0": ("ru.spread_bbb", "below", 0.0),
}


def gates(workspace: Path | Workspace, run_ids: list[str]) -> dict:
    """The G2–G5 numbers of every named run; a broken run is named, not skipped."""
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    return {"runs": [_one_run(ws.runs / run_id, run_id) for run_id in run_ids]}


def _one_run(folder: Path, run_id: str) -> dict:
    """Everything one run's saved files say about the macro gates."""
    meta_file = folder / "meta.json"
    if not meta_file.is_file():
        return {"run": run_id, "broken": "no meta.json: the run has not finished"}
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("status") in UNFINISHED:
        return {"run": run_id, "broken": f"status {meta['status']}: the run has not finished"}
    found = {"run": run_id, "scenario": meta.get("scenario")}
    found |= _macro_shares(meta)
    found |= _gain(folder)
    found |= _absurd(meta, folder)
    found |= _bound_touch(meta, folder)
    found["driving_sd"] = meta.get("diagnostics", {}).get("driving_sd")
    found["gates"] = {
        "band_outside_domain_below_five_percent": (
            found["macro_band_outside_domain"] is not None
            and found["macro_band_outside_domain"] < SHARE_LIMIT
        ),
        "band_wider_than_history_below_five_percent": (
            found["macro_band_wider_than_history"] is not None
            and found["macro_band_wider_than_history"] < SHARE_LIMIT
        ),
        "largest_gain_below_ten_notional": (
            found["largest_gain_of_notional"] is not None
            and found["largest_gain_of_notional"] < GAIN_LIMIT
        ),
    }
    return found


def _macro_shares(meta: dict) -> dict:
    """The pooled shares of the macro cells of ``checks["paths"]``; None without checks."""
    cells = {
        key.split("/", 1)[1]: values
        for key, values in (meta.get("checks", {}).get("paths") or {}).items()
        if key.startswith("macro/")
    }
    if not cells:
        return {"macro_band_outside_domain": None, "macro_band_wider_than_history": None}
    # Every cell's share is over the horizon months, so the pooled share is the mean over
    # the macro cells: the share of broken months of macro variables.
    return {
        "macro_band_outside_domain": _pooled(cells, DOMAIN),
        "macro_band_wider_than_history": _pooled(cells, WIDTH),
    }


def _pooled(cells: dict[str, dict], rule: str) -> float | None:
    values = [shares[rule] for shares in cells.values() if rule in shares]
    return float(np.mean(values)) if values else None


def _gain(folder: Path) -> dict:
    """The largest portfolio gain as a multiple of the notional; None without numbers."""
    changes_file, results_file = folder / "changes.npz", folder / "results.json"
    if not changes_file.is_file() or not results_file.is_file():
        return {"largest_gain_of_notional": None}
    with np.load(changes_file) as stored:
        changes = stored["changes"] if "changes" in stored.files else None
    results = json.loads(results_file.read_text(encoding="utf-8"))["positions"]
    notional = sum(row.get("value_current") or 0.0 for row in results)
    if changes is None or notional <= 0:
        return {"largest_gain_of_notional": None}
    # The portfolio gain of a repetition is the sum over the positions that were
    # calculated; an uncalculated one is NaN and adds nothing.
    of_repetition = np.nansum(np.asarray(changes, dtype=float), axis=1)
    if not np.isfinite(of_repetition).any():
        return {"largest_gain_of_notional": None}
    return {"largest_gain_of_notional": float(np.nanmax(of_repetition)) / notional}


def _absurd(meta: dict, folder: Path) -> dict:
    """The shares of paths that reach an absurd value at some month of the horizon."""
    paths_file = folder / "paths.npz"
    if not paths_file.is_file():
        return {"absurd": {name: None for name in ABSURD}}
    with np.load(paths_file) as stored:
        macro = stored["macro"] if "macro" in stored.files else None
    if macro is None or not (meta.get("variables") or []):
        return {"absurd": {name: None for name in ABSURD}}
    index = {id_: k for k, id_ in enumerate(meta["variables"])}
    found = {}
    for name, (id_, direction, boundary) in ABSURD.items():
        if id_ not in index:
            found[name] = None
            continue
        # The paths are in the variables' own units; month 0 is the last observed month.
        column = macro[:, 1:, index[id_]]
        beyond = column > boundary if direction == "above" else column < boundary
        found[name] = float(np.mean(np.any(beyond, axis=1)))
    return {"absurd": found}


def _bound_touch(meta: dict, folder: Path) -> dict:
    """The share of paths that touch a registered floor at some month of the horizon.

    A large share in the base means the model is pressing against the floor — a finding
    the plan asks to record, not a gate.
    """
    from mst.macro.registry import load_registry

    paths_file = folder / "paths.npz"
    ids = list(meta.get("variables") or [])
    if not paths_file.is_file() or not ids:
        return {"bound_touch": {}}
    with np.load(paths_file) as stored:
        macro = stored["macro"] if "macro" in stored.files else None
    if macro is None:
        return {"bound_touch": {}}
    known = {variable.id: variable for variable in load_registry().variables}
    found = {}
    for k, id_ in enumerate(ids):
        variable = known.get(id_)
        # The run's paths are in the variable's own unit, and so is the floor.
        if variable is not None and variable.lower is not None:
            found[id_] = float(np.mean(np.any(macro[:, 1:, k] <= variable.lower, axis=1)))
    return {"bound_touch": found}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="macro_gates", description="G2-G5 of plans/macro-models.md from saved runs"
    )
    parser.add_argument("--workspace", type=Path, required=True, help="the workspace root")
    parser.add_argument("runs", nargs="+", help="the run ids to read")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # Russian labels survive a Windows pipe.
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(gates(args.workspace, args.runs), ensure_ascii=False, indent=1))
    return 0

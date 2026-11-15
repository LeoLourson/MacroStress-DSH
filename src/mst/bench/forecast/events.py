"""The events of the plan: ordinary runs of their scenarios, read into rows.

Every scenario of ``[events]`` is computed as a run of the workspace on the versions of the
check, the plan's portfolio, draws and seed, with the contribution of every element. A run
compares itself with the empty scenario on the same random numbers; the rows take that
comparison, the shift of the median by month against the plan's base scenario run on the
same seed, the admission of the run, the price of its observations and the remoteness of its
interventions. A run that is not admitted fails its gate: its numbers are kept for diagnosis
and are not read as losses.
"""

from pathlib import Path

import numpy as np

from mst import runs
from mst import store as files

from .rows import Row, info


def _run(ws, scenario: str, plan, macro: str, satellites: str, contributions: bool) -> str:
    run_id = runs.new_id()
    request = {
        "scenario": scenario,
        "draws": plan.events.draws,
        "seed": plan.seed,
        "macro": macro,
        "satellites": satellites,
        "portfolio": plan.portfolio,
        "horizon": plan.horizon,
        "contributions": contributions,
    }
    runs.begin(ws, run_id, request)
    runs.execute(ws, run_id)
    meta = files.read_json(runs.folder(ws, run_id) / "meta.json") or {}
    if meta.get("status") != "ok":
        raise RuntimeError(f"{scenario}: run {run_id} ended {meta.get('status')}: "
                           f"{meta.get('error') or meta.get('message')}")
    return run_id


def _read(ws, run_id: str) -> tuple[dict, dict, list | None]:
    folder = runs.folder(ws, run_id)
    found = files.read_json(folder / "contributions.json")
    return (
        files.read_json(folder / "meta.json"),
        files.read_json(folder / "summary.json"),
        None if found is None else found["contributions"],
    )


def _medians(summary: dict) -> list[float]:
    return [row["p50"] for row in summary["losses_by_month"]["by_month"]]


def run(plan, *, workspace: Path, macro: str, satellites: str, value: float,
        progress=print) -> tuple[list[Row], list[dict]]:
    """The rows of every event and a record of its run."""
    if plan.events is None:
        return [], []
    ws = files.Workspace(Path(workspace))
    base_id = _run(ws, plan.events.base, plan, macro, satellites, False)
    _, base, _ = _read(ws, base_id)
    base_medians = _medians(base)
    rows, records = [], [{"scenario": plan.events.base, "run": base_id, "base": True,
                          "loss": base["loss"], "losses_by_month": base["losses_by_month"]}]
    percent = 100.0 / value
    for scenario in plan.events.scenarios:
        progress(f"event {scenario}")
        run_id = _run(ws, scenario, plan, macro, satellites, plan.events.contributions)
        meta, summary, contributions = _read(ws, run_id)
        admitted = bool(meta["admission"]["admitted"])
        rows.append(Row("scenario", "admitted", scenario, None, float(admitted), 1.0, 1,
                        "pass" if admitted else "fail", "the run is admitted"))
        medians = _medians(summary)
        for h in plan.horizons:
            # The loss is positive; the rows speak of the return, so the sign turns once.
            shift = -(medians[h - 1] - base_medians[h - 1]) * percent
            rows.append(info("scenario", "median return shift, %", scenario, h, shift, None,
                             plan.events.draws))
        comparison = summary.get("comparison") or {}
        for block, name in (("loss", "ES95 loss shift, %"), ("path", "path ES95 loss shift, %")):
            found = comparison.get(block, {}).get("es95")
            if found is not None:
                rows.append(info("scenario", name, scenario,
                                 plan.horizon if block == "loss" else None,
                                 found["value"] * percent, None, plan.events.draws))
        peak = int(np.argmax(np.abs(np.array(medians) - np.array(base_medians))))
        rows.append(info("scenario", "peak median return shift, %", scenario, peak + 1,
                         -(medians[peak] - base_medians[peak]) * percent, None,
                         plan.events.draws))
        dist = meta.get("dist") or {}
        rows.append(info("scenario", "price dist", scenario, None, dist.get("distance"),
                         dist.get("standard_error"), dist.get("observations") or 0))
        records.append(
            {
                "scenario": scenario,
                "run": run_id,
                "admission": meta["admission"],
                "interventions": meta.get("interventions"),
                "dist": dist,
                "comparison": comparison,
                "contributions": contributions,
                "loss": summary["loss"],
                "losses_by_month": summary["losses_by_month"],
            }
        )
    return rows, records

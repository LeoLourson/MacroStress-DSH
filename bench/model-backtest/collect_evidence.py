"""Archive compact evidence and apply the unchanged trust and replacement rules.

Run after both halves and all-level episode/grid reports have completed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from mst.metrics import trust
from mst.metrics.backtest import compact, jsonable
from mst.metrics.decision import compare

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/model-trust-2026-11-17"
DESTINATION = ROOT / "bench/model-backtest/experiment-results/2026-11-17"
PORTFOLIO = "model-trust-liquid"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evidence(variant):
    folder = SOURCE / variant
    tuning = read(folder / f"tuning-episodes-{PORTFOLIO}.json")
    validation = read(folder / f"validation-episodes-{PORTFOLIO}.json")
    grid = read(folder / f"all-rolling-{PORTFOLIO}.json")
    rows = tuning["rows"] + validation["rows"]
    keys = [(r["episode"], r["month"], r["level"], r["quantity"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate episode cells across tuning and validation")
    episodes = {
        "rows": rows,
        "admission": trust.admission(rows, unit="episode", block=1, calibration=None),
        "dropped": {"refused": tuning["dropped"]["refused"] | validation["dropped"]["refused"]},
        "constraint_gate": {
            "failed": tuning["constraint_gate"]["failed"] or validation["constraint_gate"]["failed"]
        },
    }
    verdicts = {}
    for level in sorted(grid["levels"]):
        decision = trust.decision(rows, grid["rows"], target=level, grid_block=grid["block_dates"])
        verdicts[level] = {
            "decision": decision,
            **trust.verdict({"episodes": episodes, "grid": grid}, decision),
        }
    return {"grid": grid, "episodes": episodes, "verdicts": verdicts}


def collect(variants):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    found = {name: evidence(name) for name in variants}
    archive = {
        "criteria_sha256": hashlib.sha256(
            (ROOT / "docs/concept/success-criteria.md").read_bytes()
        ).hexdigest(),
        "portfolio": PORTFOLIO,
        "limits": [
            "Validation history was already read before the experiment series; it is not closed.",
            "Portfolio membership is fixed but service coverage changes on early windows.",
            "Passing a low-power criterion is not evidence of nominal tail coverage.",
        ],
        "variants": {},
    }
    for name, data in found.items():
        configuration = read(SOURCE / name / "configuration.json")
        archive["variants"][name] = {
            "configuration": configuration,
            "verdicts": data["verdicts"],
            "grid": compact(data["grid"]),
        }
        for half in ("tuning", "validation"):
            report = read(SOURCE / name / f"{half}-episodes-{PORTFOLIO}.json")
            (DESTINATION / f"{name}-{half}-episodes.json").write_text(
                json.dumps(compact(report), ensure_ascii=False, indent=2), encoding="utf-8"
            )
    incumbent = found["baseline"]
    candidate = found["state"]
    archive["replacement"] = {}
    block = candidate["grid"]["block_dates"]
    if block != incumbent["grid"]["block_dates"]:
        raise ValueError("The two grids have different block lengths")
    for level in candidate["verdicts"]:
        archive["replacement"][level] = compare(
            {
                "episodes": (
                    candidate["episodes"]["rows"], incumbent["episodes"]["rows"], "episode", 1
                ),
                "grid": (candidate["grid"]["rows"], incumbent["grid"]["rows"], "date", block),
            },
            trusted=candidate["verdicts"][level]["trusted"],
            incumbent_trusted=incumbent["verdicts"][level]["trusted"],
            level=level,
            seed=20261029,
        )
    for variant in ("baseline", "band", "state", "band-state", "state-no-momentum"):
        for check in ("rolling", "episodes"):
            source = SOURCE / variant / f"tuning-{check}.json"
            if source.exists():
                (DESTINATION / f"{variant}-tuning-{check}-macro.json").write_text(
                    json.dumps(compact(read(source)), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    for name in ("liquid-catalog.json", "afk-catalog.json"):
        (DESTINATION / name).write_text(
            (SOURCE / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    path = DESTINATION / "evidence.json"
    path.write_text(
        json.dumps(archive, ensure_ascii=False, indent=2, default=jsonable), encoding="utf-8"
    )
    print(path)
    for name, data in found.items():
        print(name, {level: (v["trusted"], v["reasons"]) for level, v in data["verdicts"].items()})
    print("replacement", archive["replacement"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", default=["baseline", "state"])
    collect(parser.parse_args().variants)

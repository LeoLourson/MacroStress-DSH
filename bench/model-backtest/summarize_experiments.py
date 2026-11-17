"""Summarize completed experiment reports without reading another data half."""

import argparse
import json
from pathlib import Path

from mst.metrics.quality import relative

ROOT = Path(__file__).resolve().parents[2] / "experiments/model-trust-2026-11-17"


def summarize(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["rows"]
    own = [row for row in rows if row["level"] == "macro"]
    short = [row for row in own if row["month"] <= 3]
    criteria = report["admission"]["criteria"]
    comparisons = {}
    for benchmark in ("walk", "base", "conditional_walk"):
        paired = [row for row in own if row.get(benchmark) is not None]
        if paired:
            comparisons[benchmark] = relative(paired, "model", benchmark)["score"]
    return {
        "report": str(path.relative_to(ROOT)),
        "windows": len({row.get("date", row.get("episode")) for row in own}),
        "cells": len(own),
        "refused": report["dropped"]["refused"],
        "scores": comparisons,
        "short_below": sum(row["pit"] < 0.05 for row in short) / len(short) if short else None,
        "short_above": sum(row["pit"] > 0.95 for row in short) / len(short) if short else None,
        "failed": {key: entry["failed"] for key, entry in criteria.items() if entry["failed"]},
        "tail_tests": criteria.get("tails"),
        "portfolio_tail": criteria.get("portfolio_tail"),
        "excluded": report["dropped"].get("excluded"),
        "levels": {key: value["pairs"] for key, value in report["levels"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half", choices=("tuning", "validation", "all"), required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    args = parser.parse_args()
    summaries = []
    for name in args.variants:
        for path in sorted((ROOT / name).glob(f"{args.half}-*.json")):
            if not path.name.endswith(".scope.json"):
                summaries.append(summarize(path))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Run one isolated rw experiment using the unchanged production metrics.

Example: python bench/model-backtest/run_experiment.py band --half tuning
The experiment owns its model cache and reports; workspace data are read only.
"""

import argparse
import contextlib
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "model-trust-2026-11-17"
VARIANTS = {
    "baseline": {},
    "band": {"band_fit": True},
    "state": {"stochastic_scale": True},
    "band-state": {"band_fit": True, "stochastic_scale": True},
}


def execute(args):
    from mst.metrics import backtest, rolling
    from mst.metrics.episodes import load_episodes
    from mst.plugins.rw.model import RwModel
    from mst.plugins.rw.spec import load_spec

    spec = load_spec().model_copy(update=VARIANTS[args.variant])
    model = RwModel(spec=spec)
    folder = OUTPUT / args.variant
    folder.mkdir(parents=True, exist_ok=True)
    configuration = {
        "variant": args.variant,
        "spec": spec.model_dump(mode="json"),
        "version": model.version,
        "criteria_sha256": hashlib.sha256(
            (ROOT / "docs/concept/success-criteria.md").read_bytes()
        ).hexdigest(),
    }
    config_file = folder / "configuration.json"
    if config_file.exists():
        if json.loads(config_file.read_text(encoding="utf-8")) != configuration:
            raise ValueError("The experiment configuration changed: use a new variant directory")
    else:
        config_file.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    plan_file = ROOT / "bench/model-backtest/episodes.toml"
    plan = load_episodes(plan_file)
    common = {
        "store": ROOT / "workspace/series",
        "models": folder / "models",
        "episodes": plan_file,
        "as_of": date(2026, 10, 29),
        "macro_id": "rw.joint",
        "workers": args.workers,
    }
    if args.drivers is not None:
        common.update(drivers=args.drivers, portfolio=args.portfolio)
    # Selection happens in the parent; the explicit model is pickled into every worker.
    with patch("mst.models.selected", return_value=model):
        for check in args.checks:
            label = f"{args.half}-{check}{'-portfolio' if args.portfolio else ''}"
            log = folder / f"{label}.md"
            details = folder / f"{label}.json"
            print(f"START {args.variant} {label}", flush=True)
            with log.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
                if check == "rolling":
                    code = rolling.run(
                        **common,
                        half=None if args.half == "all" else args.half,
                        details=details,
                    )
                else:
                    only = [ep.id for ep in plan.episodes if ep.half == args.half]
                    code = backtest.run(
                        **common,
                        only=[] if args.half == "all" else only,
                        rolling=False,
                        details=details,
                    )
            print(f"DONE {args.variant} {label}: metrics exit {code}; {details}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--half", choices=("tuning", "validation", "all"), default="tuning")
    parser.add_argument(
        "--checks", nargs="+", choices=("rolling", "episodes"), default=["rolling", "episodes"]
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drivers", type=Path)
    parser.add_argument("--portfolio")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

"""The check sets of ``bench/`` and their results, kept beside each set in its ``results/``.

Two groups of sets measure different things. The model backtest (``model-backtest``) checks
a model of any level on history, on a grid of dates and on named episodes: its plan lies in
``bench/`` and its command is ``python -m mst.metrics``. The forecast set (``forecast``) reads
today's forecast of a portfolio against the whole history behind it — the width of every level,
the links of the shares, the scale of the portfolio's tail — and runs the scenarios of its plan.
The agent sets check the work of the agent: the corpus search (``corpus-search``, scored by
``python -m mst.corpus eval``), the scenario built from an event (``scenario-from-corpus``) and
the context of an expert interview (``elicitation-context``).

::

  python -m mst.bench forecast       the forecast scorecard of a macro and a satellite version
  python -m mst.bench scenarios      the verdict shares over the reviewed events
  python -m mst.bench elicitation    the context experiment of the expert interview
"""

from pathlib import Path

from mst import store

ROOT = Path(__file__).resolve().parents[3] / "bench"
"""The check sets of the repository."""


def bench_root(root: Path | None = None) -> Path:
    """``root``, or the check sets of the repository; read at call time, so tests can move it."""
    return Path(root) if root is not None else ROOT


def results(folder: Path) -> Path:
    """The results folder of the check set whose input file lies in ``folder``."""
    return folder / "results"


def save_result(folder: Path, name: str, report: dict, summary: str) -> list[Path]:
    """``name.json`` and ``name.md`` in ``folder``: the last result, written over the old one."""
    report_file, summary_file = folder / f"{name}.json", folder / f"{name}.md"
    store.write_json(report_file, report)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(summary, encoding="utf-8", newline="\n")
    return [report_file, summary_file]


from .scenarios import (  # noqa: E402 - the sets read the helpers above
    REVIEW,
    VERDICTS,
    event,
    events,
    save_scenario,
    scenario_summary,
)

__all__ = [
    "REVIEW",
    "ROOT",
    "VERDICTS",
    "bench_root",
    "event",
    "events",
    "results",
    "save_result",
    "save_scenario",
    "scenario_summary",
]

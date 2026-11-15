"""The check sets of ``bench/`` and their results, kept beside each set in its ``results/``.

Two groups of sets measure different things. The forecast set (``forecast``) checks the
numbers of the models against history at every level: the macro variables, the drivers of the
portfolio and the portfolio itself, in sample and out of sample, with the gates its plan fixes.
The agent sets check the work of the agent: the corpus search (``corpus-search``, scored by
``python -m mst.corpus eval``), the scenario built from an event (``scenario-from-corpus``) and
the context of an expert interview (``elicitation-context``). The model backtest of episodes
(``model-backtest``) keeps its plan in ``bench/`` and its command in ``mst.metrics``.

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

"""The results of the check sets of ``bench/``, kept beside each set in its ``results/``.

``model-backtest`` and ``corpus-search`` keep the last result of each model or search as a JSON
report and a Markdown summary; the history is in git. ``scenario-from-corpus`` keeps one folder
per event: the scenario the agent built, the journal of its corpus calls, and ``review.toml``,
which an expert fills by hand after comparing the scenario with the expected decision kept
outside the project. The expected decisions never enter this folder.

::

  python -m mst.bench scenarios      the verdict shares over the reviewed events
"""

import argparse
import sys
import tomllib
from pathlib import Path

from mst import store
from mst.scenario.spec import Spec

ROOT = Path(__file__).resolve().parents[2] / "bench"
"""The check sets of the repository."""
VERDICTS = ("match", "partial", "mismatch")
REVIEW = '''# The expert's review of the scenario of {event}, filled by hand after comparing it with
# the expected decision of the event. verdict: "match", "partial" or "mismatch".
verdict = ""
comment = """
"""
'''


def results(folder: Path) -> Path:
    """The results folder of the check set whose input file lies in ``folder``."""
    return folder / "results"


def save_result(folder: Path, name: str, report: dict, summary: str) -> list[Path]:
    """``name.json`` and ``name.md`` in ``folder``: the last result, written over the old one."""
    report_file, summary_file = folder / f"{name}.json", folder / f"{name}.md"
    store.write_json(report_file, report)
    summary_file.write_text(summary, encoding="utf-8", newline="\n")
    return [report_file, summary_file]


def events(root: Path | None = None) -> dict[str, str]:
    """The texts of the events of scenario-from-corpus by id."""
    root = root or ROOT
    with (root / "scenario-from-corpus" / "events.toml").open("rb") as file:
        return {event["id"]: event["text"] for event in tomllib.load(file)["events"]}


def event(id: str, root: Path | None = None) -> dict:
    """One event as the agent receives it: the id and the text, nothing else."""
    texts = events(root)
    if id not in texts:
        raise ValueError(f"no event {id!r} in bench/scenario-from-corpus/events.toml")
    return {"event": id, "text": texts[id]}


def save_scenario(
    id: str, spec: dict, calls: list[dict], notes: str | None = None, root: Path | None = None
) -> dict:
    """The scenario of event ``id``, its journal and an empty review, in ``results/<id>/``.

    The scenario is checked against the contract first; a scenario saved again replaces the
    old one and its review, which git keeps.
    """
    root = root or ROOT
    text = event(id, root)["text"]
    built = Spec.model_validate(spec).model_dump(mode="json")
    if built.get("approved_by"):
        raise ValueError("a bench scenario is not approved: leave approved_by empty")
    folder = results(root / "scenario-from-corpus") / id
    store.write_json(folder / "scenario.json", built)
    store.write_json(
        folder / "journal.json", {"event": id, "text": text, "calls": calls, "notes": notes}
    )
    (folder / "review.toml").write_text(
        REVIEW.format(event=id), encoding="utf-8", newline="\n"
    )
    return {"event": id, "folder": str(folder), "calls": len(calls)}


def scenario_summary(root: Path | None = None) -> str:
    """The Markdown table of the verdicts over the events that have a saved scenario."""
    root = root or ROOT
    folder = results(root / "scenario-from-corpus")
    counts = dict.fromkeys((*VERDICTS, "not reviewed"), 0)
    rows = []
    for review in sorted(folder.glob("*/review.toml")):
        with review.open("rb") as file:
            verdict = tomllib.load(file).get("verdict", "")
        if verdict and verdict not in VERDICTS:
            raise ValueError(f"{review}: verdict {verdict!r} is not one of {', '.join(VERDICTS)}")
        counts[verdict or "not reviewed"] += 1
        rows.append(f"| {review.parent.name} | {verdict or '-'} |")
    total = sum(counts.values())
    lines = ["# scenario-from-corpus", "", "| verdict | events | share |", "|---|---:|---:|"]
    for name, count in counts.items():
        share = f"{count / total:.2f}" if total else "-"
        lines.append(f"| {name} | {count} | {share} |")
    lines += ["", "| event | verdict |", "|---|---|", *rows, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mst.bench", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scenarios", help="the verdict shares of scenario-from-corpus")
    parser.parse_args(argv)
    summary = scenario_summary()
    file = results(ROOT / "scenario-from-corpus") / "summary.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(summary, encoding="utf-8", newline="\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

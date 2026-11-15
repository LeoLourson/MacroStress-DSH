"""scenario-from-corpus: the scenario an agent builds from the text of an event.

Each event keeps one folder in ``results/``: the scenario the agent built, the journal of its
corpus calls, and ``review.toml``, which an expert fills by hand after comparing the scenario
with the expected decision kept outside the project. The expected decisions never enter this
folder, and a bench scenario is never approved.
"""

import tomllib
from pathlib import Path

from mst import store
from mst.scenario.spec import Spec

from . import bench_root, results

VERDICTS = ("match", "partial", "mismatch")
REVIEW = '''# The expert's review of the scenario of {event}, filled by hand after comparing it with
# the expected decision of the event. verdict: "match", "partial" or "mismatch".
verdict = ""
comment = """
"""
'''


def events(root: Path | None = None) -> dict[str, str]:
    """The texts of the events of scenario-from-corpus by id."""
    root = bench_root(root)
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
    root = bench_root(root)
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
    root = bench_root(root)
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

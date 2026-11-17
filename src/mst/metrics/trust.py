"""The criteria of trust in a forecasting model (``docs/concept/success-criteria.md``).

Every property a forecast must have is a test with the null hypothesis "the model has the
property", and a criterion fails only on evidence against it. A correct model fails the whole
backtest with probability at most ``ALPHA``: half of it goes to the admission and half to the
decision, split evenly between the statistical criteria of each (``ADMISSION_SHARE``,
``DECISION_SHARE``). A criterion that runs ``m`` tests fails when the smallest of its ``m``
p-values is below its share over ``m`` (Bonferroni); the tests it names are those Holm's
step-down procedure rejects at the same level. A property a correct model never breaks has no
allowance: one band outside the domain fails the domain rule.

The admission reads every window of a check, since each date of the grid is forecast from its
own past only, and judges the law of the forecast: the domain, the moves rule against the share
of broken cells a correct band gives (``MOVES_NULL``) and, on the grid, whose dates nobody
selected, every quantity's score against its benchmark beyond the allowance ``DELTA_Q``, the
shares of the facts below and above the band of every level against ``TAIL`` and the tail of
the portfolio: the Acerbi-Szekely ``Z`` and the share of losses beyond VaR95. The calibration —
the tails and the tail of the portfolio — reads the months of the horizon no longer than the
step of the grid, where the windows of neighbouring dates do not overlap and every date is an
independent batch: the long months share their moves with the windows around them and leave a
dozen batches in the whole history. A band too wide is no failure of the tails: the scores
catch it. The decision reads the validation half of the
target level and compares: on the grid the score is not worse than the benchmark's beyond
``DELTA_S``, in the episodes the scenario does not make the forecast worse than the same model
without it. The usefulness of the scenario — the forecast beats the walk that knows the
scenario's path — is additional: reported, never required.

The windows of the grid overlap and the cells of one window move together, so no cell is an
independent draw. A statistic is computed on batches of consecutive windows as long as a
window (an episode is a batch of its own), and the test is Student's on the batch values with
one degree of freedom fewer than batches; with fewer than two batches the property is not
tested. ``mst.metrics.trial`` measures the false rejections and the power this gives.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import t as student

from .baselines import DOMAIN, MOVES
from .quality import relative
from .sanity import benchmark_of

ALPHA = 0.10
"""The probability with which a correct model may fail the whole backtest."""
ADMISSION_SHARE = ALPHA / 2 / 4
"""The share of each statistical criterion of the admission: the moves, the tails, the failing
quantity and the tail of the portfolio."""
DECISION_SHARE = ALPHA / 2 / 2
"""The share of each criterion of the decision: the accuracy and the harm of the scenario."""
DELTA_S = 0.02
"""How much the target level's score on the grid may exceed its benchmark's."""
DELTA_Q = 0.10
"""How much one quantity's score on the grid may exceed its benchmark's."""
MOVES_NULL = 0.005
"""The share of cells a correct band breaks the moves rule in, with two hundred changes known."""
TAIL = 0.05
"""The share of the facts a correct forecast puts below its p5 and above its p95."""

Rows = Sequence[Mapping[str, object]]


def admission(
    rows: Rows, *, unit: str, block: int, calibration: int | None, tail: Mapping | None = None
) -> dict:
    """The admission criteria of one check's rows. ``block`` is the length of a batch in
    windows. ``calibration``, the step of the grid in months, adds what only the grid reads: the
    failing quantity, and the tails and the tail of the portfolio on the months of the horizon
    up to it; ``tail`` is ``mst.metrics.es.tail`` of the portfolio at that month."""
    groups = _groups(rows)
    criteria = {
        "domain": {
            "rule": "no band outside the domain",
            "failed": sorted(
                _name(key) for key, own in groups.items() if any(row[DOMAIN] for row in own)
            ),
        },
        "moves": _family(
            [
                {**_where(key), **_share(own, unit, block, lambda row: row[MOVES], MOVES_NULL)}
                for key, own in groups.items()
                if any(row[MOVES] is not None for row in own)
            ],
            ADMISSION_SHARE,
        ),
    }
    if calibration is None:
        return _admitted(criteria)
    tails = []
    for level in sorted({str(row["level"]) for row in rows}):
        own = [row for row in rows if row["level"] == level and row["month"] <= calibration]
        for side, outside in (
            ("below", lambda row: row["pit"] < TAIL),
            ("above", lambda row: row["pit"] > 1 - TAIL),
        ):
            tails.append({"level": level, "side": side, **_share(own, unit, 1, outside, TAIL)})
    criteria["tails"] = _family(tails, ADMISSION_SHARE)
    failing = []
    for (level, quantity), own in groups.items():
        benchmark = benchmark_of([row for row in rows if row["level"] == level])
        if benchmark is not None:
            found = _score(own, unit, block, benchmark, np.log1p(DELTA_Q))
            if found is not None:
                failing.append({"level": level, "quantity": quantity, **found})
    criteria["failing"] = _family(failing, ADMISSION_SHARE)
    criteria["portfolio_tail"] = _portfolio_tail(tail)
    return _admitted(criteria)


def decision(episodes: Rows, grid: Rows, *, target: str, grid_block: int) -> dict:
    """The comparisons of the decision on the validation half of ``target``: the accuracy and
    the harm, required, and the usefulness, additional. ``episodes`` and ``grid`` are the rows
    of the two checks, either may be empty; a comparison whose check is missing is ``None``."""
    grid = [row for row in _validation(grid) if row["level"] == target]
    episodes = [row for row in _validation(episodes) if row["level"] == target]
    found: dict = {"target": target}
    benchmark = benchmark_of(grid)
    accuracy = (
        None
        if benchmark is None
        else _score(grid, "date", grid_block, benchmark, np.log1p(DELTA_S))
    )
    found["accuracy"] = (
        None
        if accuracy is None
        else {**accuracy, "benchmark": benchmark, "allowed": 1 + DELTA_S, "passed": _kept(accuracy)}
    )
    harm = _score([row for row in episodes if row["base"] is not None], "episode", 1, "base", 0.0)
    found["harm"] = None if harm is None else {**harm, "passed": _kept(harm)}
    useful = _score(
        [row for row in episodes if row["conditional_walk"] is not None],
        "episode",
        1,
        "conditional_walk",
        0.0,
        side="less",
    )
    found["usefulness"] = (
        None
        if useful is None
        else {**useful, "proven": useful["p"] is not None and useful["p"] < DECISION_SHARE}
    )
    return found


def verdict(checks: Mapping[str, Mapping], decided: Mapping) -> dict:
    """Whether the model is trusted: every check complete and admitted, the tail of the
    portfolio checked, every required comparison of the decision met. ``checks`` maps a
    check's name to its report."""
    reasons = []
    portfolio_checked = False
    for name, report in checks.items():
        refused = report.get("dropped", {}).get("refused")
        if refused:
            reasons.append(f"{name}: the model refused {len(refused)} window(s)")
        criteria = report["admission"]["criteria"]
        failed = [key for key, entry in criteria.items() if entry["failed"]]
        if failed:
            reasons.append(f"{name}: admission fails on {', '.join(failed)}")
        portfolio_checked |= bool(criteria.get("portfolio_tail", {}).get("checked"))
        if report.get("constraint_gate", {}).get("failed"):
            reasons.append(f"{name}: a condition of a scenario is almost impossible")
    if not portfolio_checked:
        reasons.append("portfolio_tail: not checked")
    for key in ("accuracy", "harm"):
        entry = decided.get(key)
        if entry is None:
            reasons.append(f"{key}: not checked")
        elif entry["p"] is None:
            reasons.append(f"{key}: not tested, fewer than two batches")
        elif not entry["passed"]:
            reasons.append(f"{key}: fails")
    return {"trusted": not reasons, "reasons": reasons}


def _score(
    rows: Rows, unit: str, block: int, benchmark: str, null: float, *, side: str = "greater"
) -> dict | None:
    """The log score of the model against ``benchmark`` tested against ``null``: its batches'
    log scores give the error; None without a scored cell."""
    overall = relative(rows, "model", benchmark)["score"]
    if overall is None:
        return None
    logs = [
        np.log(score)
        for batch in _batches(rows, unit, block)
        if (score := relative(batch, "model", benchmark)["score"]) is not None
    ]
    return {
        "statistic": float(overall),
        "batches": len(logs),
        "p": _p(float(np.log(overall)) - null, logs, side),
    }


def _share(own: Rows, unit: str, block: int, hit, null: float) -> dict:
    """The share of the cells ``hit`` marks, tested against ``null`` from above; a row ``hit``
    reads as None is outside the share."""
    chosen = [row for row in own if hit(row) is not None]
    if not chosen:
        return {"statistic": None, "null": null, "batches": 0, "p": None}
    share = float(np.mean([bool(hit(row)) for row in chosen]))
    values = [
        float(np.mean([bool(hit(row)) for row in batch])) for batch in _batches(chosen, unit, block)
    ]
    return {"statistic": share, "null": null, "batches": len(values), "p": _p(share - null, values)}


def _portfolio_tail(tail: Mapping | None) -> dict:
    """The Acerbi-Szekely ``Z`` of the grid's windows at a month whose windows do not overlap
    and their share of losses beyond VaR95, each tested from above at half the criterion's
    share with every window a batch; unchecked without a portfolio or with fewer than two
    windows."""
    windows = sorted((tail or {}).get("windows") or [], key=lambda entry: str(entry["id"]))
    batches = [[k] for k in range(len(windows))]
    if len(batches) < 2:
        return {"checked": False, "failed": []}
    beyond = [float(entry["loss"] > entry["var"]) for entry in windows]
    terms = [
        flag * entry["loss"] / entry["es"] / TAIL
        for flag, entry in zip(beyond, windows, strict=True)
    ]
    z_value, share = float(np.mean(terms) - 1), float(np.mean(beyond))
    p_z = _p(z_value, [float(np.mean([terms[k] for k in batch]) - 1) for batch in batches])
    p_beyond = _p(share - TAIL, [float(np.mean([beyond[k] for k in batch])) for batch in batches])
    failed = [
        name
        for name, p in (("z", p_z), ("beyond_var", p_beyond))
        if p is not None and p < ADMISSION_SHARE / 2
    ]
    return {
        "checked": True,
        "windows": len(windows),
        "batches": len(batches),
        "z": z_value,
        "p_z": p_z,
        "beyond_var": share,
        "p_beyond": p_beyond,
        "failed": failed,
    }


def _kept(found: Mapping) -> bool:
    """Whether a tested comparison stands: tested, and its p-value not below its share."""
    return found["p"] is not None and found["p"] >= DECISION_SHARE


def _p(excess: float, values: Sequence[float], side: str = "greater") -> float | None:
    """The one-sided p-value of ``excess`` with Student's error of the batch ``values``; None
    with fewer than two batches."""
    count = len(values)
    if count < 2:
        return None
    se = float(np.std(values, ddof=1)) / np.sqrt(count)
    signed = excess if side == "greater" else -excess
    if se <= 0:
        return 0.0 if signed > 0 else 1.0
    return float(student.sf(signed / se, count - 1))


def _batches(rows: Rows, unit: str, block: int) -> list[list]:
    """The rows in batches of ``block`` consecutive windows; the last batch may be shorter."""
    units = sorted({str(row[unit]) for row in rows})
    batch_of = {name: k // max(block, 1) for k, name in enumerate(units)}
    found: dict[int, list] = {}
    for row in rows:
        found.setdefault(batch_of[str(row[unit])], []).append(row)
    return [found[key] for key in sorted(found)]


def _family(tests: list[dict], share: float) -> dict:
    """Bonferroni over the tests with a p-value; Holm's step-down names the rejected ones."""
    known = sorted((test for test in tests if test["p"] is not None), key=lambda test: test["p"])
    rejected = []
    for k, test in enumerate(known):
        if test["p"] >= share / (len(known) - k):
            break
        rejected.append(test)
    return {
        "alpha": share,
        "tests": tests,
        "untested": sum(1 for test in tests if test["p"] is None),
        "failed": [_label(test) for test in rejected],
    }


def _admitted(criteria: dict) -> dict:
    return {"criteria": criteria, "failed": any(entry["failed"] for entry in criteria.values())}


def _groups(rows: Rows) -> dict[tuple[str, str], list]:
    groups: dict[tuple[str, str], list] = {}
    for row in rows:
        groups.setdefault((str(row["level"]), str(row["quantity"])), []).append(row)
    return dict(sorted(groups.items()))


def _validation(rows: Rows) -> list:
    """The validation half when the rows have one, all of them otherwise."""
    chosen = [row for row in rows if row["half"] == "validation"]
    return chosen or list(rows)


def _where(key: tuple[str, str]) -> dict:
    return {"level": key[0], "quantity": key[1]}


def _name(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}"


def _label(test: Mapping) -> str:
    where = test["level"] if "quantity" not in test else f"{test['level']}/{test['quantity']}"
    return f"{where} ({test['side']})" if "side" in test else where

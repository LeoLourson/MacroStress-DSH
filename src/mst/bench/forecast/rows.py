"""One row per measured number, and the verdict a gate gives it.

Every check of the set returns rows of one shape, so a level, a check or a horizon can be
read, compared between versions and summarised the same way. A row without a gate is
``info``: it is reported, it decides nothing.
"""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

Verdict = Literal["pass", "fail", "info", "short"]
"""``short``: too few historical windows or dates to judge."""


@dataclass(frozen=True)
class Row:
    level: str
    """``macro``, ``shares`` or ``portfolio``."""
    check: str
    item: str
    """The series, the driver, or the statistic of the portfolio."""
    horizon: int | None
    model: float | None
    history: float | None
    """What the model is judged against: history, a baseline or the nominal share."""
    n: int
    """The historical windows or dates behind ``history``."""
    verdict: Verdict
    rule: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _number(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def ratio(level, check, item, horizon, model, history, n, bounds, *, least=8) -> Row:
    """``model / history`` inside ``bounds``; ``short`` below ``least`` windows."""
    model, history = _number(model), _number(history)
    rule = f"{bounds[0]:g} <= model / history <= {bounds[1]:g}"
    if model is None or history is None or history <= 0 or n < least:
        return Row(level, check, item, horizon, model, history, n, "short", rule)
    inside = bounds[0] <= model / history <= bounds[1]
    return Row(level, check, item, horizon, model, history, n, "pass" if inside else "fail", rule)


def distance(level, check, item, horizon, model, history, n, tolerance, *, least=8) -> Row:
    """``|model - history|`` at most ``tolerance``."""
    model, history = _number(model), _number(history)
    rule = f"|model - history| <= {tolerance:g}"
    if model is None or history is None or n < least:
        return Row(level, check, item, horizon, model, history, n, "short", rule)
    inside = abs(model - history) <= tolerance
    return Row(level, check, item, horizon, model, history, n, "pass" if inside else "fail", rule)


def inside(level, check, item, horizon, value, nominal, n, bounds, *, least=8) -> Row:
    """``value`` inside ``bounds``; ``nominal`` is what a calibrated forecast gives."""
    value = _number(value)
    rule = f"{bounds[0]:g} <= value <= {bounds[1]:g}"
    if value is None or n < least:
        return Row(level, check, item, horizon, value, nominal, n, "short", rule)
    found = bounds[0] <= value <= bounds[1]
    return Row(level, check, item, horizon, value, nominal, n, "pass" if found else "fail", rule)


def info(level, check, item, horizon, model, history, n) -> Row:
    return Row(level, check, item, horizon, _number(model), _number(history), n, "info")


def share_gate(rows: list[Row], share: float, name: str) -> dict:
    """A summary gate: at least ``share`` of the judged rows pass."""
    judged = [row for row in rows if row.verdict in ("pass", "fail")]
    passed = sum(row.verdict == "pass" for row in judged)
    verdict = "short" if not judged else ("pass" if passed >= share * len(judged) else "fail")
    return {
        "gate": name,
        "rule": f"at least {share:.0%} of {len(judged)} rows pass",
        "passed": passed,
        "judged": len(judged),
        "verdict": verdict,
    }

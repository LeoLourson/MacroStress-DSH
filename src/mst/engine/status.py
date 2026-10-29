"""The status of a calculated position, and the worst of several (spec 19).

``ok`` numbers enter portfolio arithmetic. ``needs_review`` numbers do too, and the report
discloses them. ``invalid`` never does: the position was not calculated, or its model broke its
own declaration, and the run keeps the position with its reason.
"""

from mst.contracts import ResultStatus

SEVERITY: dict[str, int] = {"ok": 0, "needs_review": 1, "invalid": 2}


def worst(*statuses: str) -> ResultStatus:
    """The least usable of ``statuses``; no status means ``ok``."""
    known = [status for status in statuses if status]
    for status in known:
        if status not in SEVERITY:
            raise ValueError(f"unknown result status: {status!r}")
    if not known:
        return "ok"
    return max(known, key=SEVERITY.__getitem__)  # the union of literal statuses is a status

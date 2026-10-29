"""Fetches every series from its source and stores the new versions."""

from dataclasses import dataclass, field
from datetime import date

import httpx

from .store import SeriesStore, new_versions


@dataclass
class Outcome:
    added: dict[str, int] = field(default_factory=dict)
    """Rows added per series."""
    failed: dict[str, str] = field(default_factory=dict)
    """Why a series was not fetched; its stored rows stay as they were."""


def update(
    store: SeriesStore, sources: dict, http: httpx.Client, *, fetched_on: date, start: date
) -> Outcome:
    outcome = Outcome()
    for series, source in sources.items():
        try:
            fetched = source.fetch(http, start)
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as error:
            outcome.failed[series] = f"{type(error).__name__}: {error}"
            continue
        known = store.read(series)
        rows = new_versions(
            known,
            fetched,
            lag_days=source.lag_days,
            revised=source.revised,
            fetched_on=fetched_on,
        )
        if rows:
            store.write(series, known + rows)
        outcome.added[series] = len(rows)
    return outcome

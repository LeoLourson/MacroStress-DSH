from datetime import date

import httpx

from mst.data.release import Release
from mst.data.sources import load_sources
from mst.data.store import SeriesStore
from mst.data.update import update
from mst.macro.registry import load_registry


class Fixed(Release):
    values: dict[date, float]

    def fetch(self, http, start):
        return {month: value for month, value in self.values.items() if month >= start}


class Broken(Release):
    def fetch(self, http, start):
        raise httpx.ConnectError("no route")


def test_update_stores_new_rows_once_and_reports_failures_per_series(tmp_path):
    store = SeriesStore(tmp_path)
    sources = {
        "gl.vix": Fixed(
            lag_days=1, revised=False, values={date(2011, 12, 1): 9.0, date(2012, 1, 1): 20.0}
        ),
        "gl.brent": Broken(lag_days=1, revised=False),
    }
    outcome = update(
        store, sources, httpx.Client(), fetched_on=date(2012, 3, 1), start=date(2012, 1, 1)
    )
    assert outcome.added == {"gl.vix": 1}
    assert list(outcome.failed) == ["gl.brent"]
    assert [row.value for row in store.read("gl.vix")] == [20.0]

    again = update(
        store, sources, httpx.Client(), fetched_on=date(2012, 3, 2), start=date(2012, 1, 1)
    )
    assert again.added == {"gl.vix": 0}


def test_every_source_feeds_a_registered_variable():
    variables = {variable.id for variable in load_registry().variables}
    assert set(load_sources()) <= variables

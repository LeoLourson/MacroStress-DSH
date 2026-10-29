"""The series tool, the importer and the update task."""

import asyncio
import time
from datetime import date

import pytest
from fastmcp import Client

from mst import importer, jobs, store
from mst.data import sources
from mst.data.store import Observation, SeriesStore
from mst.mcp import build


def call(ws, name, arguments):
    async def run():
        async with Client(build(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def wait(ws, job_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not jobs.status(ws, job_id)["running"]:
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


def test_get_returns_the_values_known_on_a_day(tmp_path):
    ws = store.Workspace(tmp_path)
    SeriesStore(ws.series).write(
        "ru.test",
        [
            Observation(date(2020, 1, 1), 1.0, date(2020, 2, 10), False),
            Observation(date(2020, 1, 1), 1.1, date(2020, 3, 10), False),
            Observation(date(2020, 2, 1), 2.0, date(2020, 3, 10), False),
        ],
    )
    answer = call(ws, "series", {"action": "get", "id": "ru.test", "as_of": "2020-02-15"}).data
    assert answer["as_of"] == "2020-02-15"
    assert answer["rows"] == [
        {"month": "2020-01-01", "value": 1.0, "published": "2020-02-10", "approximate": False}
    ]


def test_search_reaches_the_shipped_sources(tmp_path):
    ws = store.Workspace(tmp_path)
    answer = call(ws, "series", {"action": "search", "text": "moex_index"}).data
    assert answer["sources"] == [{"id": "ru.imoex", "loader": "moex_index"}]


def test_add_writes_to_the_workspace_and_leaves_the_shipped_file(tmp_path):
    ws = store.Workspace(tmp_path)
    shipped = sources.SOURCES_PATH.read_bytes()
    entry = {"id": "gl.test", "loader": "fred", "series_id": "X", "lag_days": 1, "revised": False}

    assert call(ws, "series", {"action": "add", "entry": entry}).data == {"id": "gl.test"}

    assert sources.SOURCES_PATH.read_bytes() == shipped
    assert "gl.test" in sources.load_sources(extra=ws.sources)
    found = call(ws, "series", {"action": "search", "text": "gl.test"}).data
    assert found["sources"] == [{"id": "gl.test", "loader": "fred"}]


def test_add_without_an_entry_is_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "series", {"action": "add"})
    except Exception as error:
        assert "entry is required" in str(error)
    else:
        raise AssertionError("add without an entry must not be accepted")


def test_importer_copies_the_old_series_and_refuses_an_unreadable_file(tmp_path):
    old = tmp_path / "old"
    source = old / "data" / "series"
    source.mkdir(parents=True)
    (source / "gl.vix.csv").write_text(
        "month,value,published,approximate\n2012-01,20.5,2012-02-01,0\n", encoding="utf-8"
    )
    ws = store.Workspace(tmp_path / "ws")
    assert importer.import_series(old, ws)["copied"] == 1
    assert SeriesStore(ws.series).read("gl.vix")[0].value == 20.5

    (source / "bad.csv").write_text("month,value\n2012-01,1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        importer.import_series(old, ws)


def test_update_runs_as_a_job_and_names_an_unknown_series(tmp_path):
    ws = store.Workspace(tmp_path)
    answer = call(ws, "series", {"action": "update", "ids": ["gl.nosuch"]}).data
    assert answer["argv"][:4] == ["-m", "mst.data", "update", "gl.nosuch"]
    wait(ws, answer["job"])
    assert "no source for gl.nosuch" in jobs.log(ws, answer["job"])
    assert jobs.status(ws, answer["job"])["resource"] == "series"

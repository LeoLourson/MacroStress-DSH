"""Reading, searching and appending ``sources.toml``."""

from datetime import date

import pytest
from pydantic import ValidationError

from mst.data.sources import add, load_sources, search, table

VIX = '["gl.vix"]\nloader = "fred"\nseries_id = "VIXCLS"\nlag_days = 1\nrevised = false\n'


def test_search_matches_the_id_or_the_loader():
    assert {row["id"] for row in search("fred")} == {
        "gl.brent",
        "gl.fed_rate",
        "gl.ust_10y",
        "gl.vix",
    }
    assert [row["id"] for row in search("ru.cpi")] == ["ru.cpi"]
    assert {row["id"] for row in search("cbr_currency")} == {"ru.usdrub", "ru.cnyrub"}


def test_append_writes_a_table_the_loader_reads_back(tmp_path):
    path = tmp_path / "sources.toml"
    path.write_text(VIX, encoding="utf-8")
    entry = {
        "id": "ru.test_rate",
        "loader": "cbr_key_rate",
        "lag_days": 1,
        "revised": False,
        "before": [{"since": "2011-12-26", "value": 5.25}],
    }
    assert add(entry, path) == "ru.test_rate"
    loaded = load_sources(path)
    assert set(loaded) == {"gl.vix", "ru.test_rate"}
    assert loaded["ru.test_rate"].before[0].since == date(2011, 12, 26)
    assert loaded["ru.test_rate"].before[0].value == 5.25


def test_a_taken_id_or_an_unknown_field_is_refused(tmp_path):
    path = tmp_path / "sources.toml"
    path.write_text(VIX, encoding="utf-8")
    with pytest.raises(ValueError, match="already has a source"):
        taken = {"id": "gl.vix", "loader": "fred", "series_id": "X", "lag_days": 1}
        add(taken, path)
    with pytest.raises(ValidationError):
        add({"id": "ru.broken", "loader": "fred", "series_id": "X", "nonsense": 1}, path)


def test_table_writes_dates_arrays_and_inline_tables():
    text = table("x", {"a": [date(2020, 1, 1)], "b": [{"k": 1}], "c": "строка", "d": True})
    assert text == '["x"]\na = [2020-01-01]\nb = [{ k = 1 }]\nc = "строка"\nd = true\n'

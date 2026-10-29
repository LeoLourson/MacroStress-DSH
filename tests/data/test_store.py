from datetime import date

import pytest

from mst.data.store import Observation, SeriesStore, new_versions

JAN, FEB, MAR = date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)


def obs(month, value, published, approximate=False):
    return Observation(month, value, published, approximate)


def test_written_rows_read_back_sorted(tmp_path):
    store = SeriesStore(tmp_path)
    rows = [obs(FEB, 2.5, date(2024, 3, 1)), obs(JAN, 0.1 + 0.2, date(2024, 2, 1), True)]
    store.write("ru.cpi", rows)
    assert store.read("ru.cpi") == sorted(rows, key=lambda row: (row.month, row.published))


def test_unknown_series_has_no_rows(tmp_path):
    assert SeriesStore(tmp_path).read("ru.cpi") == []


def test_as_of_returns_the_last_version_published_by_that_day(tmp_path):
    store = SeriesStore(tmp_path)
    store.write(
        "ru.cpi",
        [
            obs(JAN, 1.0, date(2024, 2, 10)),
            obs(JAN, 1.5, date(2024, 3, 10)),
            obs(FEB, 2.0, date(2024, 3, 10)),
        ],
    )
    assert store.as_of("ru.cpi", date(2024, 2, 9)) == []
    assert store.as_of("ru.cpi", date(2024, 3, 9)) == [obs(JAN, 1.0, date(2024, 2, 10))]
    assert [row.value for row in store.as_of("ru.cpi", date(2024, 3, 10))] == [1.5, 2.0]


def test_backfill_of_a_revised_series_dates_rows_by_lag_and_flags_them():
    rows = new_versions([], {JAN: 1.0, FEB: 2.0}, lag_days=20, revised=True, fetched_on=MAR)
    assert rows == [obs(JAN, 1.0, date(2024, 2, 20), True), obs(FEB, 2.0, MAR, True)]


def test_series_that_is_never_revised_is_not_flagged():
    rows = new_versions([], {JAN: 1.0}, lag_days=1, revised=False, fetched_on=MAR)
    assert rows == [obs(JAN, 1.0, date(2024, 2, 1))]


def test_month_that_has_not_ended_is_skipped():
    assert new_versions([], {MAR: 1.0}, lag_days=1, revised=False, fetched_on=MAR) == []
    assert (
        new_versions([], {FEB: 1.0}, lag_days=1, revised=False, fetched_on=date(2024, 2, 29)) == []
    )


def test_after_backfill_new_months_and_revisions_are_dated_by_the_fetch():
    known = [obs(JAN, 1.0, date(2024, 2, 20), True)]
    fetched_on = date(2024, 4, 2)
    rows = new_versions(
        known, {JAN: 1.1, FEB: 2.0}, lag_days=20, revised=True, fetched_on=fetched_on
    )
    assert rows == [obs(JAN, 1.1, fetched_on), obs(FEB, 2.0, fetched_on)]


def test_repeated_fetch_adds_nothing():
    known = [obs(JAN, 1.0, date(2024, 2, 20), True), obs(JAN, 1.1, date(2024, 4, 2))]
    assert (
        new_versions(known, {JAN: 1.1}, lag_days=20, revised=True, fetched_on=date(2024, 5, 1))
        == []
    )


def test_series_id_cannot_escape_the_store(tmp_path):
    with pytest.raises(ValueError):
        SeriesStore(tmp_path).read("../secrets")

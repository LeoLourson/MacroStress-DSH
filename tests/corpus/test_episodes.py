from datetime import date, timedelta
from pathlib import Path

from mst.corpus import episodes as e
from mst.corpus import manifest as m
from mst.corpus import series as s

CORPUS = Path(__file__).resolve().parents[2] / "corpus"
EPISODE = m.Episode(
    id="R90",
    start="2014-12",
    region="ru",
    country="RU",
    title="Обвал",
    factors=["fx", "bank.liquidity"],
    data=["D1"],
    path="episodes/ru/R90.md",
    status="planned",
)
CATALOG = s.Catalog(
    series={
        "x.fx": s.Series(
            title="Курс",
            unit="pct",
            since=date(2014, 1, 1),
            source={"loader": "cbr_currency", "code": "R01235"},
        )
    },
    factors={"RU": {"fx": ["x.fx"]}},
)


class Cache(s.Cache):
    def __init__(self, values):
        self.values = values

    def read(self, series_id):
        return self.values


def daily(start, values):
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


def test_the_window_runs_three_months_before_to_twelve_after():
    span = e.window(EPISODE)
    assert (span.opens, span.start, span.closes) == (
        date(2014, 9, 1),
        date(2014, 12, 1),
        date(2015, 12, 31),
    )


def test_a_move_and_its_recovery_are_measured_from_the_last_value_before_start():
    values = daily(date(2014, 11, 29), [50, 50, 60, 80, 70, 49, 55])
    found = e.movement("fx", "x.fx", CATALOG, values, "cache", e.window(EPISODE))
    assert found.base == (date(2014, 11, 30), 50)
    assert found.extreme == (date(2014, 12, 2), 80)
    assert round(found.change, 6) == 60
    assert (found.days_to_extreme, found.days_to_recover) == (1, 2)


def test_a_card_keeps_hand_written_sections_and_redraws_the_table(tmp_path):
    cache = Cache(daily(date(2014, 11, 29), [50, 50, 60]))
    path, measured = e.write_card(EPISODE, CATALOG, cache, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "| fx | Курс | 50 (30.11.2014) | 60 (01.12.2014) | +20,00 % | 0 | не в окне |" in text
    assert "bank.liquidity: нет рыночного ряда" in text
    assert not e.written(text)
    path.write_text(
        text.replace(e.PLACEHOLDERS["Хронология"], "1 декабря рубль упал [S1/2014 p. 5]."),
        encoding="utf-8",
    )
    e.write_card(EPISODE, CATALOG, Cache(daily(date(2014, 11, 29), [50, 50, 70])), tmp_path)
    again = path.read_text(encoding="utf-8")
    assert "1 декабря рубль упал [S1/2014 p. 5]." in again and "+40,00 %" in again
    assert e.written(again)


def test_no_values_around_the_start_leave_the_factor_unmeasured():
    measured = e.measure(EPISODE, CATALOG, Cache({}))
    assert measured.movements == [] and "fx" in measured.unmeasured


def test_the_corpus_catalog_names_only_known_series():
    catalog = s.load_catalog(CORPUS / "series.toml")
    assert catalog.candidates("G34", "US", "rate.money_market") == ["fred.SOFR"]
    assert catalog.candidates("X3", "TH", "fx") == ["fred.DEXTHUS", "bis.xru.TH"]
    assert catalog.candidates("R5", "RU", "volatility") == ["fred.VIXCLS", "fred.VXOCLS"]

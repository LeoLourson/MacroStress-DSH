import json
from datetime import date

import httpx
import pytest
from pytest import approx

from mst.data.cbr import curve_at
from mst.data.moex import MoexSpread

CURVE = {0.25: 5.0, 1.0: 6.0, 2.0: 7.0}


def test_curve_is_interpolated_linearly_and_held_flat_beyond_its_terms():
    assert curve_at(CURVE, 1.5) == approx(6.5)
    assert curve_at(CURVE, 0.1) == 5.0
    assert curve_at(CURVE, 30.0) == 7.0


def zcyc_page(days):
    header = "".join(f"<th>{term}</th>" for term in ("0,25", "1", "2"))
    rows = "".join(
        f'<tr> <td><a href="zcyc/?DateTo={day}">{day}</a></td> '
        "<td>5,00</td> <td>6,00</td> <td>7,00</td> </tr>"
        for day in days
    )
    return (
        '<table class="data spaced"><tr><th rowspan="2">Дата</th>'
        f'<th colspan="3">Срок до погашения, лет</th></tr><tr>{header}</tr>{rows}</table>'
    )


def index_page(rows):
    return {
        "history": {"columns": ["TRADEDATE", "YIELD", "DURATION"], "data": rows},
        "history.cursor": {
            "columns": ["INDEX", "TOTAL", "PAGESIZE"],
            "data": [[0, len(rows), 100]],
        },
    }


@pytest.fixture
def seen():
    return []


@pytest.fixture
def http(seen):
    indices = {
        "OLD": [
            ["2020-12-01", 8.0, 365],
            ["2020-12-02", 9.0, 365],
            ["2020-12-03", 3000.0, 365],
            ["2021-01-11", 50.0, 365],
        ],
        "NEW": [
            ["2020-12-02", 0, 0],
            ["2021-01-11", 7.5, 547.5],
            ["2021-01-12", None, None],
            ["2021-01-13", 0, 0],
        ],
    }

    days = ["01.12.2020", "02.12.2020", "03.12.2020", "11.01.2021", "12.01.2021", "13.01.2021"]

    def respond(request):
        seen.append(request)
        if "zcyc" in request.url.path:
            return httpx.Response(200, text=zcyc_page(days))
        secid = request.url.path.rsplit("/", 1)[1].removesuffix(".json")
        till = request.url.params.get("till", "9999-12-31")
        since = request.url.params["from"]
        rows = [row for row in indices[secid] if since <= row[0] <= till]
        return httpx.Response(200, text=json.dumps(index_page(rows)))

    return httpx.Client(transport=httpx.MockTransport(respond))


def test_spread_is_index_yield_over_the_curve_at_its_duration_spliced_by_month(http, seen):
    source = MoexSpread(
        loader="moex_spread",
        old="OLD",
        new="NEW",
        splice=date(2021, 1, 1),
        lag_days=1,
        revised=False,
    )
    values = source.fetch(http, date(2020, 12, 1))
    # December: the old index at one year, 8, 9 and an outlier of 3000 over a 6 % curve, whose
    # median is 3; January: the new index at 1.5 years over 6.5 %, its zero and empty rows dropped.
    assert values == {date(2020, 12, 1): approx(3.0), date(2021, 1, 1): approx(1.0)}
    curve_requests = [request for request in seen if "zcyc" in request.url.path]
    assert len(curve_requests) == date.today().year - 2020 + 1


def test_splice_must_be_the_first_day_of_a_month():
    with pytest.raises(ValueError, match="first day"):
        MoexSpread(
            loader="moex_spread",
            old="OLD",
            new="NEW",
            splice=date(2021, 1, 15),
            lag_days=1,
            revised=False,
        )


def test_skipped_days_are_left_out(http):
    source = MoexSpread(
        loader="moex_spread",
        old="OLD",
        new="NEW",
        splice=date(2021, 1, 1),
        skip=[(date(2020, 12, 3), date(2020, 12, 3))],
        lag_days=1,
        revised=False,
    )
    values = source.fetch(http, date(2020, 12, 1))
    assert values[date(2020, 12, 1)] == approx(2.5)


def test_a_skipped_range_must_not_end_before_it_starts():
    with pytest.raises(ValueError, match="ends before"):
        MoexSpread(
            loader="moex_spread",
            old="OLD",
            new="NEW",
            splice=date(2021, 1, 1),
            skip=[(date(2020, 12, 3), date(2020, 12, 1))],
            lag_days=1,
            revised=False,
        )

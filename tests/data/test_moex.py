import json
from datetime import date

import httpx
from pytest import approx, raises

from mst.data.moex import MoexIndex, MoexShare


def page(rows, total):
    return {
        "history": {"columns": ["TRADEDATE", "CLOSE"], "data": rows},
        "history.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[0, total, 2]]},
    }


def test_index_closes_are_read_page_by_page_and_averaged():
    pages = {
        "0": page([["2012-01-03", 1444.76], ["2012-01-04", 1447.24]], 3),
        "2": page([["2012-02-01", 1500.0]], 3),
    }
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, text=json.dumps(pages[request.url.params["start"]]))

    source = MoexIndex(loader="moex_index", secid="IMOEX", lag_days=1, revised=False)
    values = source.fetch(httpx.Client(transport=httpx.MockTransport(respond)), date(2012, 1, 1))
    assert values == {date(2012, 1, 1): approx(1446.0), date(2012, 2, 1): 1500.0}
    assert "/securities/IMOEX.json" in seen[0].url.path
    assert seen[0].url.params["from"] == "2012-01-01"


def share_client(boards):
    """ISS answers for each board: ``boards[board]`` is a list of (day, close)."""

    def respond(request):
        board = request.url.path.split("/boards/")[1].split("/")[0]
        rows = [list(row) for row in boards.get(board, [])]
        return httpx.Response(200, text=json.dumps(page(rows, len(rows))))

    return httpx.Client(transport=httpx.MockTransport(respond))


def share(**fields):
    return MoexShare(loader="moex_share", secid="GMKN", lag_days=1, revised=False, **fields)


def test_a_later_board_wins_on_common_days_and_splits_rescale_earlier_closes():
    http = share_client(
        {
            "EQBS": [["2024-03-28", 15000.0], ["2024-04-01", 15054.0]],
            "TQBR": [["2024-04-01", 15100.0], ["2024-04-08", 152.96]],
        }
    )
    source = share(boards=["EQBS", "TQBR"], splits=[{"day": "2024-04-08", "ratio": 100.0}])
    daily = source.daily(http, date(2024, 1, 1))
    assert daily == {
        date(2024, 3, 28): approx(150.0),
        date(2024, 4, 1): approx(151.0),
        date(2024, 4, 8): 152.96,
    }


def test_an_undeclared_split_stops_the_fetch():
    http = share_client({"TQBR": [["2024-04-01", 15054.0], ["2024-04-08", 152.96]]})
    with raises(ValueError, match="declare the split"):
        share(boards=["TQBR"]).fetch(http, date(2024, 1, 1))

import io
import json
from datetime import date, datetime

import httpx
import openpyxl
import pytest

from mst.data.moex import MoexBondSpread, MoexIndex, MoexShare
from mst.drivers.catalogs import slug
from mst.drivers.catalogs.cbr import CbrActivities
from mst.drivers.catalogs.moex import MoexBonds, MoexShares


def block(name, columns, rows, cursor=None):
    body = {name: {"columns": columns, "data": rows}}
    if cursor is not None:
        body[f"{name}.cursor"] = {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [cursor]}
    return body


def iss(routes):
    """A client answering ISS paths with ``routes[path](params)``."""

    def respond(request):
        path = request.url.path.removeprefix("/iss/")
        return httpx.Response(200, text=json.dumps(routes[path](request.url.params)))

    return httpx.Client(transport=httpx.MockTransport(respond))


def security(boards, fields):
    board_columns = ["boardid", "market", "history_from"]
    return lambda params: {
        **block("boards", board_columns, boards),
        **block("description", ["name", "title", "value"], [[k, k, v] for k, v in fields.items()]),
    }


def emitter(inn, title):
    columns = ["EMITTER_ID", "TITLE", "SHORT_TITLE", "INN"]
    return lambda params: block("emitter", columns, [[1, title, title, inn]])


def test_slug_is_stable_and_latin():
    assert slug("строительство зданий и сооружений") == "stroitelstvo_zdaniy_i_sooruzheniy"
    assert slug("ВСЕГО") == "vsego"


def test_shares_are_placed_under_the_sector_whose_index_holds_them():
    analytics = ["indexid", "tradedate", "ticker", "shortnames", "secids", "weight"]
    routes = {
        "statistics/engines/stock/markets/index/analytics/MOEXFN.json": lambda params: block(
            "analytics", analytics, [["MOEXFN", "", "SBER", "", "SBER", 10]], [0, 1, 20]
        ),
        "statistics/engines/stock/splits.json": lambda params: block(
            "splits", ["tradedate", "secid", "before", "after"], [["2024-07-15", "VTBR", 5000, 1]]
        ),
        "engines/stock/markets/shares/boards/TQBR/securities.json": lambda params: block(
            "securities",
            ["SECID", "SHORTNAME", "SECTYPE", "ISIN"],
            [["SBER", "Сбербанк", "1", "RU0009029540"], ["VTBR", "ВТБ", "1", "RU000A0JP5V6"],
             ["FUND", "Фонд", "9", "RU000X"]],  # fmt: skip
        ),
        "securities/SBER.json": security(
            [["EQBR", "shares", "2012-01-03"], ["TQBR", "shares", "2013-03-25"],
             ["SMAL", "shares", "2012-01-03"], ["EQRP", "ccp", "2012-01-03"]],  # fmt: skip
            {"EMITTER_ID": "484"},
        ),
        "securities/VTBR.json": security([["TQBR", "shares", "2013-03-25"]], {}),
        "emitters/484.json": emitter("7707083893", "ПАО Сбербанк"),
    }
    catalog = MoexShares(
        loader="moex_shares", board="TQBR", boards=["EQBR", "TQBR"], root="shares",
        root_label="Акции", lag_days=1, revised=False,
        sectors=[{"id": "financials", "label": "Финансы", "index": "MOEXFN"}],
    )  # fmt: skip
    found = catalog.discover(iss(routes))
    objects = {o.id: o for o in found.objects}
    assert list(objects) == ["shares", "financials", "sber", "vtbr"]
    assert objects["sber"].parent == "financials" and objects["vtbr"].parent == "shares"
    assert objects["sber"].codes["inn"] == ["7707083893"]
    assert isinstance(found.drivers["financials"], MoexIndex)
    sber, vtbr = found.drivers["sber"], found.drivers["vtbr"]
    assert isinstance(sber, MoexShare) and sber.boards == ["EQBR", "TQBR"]
    assert vtbr.splits[0].day == date(2024, 7, 15) and vtbr.splits[0].ratio == pytest.approx(2e-4)


def test_bonds_hang_from_their_issuer_and_the_group_of_their_index():
    analytics = ["indexid", "tradedate", "ticker", "shortnames", "secids", "weight"]
    routes = {
        "statistics/engines/stock/markets/index/analytics/RUCBCPAANS.json": lambda params: block(
            "analytics", analytics, [["", "", "", "", "RU1", 1], ["", "", "", "", "RU2", 1]]
        ),
        "securities/RU1.json": security([], {"EMITTER_ID": "7", "NAME": "ГТЛК 001P-04"}),
        "securities/RU2.json": security([], {"EMITTER_ID": "7", "NAME": "ГТЛК 001P-05"}),
        "emitters/7.json": emitter("7720261827", "АО ГТЛК"),
    }
    catalog = MoexBonds(
        loader="moex_bonds", root="bonds", root_label="Облигации", lag_days=1, revised=False,
        groups=[{"id": "group_aa", "label": "AA", "index": "RUCBCPAANS"}],
    )  # fmt: skip
    found = catalog.discover(iss(routes))
    objects = {o.id: o for o in found.objects}
    assert objects["ru1"].parent == objects["ru2"].parent == "issuer_7"
    assert objects["issuer_7"].parent == "group_aa"
    assert objects["issuer_7"].codes == {"emitter": ["7"], "inn": ["7720261827"]}
    assert isinstance(found.drivers["ru1"], MoexBondSpread) and "issuer_7" not in found.drivers


def test_activities_nest_by_indent_and_carry_their_okved_codes():
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Задолженность ..., млн руб."])
    sheet.append(["Виды", datetime(2019, 2, 1)])
    for label in ["ВСЕГО", "строительство, из них:", "        строительство зданий",
                  "транспорт", "на завершение расчетов"]:  # fmt: skip
        sheet.append([label, 1])
    buffer = io.BytesIO()
    book.save(buffer)
    http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=buffer.getvalue()))
    )
    catalog = CbrActivities(
        loader="cbr_activities", lag_days=37, revised=True, skip=["на завершение расчетов"],
        okved={"строительство": ["41", "42", "43"]},
    )  # fmt: skip
    found = catalog.discover(http)
    parents = {o.id: o.parent for o in found.objects}
    assert parents == {
        "economy": None,
        "stroitelstvo": "economy",
        "stroitelstvo_zdaniy": "stroitelstvo",
        "transport": "economy",
    }
    assert found.objects[1].codes == {"okved": ["41", "42", "43"]}
    assert found.drivers["stroitelstvo_zdaniy"].activity == "строительство зданий"
    assert "на завершение расчетов" in found.skipped

import io
from datetime import date

import httpx
import openpyxl
import pytest
from pytest import approx

from mst.data.rosstat import RosstatHousingPrice, RosstatMonthly, RosstatShortTerm, chain

PAGE = "https://rosstat.gov.ru/statistics/price"


def workbook(sheets):
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def client(files, seen=None):
    """Serves PAGE linking every file, and the files themselves under /storage/mediabank/."""
    links = "".join(f'<a href="/storage/mediabank/{name}">XLSX</a>' for name in files)

    def respond(request):
        if seen is not None:
            seen.append(str(request.url))
        if str(request.url) == PAGE:
            return httpx.Response(200, text=f"<html>{links}</html>")
        name = request.url.path.removeprefix("/storage/mediabank/")
        return httpx.Response(200, content=files[name])

    return httpx.Client(transport=httpx.MockTransport(respond))


def monthly(**fields):
    return RosstatMonthly(
        loader="rosstat_monthly",
        page=PAGE,
        file=r"ipc_mes_\d{2}-\d{4}\.xlsx",
        sheet="01",
        lag_days=14,
        revised=True,
        **fields,
    )


MONTH_NAMES = [
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
]  # fmt: skip


def cpi_sheet():
    rows = [["Индексы потребительских цен"], [None, 2011, "2012*)"], ["к концу предыдущего месяца"]]
    for number, name in enumerate(MONTH_NAMES, 1):
        rows.append([name + ("1)" if number == 1 else ""), 100.5, {1: 101, 2: "102,0"}.get(number)])
        if number == 3:
            rows.append(["1 квартал", 101.0, "108,,9"])
    rows += [["к декабрю предыдущего года"], ["декабрь", 106.1, 99.0]]
    return rows


def test_monthly_table_is_found_by_its_file_pattern_and_chained_from_the_base():
    files = {
        "nedel_Ipc.xlsx": b"not this one",
        "ipc_mes_09-2026.xlsx": workbook({"Содержание": [["Содержание"]], "01": cpi_sheet()}),
    }
    seen = []
    values = monthly(base=date(2011, 12, 1)).fetch(client(files, seen), date(2012, 1, 1))
    assert values == {date(2012, 1, 1): approx(101.0), date(2012, 2, 1): approx(103.02)}
    assert seen == [PAGE, "https://rosstat.gov.ru/storage/mediabank/ipc_mes_09-2026.xlsx"]


def test_monthly_table_without_a_base_is_stored_as_published():
    files = {"ipc_mes_09-2026.xlsx": workbook({"01": cpi_sheet()})}
    values = monthly().fetch(client(files), date(2011, 1, 1))
    assert values[date(2011, 12, 1)] == 100.5
    assert values[date(2012, 2, 1)] == 102.0
    assert len(values) == 14


def test_chain_refuses_a_gap_after_the_base():
    changes = {date(2012, 1, 1): 101.0, date(2012, 3, 1): 101.0}
    with pytest.raises(ValueError, match="2012-02"):
        chain(changes, date(2011, 12, 1))


def test_a_missing_link_or_a_broken_workbook_is_an_error():
    with pytest.raises(ValueError, match="ipc_mes"):
        monthly().fetch(client({"other.xlsx": b""}), date(2012, 1, 1))
    with pytest.raises(ValueError, match="not a workbook"):
        monthly().fetch(client({"ipc_mes_09-2026.xlsx": b"<html>"}), date(2012, 1, 1))


def test_a_cell_that_is_not_a_number_is_an_error():
    rows = [[None, 2012], ["январь", "n/a"]]
    files = {"ipc_mes_09-2026.xlsx": workbook({"01": rows})}
    with pytest.raises(ValueError, match="n/a"):
        monthly().fetch(client(files), date(2012, 1, 1))


HEADER = [None, "Год", "Кварталы", None, None, None, "Янв.", "Фев.", "Март"]


def short_term(sheet="4.1", **fields):
    return RosstatShortTerm(
        loader="rosstat_short_term",
        page=PAGE,
        file=r"ind_\d{2}-\d{4}\.xlsx",
        sheet=sheet,
        lag_days=35,
        revised=True,
        **fields,
    )


def test_short_term_sections_are_spliced_with_the_later_one_winning():
    rows = [
        HEADER,
        ["4.1.5. Уровень безработицы в возрасте 15-72 лет, %"],
        ["/ Unemployment rate 15 to 72 years old, percent"],
        ["2016", 5.5, 5.9, 5.6, 5.2, 5.3, 5.8, 5.8, 6.0],
        ["20171)", 5.2, 5.5, 5.0, 5.0, 5.1, 5.6, 5.6, 5.4],
        ["4.1.6. Уровень безработицы в возрасте 15 лет и старше6), %"],
        ["2017", 5.2, 5.5, 5.0, 5.0, 5.1, 5.7, 5.6, 5.4],
        ["2018", 4.8, 5.1, 4.8, 4.6, 4.8, 5.2, None, "…"],
        ["4.1.7. Численность официально зарегистрированных безработных"],
        ["2018", 1, 1, 1, 1, 1, 9, 9, 9],
    ]
    files = {"ind_08-2026.xlsx": workbook({"4.1 ": rows})}
    source = short_term(sections=["4.1.5.", "4.1.6."])
    values = source.fetch(client(files), date(2016, 1, 1))
    assert values == {
        date(2016, 1, 1): 5.8,
        date(2016, 2, 1): 5.8,
        date(2016, 3, 1): 6.0,
        date(2017, 1, 1): 5.7,
        date(2017, 2, 1): 5.6,
        date(2017, 3, 1): 5.4,
        date(2018, 1, 1): 5.2,
    }


def test_a_section_missing_from_the_sheet_is_named():
    files = {"ind_08-2026.xlsx": workbook({"4.1": [HEADER, ["4.1.6. Уровень"]]})}
    with pytest.raises(ValueError, match="4.1.5."):
        short_term(sections=["4.1.5."]).fetch(client(files), date(2016, 1, 1))


def test_short_term_block_of_a_section_is_chained_from_the_base():
    rows = [
        HEADER,
        ["4.2.1. Среднемесячная номинальная начисленная заработная плата"],
        ["в % к предыдущему периоду"],
        ["2012", None, 90.0, 99.0, 99.0, 99.0, 50.0, 50.0, 50.0],
        ["4.2.2. Реальная начисленная заработная плата"],
        ["/ Accrued average monthly wages per employee"],
        ["2012", 108.4, 110.3, 111.3, 106.3, 106.2, 110.5, 112.1, 109.0],
        ["в % к предыдущему периоду"],
        ["/ percent of previous period"],
        ["2012", None, 89.5, 107.2, 96.3, 114.2, "72,23)", 100.8, "\xa0105,7"],
        ["1) Начиная с сентября 2019 года"],
    ]
    files = {"ind_08-2026.xlsx": workbook({"4.2 ": rows})}
    source = short_term(
        sheet="4.2", sections=["4.2.2."], block="в % к предыдущему периоду", base=date(2011, 12, 1)
    )
    values = source.fetch(client(files), date(2012, 1, 1))
    assert values == {
        date(2012, 1, 1): approx(72.2),
        date(2012, 2, 1): approx(72.2 * 1.008),
        date(2012, 3, 1): approx(72.2 * 1.008 * 1.057),
    }


def test_housing_price_takes_year_ends_and_overrides_them_with_quarters():
    annual = [
        ["К содержанию"],
        [None, "СРЕДНИЕ ЦЕНЫ на первичном рынке жилья"],
        ["Год", "Все типы квартир", "Квартиры среднего качества (типовые)"],
        [2011, 47000.0, 1.0],
        [2012, 50000.0, 1.0],
        [2021, 98909, 1.0],
        ["* без учета статистической информации"],
    ]
    quarters = [
        ["К содержанию", "Средние цены на первичном рынке жилья"],
        [None, "I квартал", None, "II квартал", None, "III квартал", None, "IV квартал"],
        [None, "Все типы квартир", "из них:", "Все типы квартир", None, "Все типы квартир"],
        ["643 - Российская Федерация", 83177.29, 1, 89007.64, 1, 93536.96, 1, 98908.96, 1],
        ["030 - Центральный федеральный округ", 9, 9, 9, 9, 9, 9, 9, 9],
    ]
    files = {
        "dinamika_sred_1998-2025.xlsx": workbook({"Содержание": [], "1": annual}),
        "sred_cen_perv_4kv-2021.xlsx": workbook({"Содержание": [], "1": quarters}),
        "sred_cen_vtor_4kv-2021.xlsx": b"secondary market, not read",
    }
    source = RosstatHousingPrice(
        loader="rosstat_housing_price",
        page=PAGE,
        annual=r"dinamika_sred_\d{4}-\d{4}\.xlsx",
        quarterly=r"sred_cen_perv_\dkv-(\d{4})_?\.xlsx",
        lag_days=35,
        revised=True,
    )
    values = source.fetch(client(files), date(2012, 1, 1))
    assert values == {
        date(2012, 12, 1): 50000.0,
        date(2021, 3, 1): 83177.29,
        date(2021, 6, 1): 89007.64,
        date(2021, 9, 1): 93536.96,
        date(2021, 12, 1): 98908.96,
    }

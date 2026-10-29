from datetime import date

import httpx
from pytest import approx

from mst.data.cbr import CbrCurrency, CbrKeyRate, CbrRuonia, CbrZeroCoupon, Step
from mst.data.daily import monthly_means


def soap(rows):
    return (
        '<?xml version="1.0" encoding="utf-8"?><soap:Envelope '
        'xmlns:soap="http://www.w3.org/2003/05/soap-envelope"><soap:Body>'
        f'<R xmlns="http://web.cbr.ru/"><X><Y xmlns="">{rows}</Y></X></R>'
        "</soap:Body></soap:Envelope>"
    )


def client(body, seen, encoding="utf-8"):
    def respond(request):
        seen.append(request)
        return httpx.Response(200, content=body.encode(encoding))

    return httpx.Client(transport=httpx.MockTransport(respond))


def test_monthly_means_average_the_days_of_each_month():
    daily = {date(2024, 1, 9): 1.0, date(2024, 1, 31): 3.0, date(2024, 2, 1): 5.0}
    assert monthly_means(daily) == {date(2024, 1, 1): 2.0, date(2024, 2, 1): 5.0}


def test_key_rate_before_its_history_comes_from_the_declared_steps():
    rows = "".join(
        f"<KR><DT>2013-09-{day}T00:00:00+04:00</DT><Rate>5.50</Rate></KR>" for day in (16, 13)
    )
    source = CbrKeyRate(
        loader="cbr_key_rate",
        lag_days=1,
        revised=False,
        before=[
            Step(since=date(2011, 12, 26), value=5.25),
            Step(since=date(2012, 9, 14), value=5.5),
        ],
    )
    seen = []
    values = source.fetch(client(soap(rows), seen), date(2012, 1, 1))
    assert b"<fromDate>2012-01-01</fromDate>" in seen[0].content
    assert values[date(2012, 1, 1)] == 5.25
    assert values[date(2012, 9, 1)] == approx((13 * 5.25 + 17 * 5.5) / 30)
    assert values[date(2013, 8, 1)] == 5.5
    assert values[date(2013, 9, 1)] == 5.5
    assert len(values) == 21


def test_ruonia_is_averaged_by_month():
    rows = "".join(
        f"<ro><D0>2012-01-{day}T00:00:00+04:00</D0><ruo>{rate}</ruo><vol>1</vol></ro>"
        for day, rate in ((10, "4.5200"), (11, "4.4800"))
    )
    source = CbrRuonia(loader="cbr_ruonia", lag_days=3, revised=False)
    assert source.fetch(client(soap(rows), []), date(2012, 1, 1)) == {date(2012, 1, 1): approx(4.5)}


def test_currency_rate_is_per_one_unit():
    body = (
        '<?xml version="1.0" encoding="windows-1251"?><ValCurs ID="R01375">'
        '<Record Date="11.01.2012" Id="R01375"><Nominal>10</Nominal><Value>50,4805</Value></Record>'
        '<Record Date="12.01.2012" Id="R01375"><Nominal>1</Nominal><Value>5,0000</Value></Record>'
        "</ValCurs>"
    )
    source = CbrCurrency(loader="cbr_currency", code="R01375", lag_days=1, revised=False)
    seen = []
    values = source.fetch(client(body, seen, "cp1251"), date(2012, 1, 1))
    assert values == {date(2012, 1, 1): approx((5.04805 + 5.0) / 2)}
    assert seen[0].url.params["VAL_NM_RQ"] == "R01375"
    assert seen[0].url.params["date_req1"] == "01/01/2012"


def test_zero_coupon_yield_is_read_for_its_term():
    header = "".join(f"<th>{term}</th>" for term in ("0,25", "0,5", "1", "10"))
    rows = "".join(
        f'<tr> <td><a href="zcyc/?DateTo={day}">{day}</a></td> '
        + "".join(f"<td>{value}</td> " for value in values)
        + "</tr>"
        for day, values in (
            ("20.01.2012", ("5,63", "5,88", "6,34", "8,49")),
            ("19.01.2012", ("5,64", "5,89", "6,36", "8,51")),
        )
    )
    body = (
        '<table class="data spaced"><tr><th rowspan="2">Дата</th>'
        f'<th colspan="4">Срок до погашения, лет</th></tr><tr>{header}</tr>{rows}</table>'
    )
    source = CbrZeroCoupon(loader="cbr_zcyc", years="1", lag_days=1, revised=False)
    seen = []
    assert source.fetch(client(body, seen), date(2012, 1, 1)) == {date(2012, 1, 1): approx(6.35)}
    assert seen[0].url.params["UniDbQuery.From"] == "01.01.2012"

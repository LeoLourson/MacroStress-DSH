from datetime import date

import httpx
import pytest

from mst.corpus import series as s


def test_bis_rows_are_read_from_sdmx_csv():
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(
            200,
            text="FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\nD,TR,2018-08-01,17.75\n"
            "D,TR,2018-08-02,\nQ,TR,2018-Q3,24\n",
        )

    loader = s.Bis(loader="bis", flow="WS_CBPOL", key="D.TR")
    values = loader.daily(httpx.Client(transport=httpx.MockTransport(respond)), date(2018, 1, 1))
    assert values == {date(2018, 8, 1): 17.75, date(2018, 9, 1): 24.0}
    assert seen[0].url.path == "/api/v1/data/WS_CBPOL/D.TR/all"
    assert seen[0].headers["accept"].startswith("application/vnd.sdmx.data+csv")


def test_old_roubles_are_redenominated_only_when_the_jump_shows():
    old = {date(1997, 12, 30): 5960.0, date(1998, 1, 6): 5.96}
    assert s.redenominate(old, date(1998, 1, 1))[date(1997, 12, 30)] == 5.96
    new = {date(1997, 12, 30): 5.96, date(1998, 1, 6): 5.97}
    assert s.redenominate(new, date(1998, 1, 1)) == new


def test_year_on_year_change_of_an_index():
    values = {date(2020, 1, 1): 100.0, date(2021, 1, 1): 104.0}
    assert s.year_on_year(values) == {date(2021, 1, 1): pytest.approx(4.0)}

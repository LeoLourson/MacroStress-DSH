import json
from datetime import date

import httpx
import pytest

from mst.data.fred import Fred

SOURCE = Fred(loader="fred", series_id="DGS10", lag_days=1, revised=False)


def client(seen):
    def respond(request):
        seen.append(request)
        observations = [
            {"date": "2012-01-01", "value": "1.97"},
            {"date": "2012-02-01", "value": "."},
            {"date": "2012-03-01", "value": "2.17"},
        ]
        return httpx.Response(200, text=json.dumps({"observations": observations}))

    return httpx.Client(transport=httpx.MockTransport(respond))


def test_monthly_means_are_read_and_missing_values_dropped(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k" * 32)
    seen = []
    values = SOURCE.fetch(client(seen), date(2012, 1, 1))
    assert values == {date(2012, 1, 1): 1.97, date(2012, 3, 1): 2.17}
    params = seen[0].url.params
    assert params["series_id"] == "DGS10"
    assert params["observation_start"] == "2012-01-01"
    assert (params["frequency"], params["aggregation_method"]) == ("m", "avg")
    assert params["api_key"] == "k" * 32


def test_missing_key_is_named(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        SOURCE.fetch(client([]), date(2012, 1, 1))

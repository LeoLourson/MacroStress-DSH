"""mst.drivers.history: a driver without a value on the date is named as short, with the others."""

from datetime import date

import pytest

from mst.data.store import Observation
from mst.drivers.history import history
from mst.model_sdk import ShortHistory
from tests.drivers import simulated


def test_drivers_without_a_value_on_the_date_are_all_named_as_short(tmp_path):
    panel = simulated.panel(48)
    store = simulated.store(tmp_path / "series", panel)
    late = date(2014, 1, 1)
    for name in ("alpha", "beta"):
        store.write(f"drv.price.{name}", [Observation(late, 1.0, late, False)])

    with pytest.raises(ShortHistory) as raised:
        history(simulated.registry(), "price", panel, store, date(2013, 6, 30))

    assert raised.value.variables == {
        "price:alpha": "series drv.price.alpha has no value known on 2013-06-30",
        "price:beta": "series drv.price.beta has no value known on 2013-06-30",
    }

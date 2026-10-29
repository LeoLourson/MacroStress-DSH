"""mst.macro.panel: the history the macro model is estimated on."""

from datetime import date

import numpy as np
import pytest

from mst.data.store import Observation, SeriesStore
from mst.macro.panel import build_panel, deseasonalize, months_between
from mst.macro.registry import Registry


def registry():
    def variable(id_, transform):
        return {"id": id_, "label": id_, "unit": "u", "transform": transform, "source": "t"}

    return Registry.model_validate(
        {"variables": [variable("ru.cpi", "log100"), variable("gl.vix", "identity")]}
    )


def test_deseasonalize_removes_a_fixed_monthly_pattern_and_keeps_the_trend():
    months = months_between(date(2015, 1, 1), date(2019, 12, 1))
    pattern = np.array([3.0, -1, 0.5, 2, -4, 1, 0, 0, -2, 1.5, 0.5, -1.5])
    pattern -= pattern.mean()
    trend = 0.7 * np.arange(len(months))
    raw = trend + pattern[[m.month - 1 for m in months]]
    raw[7] = np.nan

    adjusted = deseasonalize(raw, months)

    shift = adjusted[0] - trend[0]
    observed = ~np.isnan(raw)
    assert np.allclose(adjusted[observed], trend[observed] + shift)
    assert np.isnan(adjusted[7])


def test_deseasonalize_needs_every_calendar_month():
    months = months_between(date(2015, 1, 1), date(2015, 6, 1))
    with pytest.raises(ValueError, match="calendar month"):
        deseasonalize(np.arange(6.0), months)


def test_panel_orders_global_first_transforms_and_hides_later_releases(tmp_path):
    store = SeriesStore(tmp_path)
    jan, feb, mar = date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)
    store.write(
        "ru.cpi",
        [
            Observation(jan, 100.0, date(2020, 2, 10), True),
            Observation(feb, 101.0, date(2020, 3, 10), True),
            Observation(feb, 102.0, date(2020, 4, 10), False),
        ],
    )
    store.write(
        "gl.vix",
        [
            Observation(jan, 20.0, date(2020, 2, 1), False),
            Observation(mar, 30.0, date(2020, 4, 1), False),
        ],
    )

    panel = build_panel(store, registry(), start=jan, as_of=date(2020, 4, 5))

    assert panel.ids == ["gl.vix", "ru.cpi"]
    assert panel.n_global == 1
    assert panel.months == [jan, feb, mar]
    assert np.isnan(panel.y[1, 0]) and np.isnan(panel.y[2, 1])
    assert panel.y[0, 0] == 20.0
    assert panel.y[1, 1] == pytest.approx(100 * np.log(101.0))
    assert panel.approximate.tolist() == [[False, True], [False, True], [False, False]]


def test_panel_starts_at_start_even_without_data_there(tmp_path):
    store = SeriesStore(tmp_path)
    store.write("gl.vix", [Observation(date(2020, 3, 1), 30.0, date(2020, 4, 1), False)])
    panel = build_panel(store, registry(), start=date(2020, 1, 1), as_of=date(2020, 5, 1))
    assert panel.months[0] == date(2020, 1, 1)
    assert np.isnan(panel.y[:2]).all()

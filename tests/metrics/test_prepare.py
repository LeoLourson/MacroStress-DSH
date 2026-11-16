"""mst.metrics.prepare: the versions of the dates, estimated before the windows open."""

from datetime import date

import pytest

from mst import models as models_pkg
from mst.macro.registry import load_registry
from mst.metrics import prepare as preparing
from mst.metrics import windows

from . import simulated

DATES = [date(2023, 5, 31), date(2023, 8, 31), date(2023, 5, 31)]


@pytest.fixture
def store(tmp_path):
    return simulated.series_store(tmp_path / "series")


class Refusing(simulated.FakeMacro):
    """A model that refuses every date, as the MS-VAR whose stability filter keeps no draw."""

    def estimate(self, context):
        self.calls.append(context.as_of)
        raise ValueError("no draw keeps the companion radius")


def test_a_pool_leaves_the_versions_a_serial_check_would_read(store, tmp_path):
    registry = load_registry()
    pooled, serial = tmp_path / "pooled", tmp_path / "serial"

    preparing.prepare(simulated.FakeMacro(), store, registry, DATES, pooled, workers=2)
    preparing.prepare(simulated.FakeMacro(), store, registry, DATES, serial, workers=1)

    # The windows read the pool's versions from the cache: nothing is estimated again, and the
    # versions are the ones one process makes.
    reader = simulated.FakeMacro()
    for when in set(DATES):
        assert windows.macro_name(reader, store, registry, when, pooled) == windows.macro_name(
            reader, store, registry, when, serial
        )
    assert reader.calls == []


def test_a_refusal_is_kept_for_the_code_that_made_it(store, tmp_path, monkeypatch):
    registry, root, model = load_registry(), tmp_path / "backtest", Refusing()

    preparing.prepare(model, store, registry, DATES, root, workers=1)
    with pytest.raises(ValueError, match="companion radius"):
        windows.macro_name(model, store, registry, DATES[0], root)

    # One estimate per date: the window reads the refusal from the cache.
    assert sorted(model.calls) == sorted(set(DATES))
    # Another code — a fixed platform — estimates the date again.
    monkeypatch.setattr(windows, "code_digest", lambda: "another code")
    with pytest.raises(ValueError, match="companion radius"):
        windows.macro_name(model, store, registry, DATES[0], root)
    assert len(model.calls) == 3


def test_the_satellite_bundles_are_prepared_for_the_windows(store, tmp_path, monkeypatch):
    macro = simulated.FakeMacro()
    simulated.install(monkeypatch, macro)
    simulated.share_price(store)
    catalog = simulated.catalog(tmp_path)
    registry, root = load_registry(), tmp_path / "backtest"
    calls = []
    original = models_pkg.estimate_satellites

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(models_pkg, "estimate_satellites", counted)

    preparing.prepare(macro, store, registry, DATES, root, drivers=catalog, workers=1)
    prepared = len(calls)
    for when in set(DATES):
        name = windows.macro_name(macro, store, registry, when, root)
        windows.satellite_name(name, when, store, root, catalog)

    assert prepared == 2
    assert len(calls) == prepared


def test_the_default_leaves_half_the_processors(monkeypatch):
    monkeypatch.setattr(preparing.os, "cpu_count", lambda: 20)
    assert preparing.default_workers() == 10
    monkeypatch.setattr(preparing.os, "cpu_count", lambda: 1)
    assert preparing.default_workers() == 1

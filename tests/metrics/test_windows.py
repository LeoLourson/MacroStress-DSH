"""mst.metrics.windows: the versions of a date, their caches and the facts of a window."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from mst import models as models_pkg
from mst.data.store import Observation, SeriesStore
from mst.macro.panel import months_between, seasonal_profile
from mst.macro.registry import load_registry
from mst.metrics import windows
from mst.metrics.episodes import load_episodes

from . import simulated

KNOWN_ON = date(2030, 1, 1)
AS_OF = date(2023, 5, 31)


def plan_of(tmp_path):
    file = tmp_path / "episodes.toml"
    file.write_text(
        "\n".join(
            [
                "draws = 40",
                "seed = 3",
                "after_months = 6",
                "",
                "[[episodes]]",
                'id = "R1"',
                'start = "2023-06"',
                'half = "tuning"',
                'source = "manual"',
                'observations = ["gl.brent"]',
                'reason = "test"',
                'sources = ["S1"]',
                "",
                "[[episodes]]",
                'id = "R2"',
                'start = "2024-03"',
                'half = "validation"',
                'source = "manual"',
                'observations = ["ru.key_rate"]',
                'reason = "test"',
                'sources = ["S1"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return load_episodes(file)


@pytest.fixture
def store(tmp_path):
    return simulated.series_store(tmp_path / "series")


def test_a_version_is_estimated_once_for_the_same_data(store, tmp_path):
    macro = simulated.FakeMacro()
    root = tmp_path / "backtest"
    first = windows.estimate_macro(macro, store, load_registry(), AS_OF, root)
    again = windows.estimate_macro(macro, store, load_registry(), AS_OF, root)

    assert macro.calls == [AS_OF]
    assert again.name == first.name
    assert (root / "macro").is_dir()


def test_the_answers_of_the_date_shape_the_version(store, tmp_path):
    from mst import inputs as answers

    class Answered(simulated.FakeMacro):
        def input_keys(self, as_of):
            return ("q1",)

        def estimate(self, context):
            assert answers.read(context.inputs, context.as_of)["q1"]
            return super().estimate(context)

    macro, folder, root = Answered(), tmp_path / "elicitation", tmp_path / "backtest"
    answers.write(folder, AS_OF, "q1", {"value": 1.0})
    windows.estimate_macro(macro, store, load_registry(), AS_OF, root, folder)
    windows.estimate_macro(macro, store, load_registry(), AS_OF, root, folder)
    assert macro.calls == [AS_OF], "the same answers reuse the cached version"
    answers.write(folder, AS_OF, "q1", {"value": 2.0})
    windows.estimate_macro(macro, store, load_registry(), AS_OF, root, folder)
    assert macro.calls == [AS_OF, AS_OF], "an edited answer is estimated again"


def test_the_cache_key_covers_the_model_configuration(store, tmp_path, monkeypatch):
    config = tmp_path / "spec.toml"
    config.write_text("tightness = 0.2\n", encoding="utf-8")
    monkeypatch.setattr(windows, "_configuration_paths", lambda plugin: [config])
    plugin, registry = simulated.FakeMacro(), load_registry()

    first = windows._data_key(plugin, store, registry, AS_OF)
    assert windows._data_key(plugin, store, registry, AS_OF) == first
    config.write_text("tightness = 0.4\n", encoding="utf-8")
    assert windows._data_key(plugin, store, registry, AS_OF) != first


def test_the_configuration_paths_include_the_spec_and_the_signs():
    from mst.plugins.bvar.plugin import MODEL

    names = {path.name for path in windows._configuration_paths(MODEL)}
    assert {"spec.toml", "signs.toml", "registry.toml"} <= names


class ShortRate(simulated.FakeMacro):
    """The fake model with a rate history it finds too short on every date."""

    def estimate(self, context):
        if any(variable.id == "ru.key_rate" for variable in context.registry.variables):
            from mst.model_sdk import ShortHistory

            raise ShortHistory({"ru.key_rate": "2 changes, the model needs 3"})
        assert context.left_out == {"ru.key_rate": "2 changes, the model needs 3"}
        return super().estimate(context)


def test_a_short_variable_is_left_out_of_the_version_with_its_reason(store, tmp_path):
    version = windows.estimate_macro(ShortRate(), store, load_registry(), AS_OF, tmp_path)

    assert version.ids == ["gl.brent"]
    assert version.meta["left_out"] == {"ru.key_rate": "2 changes, the model needs 3"}


def test_a_model_that_calls_a_variable_short_it_was_not_given_is_an_error(store, tmp_path):
    class Stray(simulated.FakeMacro):
        def estimate(self, context):
            from mst.model_sdk import ShortHistory

            raise ShortHistory({"no.such": "never given"})

    with pytest.raises(ValueError, match="not given: no.such"):
        windows.estimate_macro(Stray(), store, load_registry(), AS_OF, tmp_path)


def test_a_short_history_that_names_nothing_is_no_contract():
    from mst.model_sdk import ShortHistory

    with pytest.raises(ValueError, match="names no variable"):
        ShortHistory({})


def test_a_window_runs_from_the_panel_to_the_end_of_the_span(store, tmp_path):
    plan = plan_of(tmp_path)
    window = windows.open_window(
        simulated.FakeMacro(),
        store,
        load_registry(),
        plan,
        id_="2023-06-01",
        start=date(2023, 6, 1),
        half="tuning",
        root=tmp_path / "backtest",
        known_on=KNOWN_ON,
    )
    assert window.as_of == AS_OF
    assert window.months[0] > window.panel.months[-1]
    assert window.months[-1] == date(2023, 12, 1)
    assert window.realized.shape == (window.horizon, len(window.panel.ids))
    assert window.scenario.constraints == () and window.base_shadow is None

    windows.simulate(window, load_registry(), plan)
    assert window.macro_paths.paths.shape[1] == window.horizon + 1
    assert window.base_shadow is None, "the empty scenario is its own base run"


def test_seasonal_facts_use_the_calculation_date_profile(tmp_path):
    from mst.macro.registry import Registry, Variable

    variable = Variable(
        id="ru.cpi", label="cpi", unit="index", transform="log100", source="test", seasonal=True
    )
    registry = Registry(variables=[variable])
    store = SeriesStore(tmp_path / "series")
    months = months_between(date(2019, 1, 1), date(2020, 12, 1))
    factor = np.array([1.0, 1.05, 1.10, 1.15, 1.20, 1.10, 1.0, 0.95, 1.10, 1.05, 1.0, 0.90])
    original = {
        month: float(100.0 * 1.01**t * factor[month.month - 1]) for t, month in enumerate(months)
    }
    revised = {
        date(2019, 6, 1): original[date(2019, 6, 1)] * 1.5,
        date(2019, 7, 1): original[date(2019, 7, 1)] * 0.5,
    }
    rows = [
        Observation(month, value, date(2021, 1, 10), False) for month, value in original.items()
    ]
    rows += [Observation(month, value, date(2022, 6, 1), False) for month, value in revised.items()]
    horizon = months_between(date(2021, 1, 1), date(2021, 2, 1))
    # The facts of the horizon are published after the calculation date, before known_on.
    facts = {
        month: float(100.0 * 1.01 ** (24 + t) * factor[month.month - 1])
        for t, month in enumerate(horizon)
    }
    rows += [
        Observation(month, value, date(2021, 3 + t, 10), False)
        for t, (month, value) in enumerate(facts.items())
    ]
    store.write("ru.cpi", rows)
    panel = SimpleNamespace(ids=["ru.cpi"], months=months)

    result = windows.facts(
        store,
        registry,
        panel,
        horizon,
        estimated_on=date(2021, 1, 15),
        known_on=date(2023, 1, 1),
        plan=SimpleNamespace(exclusions=[]),
    )

    known = {**original, **revised, **facts}
    calendar = [month.month - 1 for month in horizon]
    calculated = seasonal_profile(
        np.array([100 * np.log(original[month]) for month in months]), months
    )
    expected = np.array([100 * np.log(known[month]) for month in horizon]) - calculated[calendar]
    assert np.allclose(result[:, 0], expected)
    # A profile re-estimated from today's revised data would give a different scale.
    today = seasonal_profile(np.array([100 * np.log(known[month]) for month in months]), months)
    alternative = np.array([100 * np.log(known[month]) for month in horizon]) - today[calendar]
    assert not np.allclose(result[:, 0], alternative)


def test_the_satellite_bundle_is_cached_by_its_own_inputs(store, tmp_path, monkeypatch):
    satellite = simulated.install(monkeypatch, simulated.FakeMacro())
    simulated.share_price(store)
    catalog = simulated.catalog(tmp_path)
    plan = plan_of(tmp_path)
    calls = []
    original = models_pkg.estimate_satellites

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(models_pkg, "estimate_satellites", counted)

    def opened():
        window = windows.open_window(
            simulated.FakeMacro(),
            store,
            load_registry(),
            plan,
            id_="R1",
            start=date(2023, 6, 1),
            half="tuning",
            root=tmp_path / "backtest",
            known_on=KNOWN_ON,
        )
        windows.attach_satellites(window, store, tmp_path / "backtest", catalog)
        return window

    first, second = opened(), opened()
    assert len(calls) == 1, "the same inputs reuse the bundle"
    assert first.bundle_name == second.bundle_name
    assert satellite.model_id in first.bundle.plugins


def test_the_satellite_key_covers_the_inputs_that_shape_the_bundle(store, tmp_path, monkeypatch):
    from mst.drivers.registry import load_registry as load_drivers

    satellite = simulated.install(monkeypatch, simulated.FakeMacro())
    catalog = simulated.catalog(tmp_path)
    simulated.share_price(store)
    series = store.as_of("drv.share_price.sber", KNOWN_ON)
    registry = load_drivers(catalog=catalog)
    installed = {satellite.model_id: satellite}
    window = SimpleNamespace(macro=SimpleNamespace(name="macro-1"), as_of=date(2024, 1, 31))

    def key(of=window):
        return windows._satellite_key(of.macro.name, of.as_of, store, catalog, installed, registry)

    baseline = key()
    assert key() == baseline
    # Another macro version, another catalog, another series, another plugin version or
    # another assignment is another key — a stale bundle is never reused.
    assert key(SimpleNamespace(macro=SimpleNamespace(name="macro-2"), as_of=window.as_of)) != (
        baseline
    )
    catalog.write_bytes(catalog.read_bytes() + b" ")
    assert key() != baseline
    catalog.write_bytes(catalog.read_bytes()[:-1])
    assert key() == baseline
    store.write(
        "drv.share_price.sber",
        [Observation(row.month, row.value + 1.0, row.published, row.approximate) for row in series],
    )
    assert key() != baseline
    store.write("drv.share_price.sber", series)
    assert key() == baseline
    with monkeypatch.context() as patch:
        patch.setattr(type(satellite), "version", "0.2.0")
        assert key() != baseline
    with monkeypatch.context() as patch:
        patch.setattr(
            models_pkg,
            "assignment",
            lambda quantities, overrides=None: ({"share_price": "other.model"}, []),
        )
        assert key() != baseline


def test_the_plan_hash_does_not_depend_on_where_the_file_lies(tmp_path):
    old = tmp_path / "src" / "episodes.toml"
    new = tmp_path / "bench" / "episodes.toml"
    for file in (old, new):
        file.parent.mkdir()
        file.write_text("draws = 40\nseed = 3\n", encoding="utf-8")
    assert windows.file_digest(old) == windows.file_digest(new) is not None


def test_a_short_driver_is_left_out_of_the_bundle_not_the_window(store, tmp_path, monkeypatch):
    from mst.model_sdk import ShortHistory

    class ShortDriver(simulated.FakeSatellite):
        """The fake satellite, short of the history of a second driver on every date."""

        def estimate(self, context):
            if any(driver.id == "share_price:late" for driver in context.registry.drivers):
                raise ShortHistory({"share_price:late": "no value known on the date"})
            return super().estimate(context)

    simulated.install(monkeypatch, simulated.FakeMacro(), ShortDriver())
    simulated.share_price(store)
    catalog = simulated.catalog(tmp_path, late=True)
    window = windows.open_window(
        simulated.FakeMacro(),
        store,
        load_registry(),
        plan_of(tmp_path),
        id_="R1",
        start=date(2023, 6, 1),
        half="tuning",
        root=tmp_path / "backtest",
        known_on=KNOWN_ON,
    )
    windows.attach_satellites(window, store, tmp_path / "backtest", catalog)

    assert window.bundle.left_out == {"share_price:late": "no value known on the date"}
    assert window.left_out == {"share_price:late": "no value known on the date"}
    assert window.bundle.served() == ["share_price:sber"]

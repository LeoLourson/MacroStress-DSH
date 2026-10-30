"""mst.macro.version and ``mst.models.estimate_macro``: a version is named by its content."""

from datetime import date

import numpy as np
import pytest

from mst import models as models_pkg
from mst import versions
from mst.data.store import Observation, SeriesStore
from mst.macro.panel import months_between
from mst.macro.registry import Registry
from mst.macro.version import load, read, save
from mst.plugins.bvar import MODEL
from mst.plugins.bvar.estimate import estimate
from mst.plugins.bvar.identify import identify
from mst.plugins.bvar.model import parts
from mst.plugins.bvar.signs import Shock, Signs
from mst.plugins.bvar.spec import load_spec
from tests.bvar import simulated as sim

MACRO = {"bvar.sv": MODEL}


@pytest.fixture(scope="module")
def estimated():
    y = sim.simulate(80, np.random.default_rng(0))
    panel = sim.panel(y)
    spec = sim.spec(draws=10)
    shocks = [Shock(id="ru.t", label="t", rises=["ru.b"])]
    model = identify(estimate(panel, spec), shocks, sign_lag=0, tries=100, seed=1)
    return model, spec, panel


def test_a_saved_version_loads_back_unchanged(tmp_path, estimated):
    model, spec, panel = estimated
    estimate_ = parts(model, spec, panel, date(2010, 1, 1))
    name = save("bvar.sv", estimate_.meta, estimate_.arrays, tmp_path)

    version = load(tmp_path, name, MACRO)

    assert len(name) == 64 and version.meta["as_of"] == "2010-01-01"
    assert version.model.shocks == model.shocks and version.model.months == model.months
    assert np.array_equal(version.model.rotation, model.rotation)
    assert np.array_equal(version.model.dof, model.dof)
    again = parts(version.model, spec, panel, date(2010, 1, 1))
    assert save("bvar.sv", again.meta, again.arrays, tmp_path) == name


def test_a_version_names_the_model_that_estimated_it(tmp_path, estimated):
    model, spec, panel = estimated
    estimate_ = parts(model, spec, panel, date(2010, 1, 1))
    name = save("bvar.sv", estimate_.meta, estimate_.arrays, tmp_path)
    assert load(tmp_path, name, MACRO).plugin is MODEL
    with pytest.raises(KeyError, match="which is not installed"):
        load(tmp_path, name, {})


def test_a_legacy_version_loads_with_the_only_installed_model(tmp_path, estimated):
    model, spec, panel = estimated
    estimate_ = parts(model, spec, panel, date(2010, 1, 1))
    # Pre-plugin version: the meta has no "model" and the spec still has "seasonal".
    meta = dict(estimate_.meta, spec={**estimate_.meta["spec"], "seasonal": ["ru.b"]})
    name = versions.save(tmp_path / "macro", meta, estimate_.arrays)

    version = load(tmp_path, name, MACRO)

    assert version.plugin is MODEL
    assert np.array_equal(version.model.rotation, model.rotation)
    assert np.array_equal(version.model.dof, model.dof)


def test_a_legacy_version_with_several_models_is_refused_by_name(tmp_path, estimated):
    model, spec, panel = estimated
    estimate_ = parts(model, spec, panel, date(2010, 1, 1))
    name = versions.save(tmp_path / "macro", estimate_.meta, estimate_.arrays)

    with pytest.raises(ValueError, match=f"macro version {name} does not name the model"):
        load(tmp_path, name, {"bvar.sv": MODEL, "other.model": MODEL})


def test_a_version_whose_spec_has_a_removed_key_still_loads(tmp_path, estimated):
    model, spec, panel = estimated
    estimate_ = parts(model, spec, panel, date(2010, 1, 1))
    meta = dict(estimate_.meta, spec={**estimate_.meta["spec"], "seasonal": ["ru.b"]})
    name = save("bvar.sv", meta, estimate_.arrays, tmp_path)

    version = load(tmp_path, name, MACRO)

    assert version.meta["model"] == "bvar.sv"
    assert np.array_equal(version.model.lags, model.lags)


def test_a_changed_version_is_refused(tmp_path, estimated):
    model, spec, panel = estimated
    estimate_ = parts(model, spec, panel, date(2010, 1, 1))
    name = save("bvar.sv", estimate_.meta, estimate_.arrays, tmp_path)
    meta = tmp_path / "macro" / name / "meta.json"
    meta.write_text(meta.read_text(encoding="utf-8").replace("2010-01-01", "2010-01-02"))
    with pytest.raises(ValueError, match="does not match"):
        read(tmp_path, name)


def test_a_version_without_the_required_keys_is_refused(tmp_path):
    full = {"as_of": "2010-01-01", "ids": [], "months": [], "shocks": [], "n_global": 0}
    with pytest.raises(ValueError, match="a macro version must record"):
        save("bvar.sv", {}, {"panel_y": np.zeros((2, 2))}, tmp_path)
    with pytest.raises(ValueError, match="a macro version must carry"):
        save("bvar.sv", full, {}, tmp_path)


def test_the_panel_of_a_version_is_rebuilt(tmp_path, estimated):
    model, spec, panel = estimated
    estimate_ = parts(model, spec, panel, date(2010, 1, 1))
    name = save("bvar.sv", estimate_.meta, estimate_.arrays, tmp_path)
    from mst.macro.version import panel_of

    rebuilt, as_of = panel_of(tmp_path, name)
    assert rebuilt.ids == panel.ids and rebuilt.months == panel.months
    assert np.array_equal(rebuilt.y, panel.y)
    assert as_of == date(2010, 1, 1)


def test_estimate_macro_publishes_a_version_from_the_store(tmp_path, monkeypatch):
    ids = ["gl.a", "ru.b", "ru.c"]
    registry = Registry.model_validate(
        {
            "variables": [
                {"id": i, "label": i, "unit": "u", "transform": "identity", "source": "t"}
                for i in ids
            ]
        }
    )
    spec = load_spec().model_copy(
        update={
            "start": sim.START,
            "stationary": ids,
            "dummies": [],
            "volatility": "constant",
            "sampler": sim.spec(draws=5).sampler,
        }
    )
    table = Signs.model_validate({"shocks": [Shock(id="ru.t", label="t", rises=["ru.b"])]})
    y = sim.simulate(60, np.random.default_rng(3))
    store = SeriesStore(tmp_path / "series")
    months = months_between(sim.START, date(2004, 12, 1))
    for k, id_ in enumerate(ids):
        store.write(id_, [Observation(m, float(y[t, k]), m, False) for t, m in enumerate(months)])

    from mst.plugins.bvar import model as bvar_model

    monkeypatch.setattr(bvar_model, "load_spec", lambda: spec)
    monkeypatch.setattr(bvar_model, "load_signs", lambda: table)
    monkeypatch.setattr(models_pkg, "load_macro_registry", lambda: registry)

    name = models_pkg.estimate_macro(
        None, store=tmp_path / "series", models=tmp_path / "models", as_of=date(2005, 1, 1)
    )

    version = load(tmp_path / "models", name, MACRO)
    assert version.meta["months"][-1] == "2004-12-01"
    assert version.model.shocks[1] == "ru.t"
    assert version.plugin is MODEL

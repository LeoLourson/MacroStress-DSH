"""mst.macro.version and ``msa-macro estimate``: a model version is named by its content."""

import json
from datetime import date

import numpy as np
import pytest

from mst.data.store import Observation, SeriesStore
from mst.macro import cli
from mst.macro.estimate import estimate
from mst.macro.identify import identify
from mst.macro.panel import months_between
from mst.macro.registry import Registry, Shock
from mst.macro.spec import load_spec
from mst.macro.version import load, save

from . import simulated as sim


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
    name = save(model, spec, panel, date(2010, 1, 1), tmp_path)

    loaded, meta = load(tmp_path, name)

    assert len(name) == 64 and meta["as_of"] == "2010-01-01"
    assert loaded.shocks == model.shocks and loaded.months == model.months
    assert np.array_equal(loaded.rotation, model.rotation)
    assert np.array_equal(loaded.dof, model.dof)
    assert save(loaded, spec, panel, date(2010, 1, 1), tmp_path) == name


def test_a_changed_version_is_refused(tmp_path, estimated):
    model, spec, panel = estimated
    name = save(model, spec, panel, date(2010, 1, 1), tmp_path)
    meta = tmp_path / "macro" / name / "meta.json"
    meta.write_text(meta.read_text(encoding="utf-8").replace("2010-01-01", "2010-01-02"))
    with pytest.raises(ValueError, match="does not match"):
        load(tmp_path, name)


def test_estimate_command_publishes_a_version_from_the_store(tmp_path, monkeypatch, capsys):
    ids = ["gl.a", "ru.b", "ru.c"]
    registry = Registry.model_validate(
        {
            "variables": [
                {"id": i, "label": i, "unit": "u", "transform": "identity", "source": "t"}
                for i in ids
            ],
            "shocks": [{"id": "ru.t", "label": "t", "rises": ["ru.b"]}],
        }
    )
    spec = load_spec().model_copy(
        update={
            "start": sim.START,
            "stationary": ids,
            "seasonal": [],
            "dummies": [],
            "volatility": "constant",
            "sampler": sim.spec(draws=5).sampler,
        }
    )
    y = sim.simulate(60, np.random.default_rng(3))
    store = SeriesStore(tmp_path / "series")
    months = months_between(sim.START, date(2004, 12, 1))
    for k, id_ in enumerate(ids):
        store.write(id_, [Observation(m, float(y[t, k]), m, False) for t, m in enumerate(months)])
    monkeypatch.setattr(cli, "load_registry", lambda: registry)
    monkeypatch.setattr(cli, "load_spec", lambda: spec)

    code = cli.main(
        [
            "estimate",
            "--as-of",
            "2005-01-01",
            "--store",
            str(tmp_path / "series"),
            "--models",
            str(tmp_path / "models"),
        ]
    )

    name, diagnostics = capsys.readouterr().out.splitlines()
    assert code == 0
    assert json.loads(diagnostics)["estimated"] == 5
    model, meta = load(tmp_path / "models", name)
    assert meta["months"][-1] == "2004-12-01" and model.shocks[1] == "ru.t"

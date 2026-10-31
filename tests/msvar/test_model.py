"""The plugin adapter: a version round-trips through parts, the store and load."""

from datetime import date

import numpy as np
import pytest

from mst.macro import version as macro_version
from mst.model_sdk.contracts import LinearConstraint, LinearTerm, ScenarioPlan
from mst.plugins.msvar import estimate, identify
from mst.plugins.msvar.model import MSVarModel, parts
from mst.plugins.msvar.signs import Signs
from tests.msvar import simulated as sim

SIGNS = Signs.model_validate(
    {
        "shocks": [
            {"id": "gl.s", "label": "g", "rises": ["gl.a"]},
            {"id": "ru.t", "label": "t", "rises": ["ru.b", "ru.c"]},
        ],
        "regimes": [
            {"id": "regime.calm", "label": "спокойный"},
            {"id": "regime.stress", "label": "стрессовый"},
        ],
    }
)


def model():
    y, _ = sim.simulate(120, np.random.default_rng(4))
    posterior = estimate(sim.panel(y), sim.spec(draws=6, burn=6), 2)
    return identify(posterior, SIGNS, sign_lag=2, tries=2000, seed=3), sim.panel(y)


def plan() -> ScenarioPlan:
    return ScenarioPlan(
        horizon=3,
        constraints=(
            LinearConstraint(
                terms=(LinearTerm(variable="ru.c", month=2, weight=1.0),),
                low=5.0,
                high=6.0,
            ),
        ),
        driving_shocks=("ru.t",),
    )


def test_a_version_round_trips_through_parts_and_load():
    identified, panel = model()
    estimate_version = parts(identified, sim.spec(draws=6, burn=6), panel, date(2009, 12, 1))

    # The J set the platform checks a scenario against: shocks and regime elements.
    assert estimate_version.meta["shocks"] == ["gl.s", "ru.t", "ru.unnamed_1"] + SIGNS.regime_ids
    assert estimate_version.meta["regimes"] == SIGNS.regime_ids

    loaded = MSVarModel().load(estimate_version.meta, estimate_version.arrays)
    assert loaded.shocks == identified.shocks
    assert loaded.regime_ids == identified.regime_ids
    assert loaded.n_regimes == identified.n_regimes

    chosen = plan()
    a = MSVarModel().simulate(loaded, chosen, draws=8, seed=7)
    b = MSVarModel().simulate(identified, chosen, draws=8, seed=7)
    assert np.array_equal(a.paths, b.paths)
    assert np.all((a.paths[:, 2, 2] >= 5.0) & (a.paths[:, 2, 2] <= 6.0))


def test_the_meta_passes_the_version_checks(tmp_path):
    identified, panel = model()
    estimate_version = parts(identified, sim.spec(draws=6, burn=6), panel, date(2009, 12, 1))

    name = macro_version.save(
        MSVarModel.model_id, estimate_version.meta, estimate_version.arrays, tmp_path
    )
    version = macro_version.load(tmp_path, name, {MSVarModel.model_id: MSVarModel()})
    assert version.name == name
    assert version.meta["regimes"] == SIGNS.regime_ids
    run = version.plugin.simulate(version.model, plan(), draws=4, seed=7)
    assert run.paths.shape == (4, 4, 3)

    described = macro_version.describe(tmp_path, name)
    assert described["model"] == MSVarModel.model_id
    assert described["shocks"] == estimate_version.meta["shocks"]
    assert described["identification_share"] == pytest.approx(
        identified.diagnostics["identified"] / identified.diagnostics["estimated"]
    )

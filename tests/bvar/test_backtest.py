"""The episode check of the BVAR plugin on historical episodes."""

from datetime import date

import numpy as np

from mst.data.store import Observation, SeriesStore
from mst.macro import version as macro_version
from mst.macro.panel import months_between
from mst.macro.registry import Registry, block_of, load_registry
from mst.plugins.bvar.backtest import (
    Case,
    Exclusion,
    calculation_date,
    coverage,
    explosive_share,
    history_spec,
    load_backtest,
    models_for,
    realized_paths,
)
from mst.plugins.bvar.estimate import Posterior
from mst.plugins.bvar.forecast import Simulation
from mst.plugins.bvar.signs import Shock, load_signs
from mst.plugins.bvar.spec import load_spec
from tests.bvar import simulated as sim


def test_every_episode_of_backtest_toml_fits_the_registry():
    plan, registry, signs = load_backtest(), load_registry(), load_signs()
    variables = {v.id for v in registry.variables}
    shocks = set(signs.ids)
    for check in plan.episodes:
        assert set(check.constrained) <= variables, check.id
        assert set(check.driving) <= shocks, check.id
        assert len(check.driving) >= len(check.constrained), check.id
        globals_ = [v for v in check.constrained if block_of(v) == "gl"]
        assert len([s for s in check.driving if block_of(s) == "gl"]) >= len(globals_), check.id


def test_the_calculation_date_is_the_day_before_the_episode():
    assert calculation_date(date(2015, 3, 1)) == date(2015, 2, 28)


def test_dummies_after_the_history_are_left_out():
    spec = load_spec()
    months = [d.month for d in history_spec(spec, date(2021, 6, 1)).dummies]
    assert months == [date(2014, 11, 1), date(2021, 1, 1)]


def registry():
    def variable(id_, transform):
        return {"id": id_, "label": id_, "unit": "u", "transform": transform, "source": "t"}

    return Registry.model_validate(
        {"variables": [variable("gl.a", "log100"), variable("ru.b", "identity")]}
    )


def test_realized_paths_are_in_the_model_scale_and_respect_exclusions(tmp_path):
    store = SeriesStore(tmp_path)
    months = months_between(date(2020, 11, 1), date(2021, 2, 1))
    for id_, values in {"gl.a": [10.0, 20.0, 40.0, 80.0], "ru.b": [1.0, 2.0, 3.0, 4.0]}.items():
        store.write(
            id_,
            [
                Observation(m, v, date(2026, 1, 1), False)
                for m, v in zip(months, values, strict=True)
            ],
        )
    profile = np.arange(12.0)
    excluded = Exclusion(variables=["ru.b"], start=date(2021, 1, 1), reason="splice")

    y = realized_paths(
        store,
        registry(),
        ids=["gl.a", "ru.b"],
        months=months,
        history_end=date(2020, 10, 1),
        known_on=date(2026, 10, 29),
        profiles={"gl.a": profile},
        exclusions=[excluded],
    )
    assert np.allclose(y[:, 0], 100 * np.log([10, 20, 40, 80]) - profile[[10, 11, 0, 1]])
    assert np.allclose(y[:2, 1], [1, 2]) and np.isnan(y[2:, 1]).all()

    after = realized_paths(
        store,
        registry(),
        ids=["ru.b"],
        months=months,
        history_end=date(2021, 1, 1),
        known_on=date(2026, 10, 29),
        profiles={},
        exclusions=[excluded],
    )
    assert np.allclose(after[:, 0], [1, 2, 3, 4])

    unpublished = realized_paths(
        store,
        registry(),
        ids=["ru.b"],
        months=months,
        history_end=date(2020, 10, 1),
        known_on=date(2025, 12, 31),
        profiles={},
        exclusions=[],
    )
    assert np.isnan(unpublished).all()


def test_coverage_counts_unconstrained_known_months_inside_the_band():
    months = months_between(date(2020, 1, 1), date(2020, 3, 1))
    # 101 paths of two variables: path r is r - 50 in every month after t = 0.
    paths = np.zeros((101, 4, 2))
    paths[:, 1:, :] = (np.arange(101) - 50.0)[:, None, None]
    simulation = Simulation(ids=["gl.a", "ru.b"], months=[date(2019, 12, 1), *months], paths=paths)
    realized = np.array([[0.0, 0.0], [100.0, 44.0], [np.nan, -46.0]])
    case = Case("X", date(2019, 12, 31), months, realized, ["gl.a"], scenario=None)

    result = coverage(case, simulation)

    assert result.counts == {"ru.b": 3}
    assert result.hits == {"ru.b": 2}
    assert result.share == 2 / 3
    assert np.allclose(result.quantiles[:, 0, 1], [-45, 0, 45])


def test_explosive_share_counts_draws_with_a_root_outside_the_unit_circle():
    # Two draws of a VAR(2) in two variables: y_t = 0.5 y_{t-1} + 0.3 y_{t-2} is stable,
    # y_t = 0.9 y_{t-1} + 0.3 y_{t-2} has a root of modulus 1.17.
    lags = np.zeros((2, 2, 2, 2))
    lags[0, 0], lags[0, 1] = 0.5 * np.eye(2), 0.3 * np.eye(2)
    lags[1, 0], lags[1, 1] = 0.9 * np.eye(2), 0.3 * np.eye(2)
    zeros = np.zeros((2, 2))
    model = Posterior(
        ids=["gl.a", "ru.b"],
        n_global=1,
        months=[date(2020, 1, 1)],
        lags_count=2,
        dummies=[],
        const=zeros,
        lags=lags,
        dummy=np.zeros((2, 2, 0)),
        chol=np.array([np.eye(2)] * 2),
        logvar=np.zeros((2, 1, 2)),
        volvol=zeros,
        dof=np.full((2, 2), np.inf),
        history=np.zeros((2, 2, 2)),
    )
    assert explosive_share(model) == 0.5


def test_episode_models_stay_out_of_the_versions_of_the_workspace(tmp_path):
    panel = sim.panel(sim.simulate(80, np.random.default_rng(0)))
    spec = sim.spec(draws=10)
    tables = {"registry": [Shock(id="ru.t", label="t", rises=["ru.b"])]}
    models = tmp_path / "models"

    from mst.plugins.bvar import MODEL

    first = models_for(MODEL, panel, spec, date(2010, 1, 1), tables, models)
    again = models_for(MODEL, panel, spec, date(2010, 1, 1), tables, models)

    assert macro_version.catalogue(models) == []
    assert again["registry"][0] == first["registry"][0]


def test_an_episode_missing_from_the_manifest_is_reported_not_raised(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"corpus_id": "c", "version": "1", "updated": "2026-10-29", "description": "",'
        ' "vocabularies": {"factors": {}, "flags": {}}, "sources": [], "data_series": [],'
        ' "episodes": []}',
        encoding="utf-8",
    )
    from mst.plugins.bvar import MODEL

    code = MODEL.backtest(
        store=tmp_path / "s",
        models=tmp_path / "m",
        corpus=manifest,
        as_of=date(2026, 10, 29),
        only=["R13"],
    )

    assert code == 1
    assert "R13: not in the corpus manifest" in capsys.readouterr().out

"""The structural form of the BVAR: the intervention inside the VAR recursion."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from mst import versions
from mst.macro.panel import Panel
from mst.macro.version import describe, load, read, save
from mst.model_sdk.contracts import Intervention, LinearConstraint, LinearTerm, ScenarioPlan
from mst.model_sdk.interventions import apply_default
from mst.model_sdk.testing import check_interventions
from mst.plugins.bvar.estimate import Posterior
from mst.plugins.bvar.forecast import constraint_law, simulate
from mst.plugins.bvar.model import BvarModel, parts
from tests.bvar import simulated as sim
from tests.bvar.test_bounds import bounded

REPOSITORY = Path(__file__).resolve().parents[2]
MONTHS = [date(2020 + m // 12, m % 12 + 1, 1) for m in range(13)]
A = np.array([[0.5, 0.1], [0.2, 0.4]])
CONST = np.array([[0.1, 0.2]])
START = np.array([1.0, 5.0])
FORM = {"time_recursion": True, "order_within_month": ["ru.a", "ru.key_rate"]}


def var_model(*, shocks: bool = True, structural: dict | None = None) -> Posterior:
    """A two-variable VAR(1) with unit errors when ``shocks``, none otherwise."""
    chol = np.eye(2)[None] if shocks else np.zeros((1, 2, 2))
    return Posterior(
        ids=["ru.a", "ru.key_rate"],
        n_global=0,
        months=MONTHS,
        lags_count=1,
        dummies=[],
        const=CONST.copy(),
        lags=A[None, None],
        dummy=np.zeros((1, 2, 0)),
        chol=chol,
        logvar=np.zeros((1, 12, 2)),
        volvol=np.zeros((1, 2)),
        dof=np.full((1, 2), np.inf),
        history=START[None, None],
        rotation=np.eye(2)[None],
        shocks=["ru.a", "ru.key_rate"],
        structural=structural,
    )


def plan(month: int, value: float, *, kind: str = "set", draws: int = 1, horizon: int = 6):
    return ScenarioPlan(
        horizon=horizon,
        interventions=(
            Intervention("ru.key_rate", (month,), kind, np.full((draws, 1), value)),
        ),
    )


def hand_recursion(month: int, value: float, horizon: int):
    """The zero-shock VAR recursion with the rate set in ``month``, computed by hand."""
    path, y = [], START.copy()
    for t in range(1, horizon + 1):
        y = CONST[0] + A @ y
        if t == month:
            y = y.copy()
            y[1] = value
        path.append(y.copy())
    return np.array(path)


def test_zero_shocks_reproduce_the_hand_recursion():
    # With no shocks the path is the VAR recursion itself: the rate is replaced in its month,
    # the other equation of that month does not see it, and the next month reads it through A.
    run = simulate(
        var_model(shocks=False, structural=FORM), plan(2, 9.0), draws=1, seed=1, candidates=2
    )
    expected = hand_recursion(2, 9.0, 6)
    assert np.allclose(run.paths[0, 1:], expected, atol=1e-12, rtol=0)
    # Month two's other variable is the unshocked recursion, not moved by the set.
    assert np.isclose(run.paths[0, 2, 0], expected[1, 0])


def test_default_form_reverts_after_the_set_month_and_recursion_does_not():
    chosen = plan(2, 9.0)
    base = simulate(var_model(), ScenarioPlan(horizon=6), draws=1, seed=3)
    default = simulate(var_model(), chosen, draws=1, seed=3)
    final = apply_default(default.paths, default.ids, chosen.interventions)
    recursive = simulate(var_model(structural=FORM), chosen, draws=1, seed=3)
    assert np.isclose(final[0, 2, 1], 9.0)
    # The default form sets the cell after drawing: next month is the base path again.
    assert np.array_equal(final[0, 3:], base.paths[0, 3:])
    # The structural form carries the imposed value into the next month through the lag.
    assert not np.array_equal(recursive.paths[0, 3:], base.paths[0, 3:])
    assert np.isclose(recursive.paths[0, 2, 1], 9.0)


def test_shift_keeps_its_equation_and_the_same_quantity_can_be_set_twice():
    # A shift adds to the equation, so the drawn shock stays; two sets in two months each
    # replace their own month and the recursion reads both through the lags.
    base = simulate(var_model(structural=FORM), ScenarioPlan(horizon=6), draws=2, seed=4)
    shifted = simulate(
        var_model(structural=FORM), plan(2, 2.0, kind="shift", draws=2), draws=2, seed=4
    )
    assert np.allclose(shifted.paths[:, 2, 1] - base.paths[:, 2, 1], 2.0, atol=1e-12)
    assert not np.array_equal(shifted.paths[:, 3:], base.paths[:, 3:])
    two = ScenarioPlan(
        horizon=6,
        interventions=(
            Intervention("ru.key_rate", (1, 3), "set", np.full((1, 2), [8.0, 10.0])),
        ),
    )
    run = simulate(
        var_model(shocks=False, structural=FORM), two, draws=1, seed=5, candidates=1
    )
    assert np.isclose(run.paths[0, 1, 1], 8.0)
    assert np.isclose(run.paths[0, 3, 1], 10.0)


def test_recursive_law_fixes_the_set_coordinate_and_keeps_the_original_rows():
    observation = LinearConstraint(
        (LinearTerm("ru.a", 2, 1.0),), -np.inf, np.inf
    )
    chosen = ScenarioPlan(
        horizon=3,
        constraints=(observation,),
        interventions=(Intervention("ru.key_rate", (1,), "set", np.full((2, 1), 9.0)),),
    )
    law = constraint_law(var_model(structural=FORM), chosen, draws=2, seed=1, candidates=2)
    assert law.means.shape == (4, 1) and law.covs.shape == (4, 1, 1)
    assert np.array_equal(law.group, [0, 0, 1, 1])
    # The observation is on ru.a in month 2: it reads the set of month 1 through the lag.
    without = constraint_law(
        var_model(structural=FORM),
        replace(chosen, interventions=()),
        draws=2,
        seed=1,
        candidates=2,
    )
    assert not np.allclose(law.means[:, 0], without.means[:, 0])


def test_recursive_law_keeps_a_set_row_deterministic():
    row = LinearConstraint((LinearTerm("ru.key_rate", 2, 1.0),), 8.8, 9.2)
    chosen = ScenarioPlan(
        horizon=3,
        constraints=(row,),
        interventions=(Intervention("ru.key_rate", (2,), "set", np.full((2, 1), 9.0)),),
    )
    law = constraint_law(var_model(structural=FORM), chosen, draws=2, seed=1, candidates=2)
    assert np.allclose(law.means[:, 0], 9.0)
    assert np.allclose(law.covs[:, 0, 0], 0.0)


def test_a_false_set_observation_is_refused_with_its_original_number():
    from mst.model_sdk.dist import ContradictsModels

    outside = LinearConstraint((LinearTerm("ru.key_rate", 2, 1.0),), 10.0, 11.0)
    chosen = ScenarioPlan(
        horizon=3,
        constraints=(outside,),
        interventions=(Intervention("ru.key_rate", (2,), "set", np.full((2, 1), 9.0)),),
    )
    with pytest.raises(ContradictsModels) as caught:
        simulate(var_model(structural=FORM), chosen, draws=2, seed=1, candidates=1)
    assert caught.value.constraints == (1,)
    assert caught.value.group == 0


def test_check_interventions_passes_for_the_recursive_form():
    model = replace(bounded(0.0), structural={"time_recursion": True,
                                              "order_within_month": ["ru.key_rate"]})
    assert BvarModel().structural_form(model) is not None
    assert check_interventions(BvarModel(), model, horizon=3, draws=2, inner=2) == []


def test_default_parts_omit_the_form_and_a_recursive_version_loads_it(tmp_path):
    spec = sim.spec()
    panel = Panel(
        ids=["ru.key_rate"],
        n_global=0,
        months=MONTHS,
        y=np.zeros((len(MONTHS), 1)),
        approximate=np.zeros((len(MONTHS), 1), dtype=bool),
    )
    default = parts(bounded(0.0), spec, panel, date(2010, 1, 1))
    assert "structural_form" not in default.meta
    recursive = parts(
        replace(bounded(0.0), structural=FORM), spec, panel, date(2010, 1, 1)
    )
    assert recursive.meta["structural_form"] == FORM
    name = save("bvar.sv", recursive.meta, recursive.arrays, tmp_path / "models")
    from mst import plugins

    version = load(tmp_path / "models", name, plugins.registry().macro)
    form = BvarModel().structural_form(version.model)
    assert form is not None and form.time_recursion
    assert form.order_within_month == ("ru.a", "ru.key_rate")
    assert describe(tmp_path / "models", name)["structural_form"] == FORM


@pytest.mark.skipif(
    not (REPOSITORY / "workspace/models/macro"
         / "51ead9204774f7ef0b4657c4e1a3997c77c584a902ebb6d306c68439977658f8").is_dir(),
    reason="the stored golden version is absent",
)
def test_the_stored_default_version_keeps_its_hash_and_has_no_form():
    name = "51ead9204774f7ef0b4657c4e1a3997c77c584a902ebb6d306c68439977658f8"
    root = REPOSITORY / "workspace/models"
    meta, arrays = read(root, name)
    assert "structural_form" not in meta
    assert versions.digest(meta, arrays) == name


def test_the_estimate_switch_reaches_the_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from mst import models

    class Switched:
        model_id = "bvar.sv"

        def estimate(self, context):
            raise AssertionError("the default estimate must not run when a form is asked")

        def estimate_with_form(self, context, *, form):
            return SimpleNamespace(meta={"form": form}, arrays={})

    class Plain:
        model_id = "plain"

        def estimate(self, context):
            return SimpleNamespace(meta={}, arrays={})

    monkeypatch.setattr(models, "SeriesStore", lambda path: None)
    monkeypatch.setattr(models, "load_macro_registry", lambda: None)
    saved = {}
    monkeypatch.setattr(
        models.macro_version,
        "save",
        lambda model_id, meta, arrays, root: saved.update(meta=meta) or "version",
    )
    monkeypatch.setattr(models, "selected", lambda model_id: Switched())
    name = models.estimate_macro(
        "bvar.sv", store=tmp_path, models=tmp_path, as_of=date(2026, 10, 29),
        form="recursive",
    )
    assert name == "version" and saved["meta"] == {"form": "recursive"}
    with pytest.raises(ValueError, match="unknown structural form"):
        models.estimate_macro(
            "bvar.sv", store=tmp_path, models=tmp_path, as_of=date(2026, 10, 29),
            form="nope",
        )
    monkeypatch.setattr(models, "selected", lambda model_id: Plain())
    with pytest.raises(ValueError, match="offers no structural form switch"):
        models.estimate_macro(
            "bvar.sv", store=tmp_path, models=tmp_path, as_of=date(2026, 10, 29),
            form="recursive",
        )


def test_a_fresh_estimation_stores_the_recursive_form(monkeypatch):
    from types import SimpleNamespace

    from mst.plugins.bvar import model as bvar_model
    from mst.plugins.bvar.estimate import estimate as real_estimate
    from mst.plugins.bvar.identify import identify as real_identify
    from mst.plugins.bvar.signs import Shock

    panel = sim.panel(sim.simulate(80, np.random.default_rng(0)))
    spec = sim.spec(draws=10, burn=0)
    shock = Shock(id="ru.t", label="t", rises=["ru.b"])
    monkeypatch.setattr(bvar_model, "load_spec", lambda: spec)
    monkeypatch.setattr(bvar_model, "build_panel", lambda *args, **kwargs: panel)
    monkeypatch.setattr(bvar_model, "estimate", real_estimate)
    monkeypatch.setattr(
        bvar_model,
        "identify",
        lambda posterior, shocks, **kwargs: real_identify(
            posterior, [shock], sign_lag=0, tries=100, seed=1
        ),
    )
    context = SimpleNamespace(
        store=None, registry=SimpleNamespace(variables=[]), as_of=date(2010, 1, 1)
    )
    default = bvar_model.BvarModel().estimate(context)
    recursive = bvar_model.BvarModel().estimate_with_form(context, form="recursive")
    assert "structural_form" not in default.meta
    assert recursive.meta["structural_form"]["time_recursion"] is True
    assert recursive.meta["structural_form"]["order_within_month"] == list(panel.ids)


def test_the_model_tool_passes_the_form_flag(tmp_path):
    from mst import store
    from mst.mcp import _macro_argv

    ws = store.Workspace(tmp_path)
    with_form = _macro_argv(
        ws, "estimate-macro", "2026-10-29", "bvar.sv", form="recursive"
    )
    assert with_form[-2:] == ["--form", "recursive"]
    plain = _macro_argv(ws, "estimate-macro", "2026-10-29", "bvar.sv")
    assert "--form" not in plain

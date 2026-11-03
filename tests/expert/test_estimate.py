"""The estimate: from the answers of the date and the history to a model version."""

import numpy as np
import pytest

from mst.macro import version as macro_version
from mst.plugins.expert.estimate import (
    ResidualNotAdmitted,
    estimate,
    from_history,
)
from mst.plugins.expert.model import MARKET
from tests.expert import fakes


@pytest.fixture(scope="module")
def spec():
    return fakes.quick_spec()


def test_with_every_frequency_zero_the_model_is_the_benchmark(tmp_path, spec):
    ctx = fakes.answers_context(
        tmp_path,
        changes={key: (lambda payload: {**payload, "frequency": 0.0}) for key in fakes.answers()},
    )
    found = estimate(ctx, base="rw", spec=spec)

    from mst.macro.panel import build_panel

    panel = build_panel(
        ctx.store,
        ctx.registry,
        start=spec.history_start,
        as_of=ctx.as_of,
        seasonal=[v.id for v in ctx.registry.variables if v.seasonal],
    )
    x0, mu, sigma = from_history(panel)
    assert np.allclose(found.arrays["chol_u"] @ found.arrays["chol_u"].T, sigma, atol=1e-10)
    assert np.allclose(found.arrays["base_path"], x0 + np.arange(13)[:, None] * mu)
    assert found.meta["shocks"][-1] == "residual" and len(found.meta["shocks"]) == 19
    assert set(found.meta["answers"]) == set(fakes.answers())
    assert found.meta["fit_misses"] == []


def test_answers_that_take_more_than_history_are_refused(tmp_path, spec):
    def huge(payload):
        responses = {
            variable: {
                horizon: {
                    node: {
                        "q5": 40.0 * cell["q5"],
                        "q50": 40.0 * cell["q50"],
                        "q95": 40.0 * cell["q95"],
                    }
                    for node, cell in by_node.items()
                }
                for horizon, by_node in by_horizon.items()
            }
            for variable, by_horizon in payload["responses"].items()
        }
        return {**payload, "frequency": 6.0, "responses": responses}

    ctx = fakes.answers_context(tmp_path, changes={"ru.monetary": huge})
    with pytest.raises(ResidualNotAdmitted) as refused:
        estimate(ctx, base="rw", spec=spec)
    assert any(
        variable.id in str(refused.value) for variable in ctx.registry.variables
    ), "the refusal names the variables"


def test_the_version_hash_follows_the_answers(tmp_path, spec):
    first = estimate(fakes.answers_context(tmp_path / "one"), base="rw", spec=spec)
    second = estimate(fakes.answers_context(tmp_path / "two"), base="rw", spec=spec)
    other = estimate(
        fakes.answers_context(
            tmp_path / "three",
            changes={"ru.monetary": lambda payload: {**payload, "rho": 0.9}},
        ),
        base="rw",
        spec=spec,
    )
    name_first = macro_version.save("expert.elicited.rw", first.meta, first.arrays, tmp_path / "a")
    name_second = macro_version.save(
        "expert.elicited.rw", second.meta, second.arrays, tmp_path / "b"
    )
    name_other = macro_version.save("expert.elicited.rw", other.meta, other.arrays, tmp_path / "c")
    assert name_first == name_second
    assert name_first != name_other


def test_an_interrupted_estimate_resumes_from_the_checkpoint(tmp_path, spec, monkeypatch):
    from dataclasses import replace

    from mst.plugins.expert import estimate as estimate_module

    broken = replace(fakes.answers_context(tmp_path), checkpoints=tmp_path / "checkpoints")
    checkpoints = tmp_path / "checkpoints"
    original = estimate_module._shock_unit
    calls = []

    def counting(shock, answer, ids, specification, index):
        if len(calls) >= 3:
            raise RuntimeError("interrupted")
        calls.append(shock.id)
        return original(shock, answer, ids, specification, index)

    monkeypatch.setattr(estimate_module, "_shock_unit", counting)
    with pytest.raises(RuntimeError, match="interrupted"):
        estimate(broken, base="rw", spec=spec)
    assert len(list(checkpoints.iterdir())) == 3, "the finished shocks are kept"

    wrapper_calls = []

    def resuming(shock, answer, ids, specification, index):
        wrapper_calls.append(shock.id)
        return original(shock, answer, ids, specification, index)

    monkeypatch.setattr(estimate_module, "_shock_unit", resuming)
    finished = estimate(broken, base="rw", spec=spec)
    assert len(wrapper_calls) == 15, "the three finished shocks are resumed, not recomputed"

    whole = estimate(fakes.answers_context(tmp_path / "whole"), base="rw", spec=spec)
    assert (
        macro_version.save("expert.elicited.rw", finished.meta, finished.arrays, tmp_path / "x")
        == macro_version.save("expert.elicited.rw", whole.meta, whole.arrays, tmp_path / "y")
    ), "a resumed estimate gives the same version"


def test_a_checkpoint_of_another_model_version_is_not_resumed(tmp_path, spec, monkeypatch):
    from dataclasses import replace

    from mst.plugins.expert import estimate as estimate_module
    from mst.plugins.expert.model import ExpertModel

    context = replace(fakes.answers_context(tmp_path), checkpoints=tmp_path / "checkpoints")
    original = estimate_module._shock_unit
    calls = []

    def interrupted(shock, answer, ids, specification, index):
        if len(calls) >= 3:
            raise RuntimeError("interrupted")
        calls.append(shock.id)
        return original(shock, answer, ids, specification, index)

    monkeypatch.setattr(estimate_module, "_shock_unit", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        estimate(context, base="rw", spec=spec)

    recomputed = []

    def counting(shock, answer, ids, specification, index):
        recomputed.append(shock.id)
        return original(shock, answer, ids, specification, index)

    monkeypatch.setattr(estimate_module, "_shock_unit", counting)
    monkeypatch.setattr(ExpertModel, "version", ExpertModel.version + ".next")
    estimate(context, base="rw", spec=spec)
    assert len(recomputed) == 18, "the code changed: no shock is read from the old checkpoint"


def test_the_market_variant_needs_the_market_answer(tmp_path, spec):
    ctx = fakes.answers_context(tmp_path, with_market=False)
    with pytest.raises(ValueError, match=MARKET):
        estimate(ctx, base="market", spec=spec)


def test_the_market_variant_takes_the_market_path(tmp_path, spec):
    plain = estimate(fakes.answers_context(tmp_path / "rw"), base="rw", spec=spec)
    market = estimate(
        fakes.answers_context(tmp_path / "m", base="market"), base="market", spec=spec
    )
    covered = {"gl.brent", "ru.usdrub"}
    uncovered = [
        column
        for column, id_ in enumerate(plain.meta["ids"])
        if id_ not in covered
    ]
    assert np.allclose(
        plain.arrays["base_path"][:, uncovered], market.arrays["base_path"][:, uncovered]
    ), "variables without a market value keep the walk"
    brent = plain.meta["ids"].index("gl.brent")
    assert not np.allclose(
        plain.arrays["base_path"][:, brent], market.arrays["base_path"][:, brent]
    ), "a market value replaces the walk"
    moved = estimate(
        fakes.answers_context(
            tmp_path / "m2",
            base="market",
            changes={
                MARKET: lambda payload: {
                    "values": {
                        "gl.brent": {
                            "horizons": {"1": 700.0, "3": 700.0, "6": 700.0, "12": 700.0},
                            "fragment": "M1",
                        }
                    }
                }
            },
        ),
        base="market",
        spec=spec,
    )
    column = plain.meta["ids"].index("gl.brent")
    assert np.allclose(moved.arrays["base_path"][1:, column], 700.0, atol=1.0)
    assert np.isclose(
        moved.arrays["base_path"][0, column], plain.arrays["base_path"][0, column]
    )
    other = plain.meta["ids"].index("ru.key_rate")
    assert np.allclose(moved.arrays["base_path"][:, other], plain.arrays["base_path"][:, other])


def test_a_saved_version_loads_back(tmp_path, spec):
    from mst import plugins

    found = estimate(fakes.answers_context(tmp_path), base="rw", spec=spec)
    name = macro_version.save("expert.elicited.rw", found.meta, found.arrays, tmp_path / "models")
    version = macro_version.load(
        tmp_path / "models", name, plugins.registry().macro
    )
    assert version.model.ids == found.meta["ids"]
    assert len(version.model.tables) == 18
    assert version.model.base_path.shape == (13, 23)
    assert version.meta["base"] == "rw"


def test_a_residual_the_check_admits_has_a_cholesky_factor():
    from mst.plugins.expert.estimate import _cholesky

    # The check admits an eigenvalue down to -1e-10 of the largest; the factor must exist.
    sigma = np.diag([1.0, -5e-11])
    factor = _cholesky(sigma)
    assert np.allclose(factor @ factor.T, sigma, atol=1e-9)

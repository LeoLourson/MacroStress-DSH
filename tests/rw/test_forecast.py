"""Behavior of the joint walk and its intervened conditional law."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import ks_2samp, kurtosis, norm

from mst.model_sdk import (
    InfeasibleScenario,
    Intervention,
    LinearConstraint,
    LinearTerm,
    ScenarioPlan,
)
from mst.model_sdk.bounds import observed
from mst.plugins.rw.forecast import constraint_law, scale_candidates, simulate, transfer


def model(scale=None, start_sd=None):
    return SimpleNamespace(
        ids=["gl.x", "ru.y"],
        n_global=1,
        months=[date(2026, 9, 1)],
        drift=np.array([0.1, 0.0]),
        scale=np.array([[1.0, 0.6], [0.6, 2.0]]) if scale is None else scale,
        nu=8.0,
        start_mean=np.array([10.0, 20.0]),
        start_sd=np.zeros(2) if start_sd is None else start_sd,
    )


def condition(variable, month, low, high, weight=1.0):
    return LinearConstraint((LinearTerm(variable, month, weight),), low, high)


def test_transfer_is_full_conditional_including_global_response():
    m = model()
    gain, factor = transfer(m.scale, (1,))
    assert_allclose(gain, [[0.3], [1.0]])
    assert_allclose(factor @ factor.T, [[0.82, 0.0], [0.0, 0.0]], atol=1e-14)


def test_base_walk_moments_heavy_tails_and_repeatability():
    m = model()
    plan = ScenarioPlan(3)
    paths = simulate(m, plan, draws=24000, seed=31).paths
    delta = paths[:, 3] - paths[:, 0]
    assert_allclose(delta.mean(axis=0), 3 * m.drift, atol=0.065)
    assert_allclose(np.cov(delta.T), 3 * m.nu / (m.nu - 2) * m.scale, rtol=0.06)
    monthly = (paths[:, 1:] - paths[:, :-1]).reshape(-1, 2)
    assert_allclose(kurtosis(monthly[:, 0], fisher=False), 4.5, atol=0.45)
    first = simulate(m, plan, draws=8, seed=31)
    again = simulate(m, plan, draws=8, seed=31)
    assert_array_equal(first.paths, again.paths)
    assert_array_equal(first.group, np.arange(8))
    assert_array_equal(first.history[:, 0], first.paths[:, 0])


def test_set_projects_same_month_and_persists_only_through_walk():
    m = model()
    intervention = Intervention("ru.y", (2,), "set", np.array([[24.0]]))
    plan = ScenarioPlan(3, interventions=(intervention,), inner=14000)
    simulated = simulate(m, plan, draws=1, seed=8)
    p = simulated.paths
    assert_array_equal(p[:, 2, 1], np.full(len(p), 24.0))
    # Month two reads the actual random previous value, not only its unconditional mean.
    residual = p[:, 2, 0] - p[:, 1, 0] - 0.1 - 0.3 * (24 - p[:, 1, 1])
    assert abs(residual.mean()) < 0.035
    assert_allclose(residual.var(), 0.82 * 8 / 6, rtol=0.06)
    assert abs((p[:, 3, 1] - p[:, 2, 1]).mean()) < 0.06
    baseline = simulate(m, ScenarioPlan(3, inner=14000), draws=1, seed=8)
    assert_array_equal(p[:, :2], baseline.paths[:, :2])


def test_shifts_are_recursive_and_inactive_sets_are_identity():
    m = model()
    values = np.array([[3.0], [90.0]])
    inactive = Intervention("ru.y", (1,), "set", values, np.zeros((2, 1), dtype=bool))
    base = simulate(m, ScenarioPlan(3, inner=2), draws=2, seed=4)
    off = simulate(m, ScenarioPlan(3, interventions=(inactive,), inner=2), draws=2, seed=4)
    assert_array_equal(off.paths, base.paths)
    shift = Intervention("ru.y", (1,), "shift", values)
    shifted = simulate(m, ScenarioPlan(3, interventions=(shift,), inner=2), draws=2, seed=4)
    expected = base.paths.copy()
    expected[:, 1:, 1] += np.repeat(values[:, 0], 2)[:, None]
    assert_allclose(shifted.paths, expected, atol=1e-14)
    assert_array_equal(shifted.group, [0, 0, 1, 1])


def test_law_has_original_rows_and_correct_intervened_covariance():
    m = model(start_sd=np.array([0.2, 0.4]))
    constraints = (condition("gl.x", 2, -100, 100), condition("ru.y", 1, 23, 25))
    intervention = Intervention("ru.y", (1,), "set", np.array([[24.0], [25.0]]))
    plan = ScenarioPlan(2, constraints, (intervention,))
    law = constraint_law(m, plan, draws=2, seed=2)
    assert law.means.shape == (128, 2)
    assert_array_equal(law.group, np.repeat([0, 1], 64))
    assert_array_equal(law.log_weight, np.zeros(128))
    assert_allclose(law.means[:64], np.tile([11.4, 24.0], (64, 1)))
    assert_allclose(law.means[64:], np.tile([11.7, 25.0], (64, 1)))
    assert_array_equal(law.covs[:, 1], np.zeros((128, 2)))
    assert_array_equal(law.covs[:, :, 1], np.zeros((128, 2)))
    assert np.all(law.covs[:, 0, 0] > 0.2**2 + (0.3 * 0.4) ** 2)


def test_constraint_law_prices_base_box_like_unconditioned_simulation():
    m = model()
    plan = ScenarioPlan(2, (condition("gl.x", 2, 11, 13),))
    law = constraint_law(m, plan, draws=32, seed=13)
    sd = np.sqrt(law.covs[:, 0, 0])
    probability = np.mean(
        norm.cdf((13 - law.means[:, 0]) / sd) - norm.cdf((11 - law.means[:, 0]) / sd)
    )
    p = simulate(m, ScenarioPlan(2), draws=12000, seed=19).paths[:, 2, 0]
    empirical = np.mean((11 <= p) & (p <= 13))
    assert abs(probability - empirical) < 0.02


def test_observations_hold_with_sets_and_uncertain_start():
    m = model(start_sd=np.array([0.5, 0.3]))
    set_y = Intervention("ru.y", (1,), "set", np.array([[24.0], [26.0]]))
    rows = (condition("gl.x", 2, 12.0, 12.2), condition("ru.y", 1, 23, 27))
    result = simulate(m, ScenarioPlan(2, rows, (set_y,), inner=3), draws=2, seed=9)
    assert np.all((12 <= result.paths[:, 2, 0]) & (result.paths[:, 2, 0] <= 12.2))
    assert_array_equal(result.paths[:, 1, 1], [24, 24, 24, 26, 26, 26])
    assert_array_equal(result.history[:, 0], result.paths[:, 0])
    assert result.diagnostics["latent_candidates"] == 64
    assert result.diagnostics["driving_sd"]["max"] > 0


def test_exact_and_cross_month_observations_and_duplicate_refusal():
    m = model()
    row = LinearConstraint((LinearTerm("gl.x", 2, 1), LinearTerm("gl.x", 1, -1)), 1, 1)
    p = simulate(m, ScenarioPlan(2, (row,)), draws=4, seed=3).paths
    assert_allclose(p[:, 2, 0] - p[:, 1, 0], 1, atol=1e-12)
    with pytest.raises(InfeasibleScenario) as error:
        simulate(m, ScenarioPlan(2, (row, row)), draws=1, seed=3)
    assert error.value.constraints == [1]


def test_determined_false_row_reports_original_number():
    rows = (condition("gl.x", 1, 0, 100), condition("ru.y", 1, 0, 1))
    item = Intervention("ru.y", (1,), "set", np.array([[24.0]]))
    with pytest.raises(InfeasibleScenario) as error:
        simulate(model(), ScenarioPlan(1, rows, (item,)), draws=1, seed=1)
    assert error.value.constraints == [1]


def test_large_observation_reweights_monthly_scale():
    m = model()
    base = simulate(m, ScenarioPlan(1), draws=100, seed=2)
    stressed = simulate(m, ScenarioPlan(1, (condition("gl.x", 1, 17, 19),)), draws=100, seed=2)
    assert stressed.diagnostics["scale_path_mean"][0] > base.diagnostics["scale_path_mean"][0] * 2


def test_group_and_quantity_validation():
    item = Intervention("ru.y", (1,), "set", np.zeros((2, 1)))
    with pytest.raises(ValueError, match="groups"):
        simulate(model(), ScenarioPlan(1, interventions=(item,)), draws=1, seed=2)
    with pytest.raises(ValueError, match="unknown"):
        simulate(model(), ScenarioPlan(1, (condition("unknown", 1, 0, 1),)), draws=1, seed=2)
    with pytest.raises(ValueError, match="month"):
        simulate(model(), ScenarioPlan(1, (condition("gl.x", 2, 0, 1),)), draws=1, seed=2)


def test_inverse_chi_square_scale_mean():
    scales = scale_candidates(8, 3, 40000, np.random.default_rng(4))
    assert scales.shape == (40000, 3)
    assert_allclose(scales.mean(axis=0), 8 / 6, rtol=0.025)


def test_persistent_scale_keeps_the_unconditional_mean_and_correlates_months():
    scales = scale_candidates(
        8, 12, 40000, np.random.default_rng(6), phi=0.9, sigma=0.3
    )
    assert_allclose(scales.mean(axis=0), 8 / 6, rtol=0.03)
    assert np.corrcoef(scales[:, 0], scales[:, 1])[0, 1] > 0.5
    with pytest.raises(ValueError, match="persistence"):
        scale_candidates(8, 2, 10, np.random.default_rng(1), phi=1.0, sigma=0.3)


def test_persistence_centres_the_base_law_and_narrows_it():
    m = model()
    m.persistence = np.array([0.8, 0.95])
    m.intercept = np.array([0.2, 1.0])
    paths = simulate(m, ScenarioPlan(12, inner=8000), draws=1, seed=15).paths
    first = paths[:, 12, 0] - paths[:, 0, 0]
    expected = 1.0 + 0.8**12 * (10.0 - 1.0) - 10.0
    assert abs(first.mean() - expected) < 0.06
    second = paths[:, 12, 1] - paths[:, 0, 1]
    expected_second = 20.0 + 0.95**12 * (20.0 - 20.0) - 20.0
    assert abs(second.mean() - expected_second) < 0.06
    assert first.std() < np.sqrt(12 * m.nu / (m.nu - 2) * m.scale[0, 0])


def test_set_response_uses_the_ar1_baseline_of_the_set_month():
    m = model()
    m.persistence = np.array([1.0, 0.5])
    m.intercept = np.array([0.0, 10.0])
    item = Intervention("ru.y", (1,), "set", np.array([[24.0]]))
    result = simulate(m, ScenarioPlan(1, interventions=(item,), inner=16000), draws=1, seed=21)
    response = result.paths[:, 1, 0] - m.start_mean[0]
    assert abs(response.mean() - 0.3 * 4.0) < 0.05


def test_platform_reads_shadow_paths_through_registry_floor():
    from mst.macro.registry import load_registry
    from mst.model_sdk import MacroSimulation

    registry = load_registry()
    ids = ["ru.key_rate"]
    p = np.array([[[-2.0], [1.0]]])
    simulation = MacroSimulation(ids, [date(2026, 9, 1), date(2026, 10, 1)], p)
    result = observed(simulation, registry)
    assert_array_equal(result.paths, [[[0.0], [1.0]]])
    assert p[0, 0, 0] == -2


def test_reordering_registry_preserves_base_and_intervened_laws():
    m = model()
    reverse = SimpleNamespace(**vars(m))
    reverse.ids = m.ids[::-1]
    reverse.drift = m.drift[::-1]
    reverse.scale = m.scale[::-1, ::-1]
    reverse.start_mean = m.start_mean[::-1]
    reverse.start_sd = m.start_sd[::-1]
    left = simulate(m, ScenarioPlan(3), draws=5000, seed=88).paths
    right = simulate(reverse, ScenarioPlan(3), draws=5000, seed=89).paths[:, :, ::-1]
    for t in (1, 3):
        for i in range(2):
            assert ks_2samp(left[:, t, i], right[:, t, i]).statistic < 0.04
    item = Intervention("ru.y", (2,), "set", np.array([[24.0]]))
    rows = (condition("gl.x", 3, 0, 100), condition("ru.y", 3, 0, 100))
    plan = ScenarioPlan(3, rows, (item,))
    law = constraint_law(m, plan, draws=1, seed=2)
    reversed_law = constraint_law(reverse, plan, draws=1, seed=2)
    assert_allclose(law.means, reversed_law.means)
    assert_allclose(law.covs, reversed_law.covs, atol=1e-14)


def test_intervened_law_matches_path_covariance_with_uncertain_start():
    m = model(start_sd=np.array([0.2, 0.4]))
    item = Intervention("ru.y", (1,), "set", np.array([[24.0]]))
    rows = (condition("gl.x", 2, 0, 100), condition("ru.y", 2, 0, 100))
    law_item = Intervention("ru.y", (1,), "set", np.full((64, 1), 24.0))
    law = constraint_law(m, ScenarioPlan(2, rows, (law_item,)), draws=64, seed=7)
    p = simulate(m, ScenarioPlan(2, interventions=(item,), inner=15000), draws=1, seed=11).paths
    assert_allclose(p[:, 2].mean(axis=0), law.means.mean(axis=0), atol=0.06)
    assert_allclose(np.cov(p[:, 2].T), law.covs.mean(axis=0), rtol=0.06)


def test_all_cells_set_and_multiple_constraints_with_a_determined_row():
    m = model()
    items = tuple(
        Intervention(id_, (1,), "set", np.array([[amount]]))
        for id_, amount in zip(m.ids, [15.0, 25.0], strict=True)
    )
    row = condition("gl.x", 1, 14, 16)
    sim = simulate(m, ScenarioPlan(2, (row,), items, inner=3), draws=1, seed=8)
    assert_array_equal(sim.paths[:, 1], np.tile([15, 25], (3, 1)))
    rows = (row, condition("ru.y", 2, 25.1, 25.2), condition("gl.x", 2, 15.1, 15.2))
    sim = simulate(m, ScenarioPlan(2, rows, items, inner=3), draws=1, seed=8)
    assert np.all((25.1 <= sim.paths[:, 2, 1]) & (sim.paths[:, 2, 1] <= 25.2))
    assert np.all((15.1 <= sim.paths[:, 2, 0]) & (sim.paths[:, 2, 0] <= 15.2))

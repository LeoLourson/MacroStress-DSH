"""Interventions act per preserved group, in the model scale."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from mst.model_sdk.contracts import (
    Intervention,
    LinearConstraint,
    LinearTerm,
    ScenarioPlan,
    StructuralForm,
)
from mst.model_sdk.dist import ContradictsModels
from mst.model_sdk.interventions import (
    affine_constraints_under_default,
    apply_default,
    constraints_under_default,
    inner_of_groups,
)


def intervention(target="a", months=(1,), kind="set", values=((7.0,),), active=None):
    return Intervention(target, months, kind, np.asarray(values, dtype=float), active)


def condition(terms, low, high):
    return LinearConstraint(tuple(LinearTerm(*term) for term in terms), low, high)


def test_plan_carries_interventions_without_model_coordinates():
    item = intervention()
    plan = ScenarioPlan(horizon=2, interventions=(item,), inner=2)
    assert plan.interventions == (item,)
    assert {field.name for field in fields(plan)} == {
        "horizon", "constraints", "interventions", "inner"
    }
    assert StructuralForm(("a", "b"), True).time_recursion


@pytest.mark.parametrize("values", [(), (1.0,), ((1.0, 2.0),), ((np.nan,),), ((np.inf,),)])
def test_intervention_rejects_nonfinite_or_wrong_shape_values(values):
    with pytest.raises(ValueError):
        intervention(values=values)


@pytest.mark.parametrize("months", [(), (0,), (-1,), (1, 1)])
def test_intervention_rejects_empty_nonpositive_or_duplicate_months(months):
    with pytest.raises(ValueError):
        intervention(months=months, values=np.ones((2, len(months))))


def test_intervention_rejects_unknown_kind_and_wrong_mask_shape():
    with pytest.raises(ValueError):
        intervention(kind="replace")
    with pytest.raises(ValueError):
        intervention(active=np.ones((2, 1), dtype=bool))


def test_plan_rejects_overlaps_outside_horizon_and_invalid_inner():
    with pytest.raises(ValueError):
        ScenarioPlan(horizon=2, interventions=(intervention(), intervention(kind="shift")))
    with pytest.raises(ValueError):
        ScenarioPlan(horizon=1, interventions=(intervention(months=(2,)),))
    with pytest.raises(ValueError):
        ScenarioPlan(horizon=1, inner=0)


@pytest.mark.parametrize("group,inner", [(None, 1), ([0, 1, 2, 3], 1), ([0, 0, 1, 1], 2)])
def test_inner_is_derived_from_uniform_contiguous_groups(group, inner):
    simulation = SimpleNamespace(paths=np.zeros((4, 3, 1)), group=group)
    assert inner_of_groups(simulation) == inner


@pytest.mark.parametrize("group", [[0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 2, 2], [-1] * 4, [0]])
def test_inner_rejects_invalid_group_layout(group):
    simulation = SimpleNamespace(paths=np.zeros((4, 3, 1)), group=group)
    with pytest.raises(ValueError):
        inner_of_groups(simulation)


def test_default_set_is_exact_per_group_and_changes_no_other_cell():
    paths = np.arange(4 * 4 * 2, dtype=float).reshape(4, 4, 2)
    original = paths.copy()
    items = (intervention(months=(1, 3), values=((7.0, 9.0), (11.0, 13.0))),)
    found = apply_default(paths, ("a", "b"), items, inner=2)
    expected = original.copy()
    expected[:2, 1, 0], expected[:2, 3, 0] = 7.0, 9.0
    expected[2:, 1, 0], expected[2:, 3, 0] = 11.0, 13.0
    np.testing.assert_array_equal(found, expected)
    np.testing.assert_array_equal(paths, original)


def test_default_shift_has_no_dynamics_and_inactive_set_is_identity():
    paths = np.arange(4 * 4 * 2, dtype=float).reshape(4, 4, 2)
    items = (
        intervention(kind="shift", values=((2.0,), (-3.0,))),
        intervention(target="b", values=((99.0,), (99.0,)),
                     active=np.array([[True], [False]])),
    )
    expected = paths.copy()
    expected[:2, 1, 0] += 2.0
    expected[2:, 1, 0] -= 3.0
    expected[:2, 1, 1] = 99.0
    np.testing.assert_array_equal(apply_default(paths, ("a", "b"), items, inner=2), expected)


def test_default_rejects_group_mismatch_unknown_target_and_t0():
    paths = np.zeros((4, 3, 2))
    with pytest.raises(ValueError):
        apply_default(paths, ("a", "b"), (intervention(),), inner=2)
    with pytest.raises(ValueError):
        apply_default(paths, ("a", "b"), (intervention(target="missing"),))
    with pytest.raises(ValueError):
        apply_default(paths, ("a", "b"), (), inner=3)


def test_default_empty_interventions_preserve_paths_bitwise():
    paths = np.random.default_rng(31).standard_normal((4, 3, 2))
    np.testing.assert_array_equal(apply_default(paths, ("a", "b"), (), inner=2), paths)


def test_default_helpers_do_not_draw_random_numbers(monkeypatch):
    def unexpected_draw(*args, **kwargs):
        raise AssertionError("default intervention helpers must not draw random numbers")

    monkeypatch.setattr(np.random, "default_rng", unexpected_draw)
    paths = np.zeros((2, 3, 1))
    item = intervention(values=((3.0,), (4.0,)))
    result = apply_default(paths, ("a",), (item,))
    np.testing.assert_array_equal(result[:, 1, 0], [3.0, 4.0])
    constraint = condition((("a", 1, 1.0),), 2.0, 5.0)
    assert constraints_under_default((constraint,), (item,), ("a",), group=1) == ()


def test_default_constraints_substitute_set_and_shift_for_selected_group():
    constraint = condition((("a", 1, 2.0), ("b", 2, -3.0), ("c", 1, 4.0)), -5.0, 20.0)
    items = (
        intervention(values=((7.0,), (9.0,))),
        intervention(target="b", months=(2,), kind="shift", values=((2.0,), (-1.0,))),
    )
    first = constraints_under_default((constraint,), items, ("a", "b", "c"), group=0)[0]
    second = constraints_under_default((constraint,), items, ("a", "b", "c"), group=1)[0]
    assert first.terms == second.terms == constraint.terms[1:]
    assert (first.low, first.high) == (-13.0, 12.0)
    assert (second.low, second.high) == (-26.0, -1.0)


def test_default_constraints_drop_true_constants_and_name_contradiction():
    matching = condition((("a", 1, 1.0),), 7.0, 7.0)
    wrong = condition((("a", 1, 1.0),), 8.0, 9.0)
    assert constraints_under_default((matching,), (intervention(),), ("a",), group=0) == ()
    with pytest.raises(ContradictsModels, match=r"(?:constraint|condition).*2"):
        constraints_under_default((matching, wrong), (intervention(),), ("a",), group=0)


def test_affine_constraints_keep_original_rows_for_the_law():
    constraints = (
        condition((("a", 1, 2.0), ("b", 2, -3.0)), -5.0, 20.0),
        condition((("a", 1, 1.0),), 7.0, 7.0),
    )
    items = (
        intervention(values=((7.0,),)),
        intervention(target="b", months=(2,), kind="shift", values=((2.0,),)),
    )
    rows, offsets = affine_constraints_under_default(constraints, items, ("a", "b"), group=0)
    assert len(rows) == 2
    assert rows[0].terms == constraints[0].terms[1:]
    assert rows[1].terms == ()
    assert (rows[0].low, rows[0].high) == (-5.0, 20.0)
    np.testing.assert_array_equal(offsets, [8.0, 7.0])


def test_inactive_interventions_leave_constraint_unchanged():
    constraint = condition((("a", 1, 1.0),), 8.0, 9.0)
    item = intervention(active=np.array([[False]]))
    assert constraints_under_default((constraint,), (item,), ("a",), group=0) == (constraint,)

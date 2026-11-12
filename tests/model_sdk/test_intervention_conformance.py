"""Conformance follows the platform's default/recursive macro dispatch boundary."""

from datetime import date

import numpy as np

from mst.model_sdk.contracts import MacroModel, MacroSimulation, StructuralForm
from mst.model_sdk.testing import check_interventions


class DefaultModel(MacroModel):
    model_id = "fake.default"
    version = "1"
    label = "default"
    variables = ("a", "b")

    def estimate(self, context):
        raise NotImplementedError

    def load(self, meta, arrays):
        raise NotImplementedError

    def simulate(self, model, plan, *, draws, seed):
        paths = np.random.default_rng(seed).standard_normal(
            (draws * plan.inner, plan.horizon + 1, 2)
        )
        return MacroSimulation(
            ids=["a", "b"],
            months=[date(2026, month + 1, 1) for month in range(plan.horizon + 1)],
            paths=paths,
            group=np.repeat(np.arange(draws), plan.inner),
        )


class RecursiveModel(DefaultModel):
    def structural_form(self, model):
        return StructuralForm(("a", "b"), True)

    def simulate(self, model, plan, *, draws, seed):
        found = super().simulate(model, plan, draws=draws, seed=seed)
        for item in plan.interventions:
            k = found.ids.index(item.target)
            for position, month in enumerate(item.months):
                for group in range(draws):
                    rows = slice(group * plan.inner, (group + 1) * plan.inner)
                    if item.active is not None and not item.active[group, position]:
                        continue
                    if item.kind == "set":
                        delta = item.values[group, position] - found.paths[rows, month, k]
                    else:
                        delta = np.full(plan.inner, item.values[group, position])
                    found.paths[rows, month:, k] += delta[:, None]
                    if item.kind == "set":
                        found.paths[rows, month, k] = item.values[group, position]
        return found


class WrongGroup(DefaultModel):
    def simulate(self, model, plan, *, draws, seed):
        found = super().simulate(model, plan, draws=draws, seed=seed)
        found.group[:] = 0
        return found


class IgnoredRecursiveIntervention(DefaultModel):
    def structural_form(self, model):
        return StructuralForm(("a", "b"), True)


def test_default_plugin_returns_preintervention_paths_and_conforms():
    assert check_interventions(DefaultModel(), object(), horizon=3, draws=2, inner=2) == []


def test_recursive_plugin_executes_interventions_itself():
    assert check_interventions(RecursiveModel(), object(), horizon=3, draws=2, inner=2) == []


def test_conformance_reports_wrong_contiguous_group_membership():
    findings = check_interventions(WrongGroup(), object(), horizon=3, draws=2, inner=2)
    assert any("group" in str(finding) for finding in findings)


def test_conformance_reports_recursive_plugin_ignoring_set():
    findings = check_interventions(
        IgnoredRecursiveIntervention(), object(), horizon=3, draws=2, inner=2
    )
    assert any("set" in str(finding) for finding in findings)

"""The reverse problem: the worst outcome a given price still allows.

The checks follow the plan: a family with a closed-form answer — a price and a loss both
linear in the width — is searched by both questions and the boundary is found; the family of
the scenario fixture does not get cheaper as its conditions narrow; a family whose
interventions move with the parameter is refused, because a free intervention costs nothing
and a bound on the price would bound nothing; and the command line takes the versions as
arguments or from the newest finished pair, never from the scenario file.
"""

import json
import os
from types import SimpleNamespace

import pytest

from mst import plugins, store
from mst.macro.registry import load_registry
from mst.macro.version import load as load_macro
from mst.scenario.context import context_from_versions
from mst.scenario.contracts import Scenario
from mst.scenario.reverse import Family, width_family
from mst.scenario.translate import calendar_month, observations_of, tolerance_of

OLDER, NEWER = "2" * 64, "3" * 64
SATELLITES, OTHER = "5" * 64, "6" * 64


def month(versions, index: int) -> str:
    return calendar_month(index, versions.data.months[-1].strftime("%Y-%m"))


def anchor(versions) -> float:
    """The last panel coordinate of the oil price, in the model scale."""
    return float(versions.data.y[-1, versions.data.ids.index("gl.brent")])


class _Known:
    """A run context that answers from a table: the price and the loss of every width.

    The price falls with the width and the loss falls with it too — the wider the
    conditions, the milder the scenario — so the two questions have closed answers:
    ``sup_rho`` under ``kappa`` stops where ``a - b c <= kappa`` and ``inf_dist`` over
    ``level`` stops where ``d - e c >= level``.
    """

    def __init__(self, price, loss):
        self.price = price  # dist(c) = a - b c
        self.loss = loss  # rho(c) = d - e c

    def point(self, c):
        class Row:
            dist = self.price[0] - self.price[1] * c
            rho = self.loss[0] - self.loss[1] * c

        return Row


def _family_of(known):
    """A family whose evaluation reads the known table: interventions never move."""

    def build(c: float):
        return Scenario.model_validate(
            {
                "narrative": "test",
                "approved_by": "test",
                "observations": [
                    {
                        "kind": "constraint",
                        "terms": [{"quantity": "ru.key_rate", "month": "2026-11", "weight": 1.0}],
                        "low": -c,
                        "high": c,
                        "unit": "level",
                        "basis": "assumption",
                    }
                ],
            }
        )

    return Family(build=build)


def _patch_evaluate(monkeypatch, known):
    """Point the reverse search at the known table instead of a run."""
    from mst.scenario import reverse

    def evaluate(spec, macro, satellites, positions, *, draws, seed, base_currency="RUB"):
        c = spec.observations[0].high
        row = known.point(c)
        return reverse.Point(c=c, dist=row.dist, rho=row.rho)

    monkeypatch.setattr(reverse, "evaluate", evaluate)


def test_sup_rho_finds_the_boundary_of_a_linear_family(monkeypatch):
    from mst.scenario import reverse

    known = _Known(price=(4.0, 2.0), loss=(10.0, 3.0))
    _patch_evaluate(monkeypatch, known)
    family = _family_of(known)
    found = reverse.sup_rho(family, 3.0, macro=None, satellites=None, positions=[], draws=1, seed=1)
    # dist(c) = 4 - 2c <= 3 stops at c = 0.5, and rho there is 10 - 3 * 0.5 = 8.5.
    assert found["answer"]["c"] == pytest.approx(0.5, abs=0.05)
    assert found["answer"]["rho"] == pytest.approx(8.5, abs=0.2)


def test_sup_rho_refines_a_boundary_between_the_grid_points(monkeypatch):
    from mst.scenario import reverse

    known = _Known(price=(4.0, 2.0), loss=(10.0, 3.0))
    _patch_evaluate(monkeypatch, known)
    found = reverse.sup_rho(
        _family_of(known), 3.4, macro=None, satellites=None, positions=[], draws=1, seed=1
    )
    # 4 - 2c <= 3.4 stops at c = 0.3, between the grid points 0.2 and 0.35; five bisections of
    # that step come within 0.005 of it, and the loss is the largest the price allows.
    assert 0.3 <= found["answer"]["c"] <= 0.305
    assert found["answer"]["dist"] <= 3.4
    assert found["answer"]["rho"] == pytest.approx(9.1, abs=0.02)


def test_inf_dist_finds_the_boundary_of_a_linear_family(monkeypatch):
    from mst.scenario import reverse

    known = _Known(price=(4.0, 2.0), loss=(10.0, 3.0))
    _patch_evaluate(monkeypatch, known)
    family = _family_of(known)
    found = reverse.inf_dist(
        family, 8.5, macro=None, satellites=None, positions=[], draws=1, seed=1
    )
    # rho(c) = 10 - 3c >= 8.5 stops at c = 0.5, and the price there is 3.
    assert found["answer"]["c"] == pytest.approx(0.5, abs=0.05)
    assert found["answer"]["dist"] == pytest.approx(3.0, abs=0.2)


def test_no_affordable_scenario_answers_none(monkeypatch):
    from mst.scenario import reverse

    known = _Known(price=(40.0, 2.0), loss=(10.0, 3.0))
    _patch_evaluate(monkeypatch, known)
    family = _family_of(known)
    found = reverse.sup_rho(family, 0.1, macro=None, satellites=None, positions=[], draws=1, seed=1)
    assert found["answer"] is None
    assert found["points"]


def test_the_cheapest_point_of_a_large_error_is_not_chosen(monkeypatch):
    # The cheapest scenarios of the family carry a price whose error is too large: they are
    # treated as points that do not reach the level, so the answer rests on the estimated
    # boundary and names its own mark.
    from mst.scenario import reverse
    from mst.scenario.reverse import Point

    def evaluate(spec, macro, satellites, positions, *, draws, seed, base_currency="RUB"):
        c = spec.observations[0].high
        return Point(
            c=c,
            dist=4.0 - 2.0 * c,
            rho=10.0 - 3.0 * c,
            estimated=c <= 0.45,  # the widest, and so the cheapest, points are noise
        )

    monkeypatch.setattr(reverse, "evaluate", evaluate)
    found = reverse.inf_dist(
        _family_of(None), 8.5, macro=None, satellites=None, positions=[], draws=1, seed=1
    )
    # rho >= 8.5 holds up to c = 0.5, but only the points up to 0.45 are estimated: the
    # answer stops at the estimated boundary, with the price 4 - 2 * 0.45.
    assert found["answer"]["c"] == pytest.approx(0.45, abs=0.02)
    assert found["answer"]["dist"] == pytest.approx(3.1, abs=0.06)
    assert found["answer"]["estimated"] is True
    assert any(row["estimated"] is False for row in found["points"])


def test_the_worst_point_of_a_large_error_is_not_chosen(monkeypatch):
    # The narrow, expensive scenarios carry the largest losses and a price that is noise:
    # they do not open the affordable region, and the worst outcome is read on the
    # estimated points alone.
    from mst.scenario import reverse
    from mst.scenario.reverse import Point

    def evaluate(spec, macro, satellites, positions, *, draws, seed, base_currency="RUB"):
        c = spec.observations[0].high
        return Point(
            c=c,
            dist=4.0 - 2.0 * c,
            rho=10.0 - 3.0 * c,
            estimated=c > 0.75,  # the narrowest, and so the dearest, points are noise
        )

    monkeypatch.setattr(reverse, "evaluate", evaluate)
    found = reverse.sup_rho(
        _family_of(None), 3.0, macro=None, satellites=None, positions=[], draws=1, seed=1
    )
    # Without the rule the answer would sit at c = 0.5 with rho = 8.5; the estimated
    # boundary opens at 0.75, and the worst outcome there is rho = 10 - 3 * 0.75.
    assert found["answer"]["c"] == pytest.approx(0.75, abs=0.02)
    assert found["answer"]["rho"] == pytest.approx(7.75, abs=0.06)
    assert found["answer"]["estimated"] is True
    assert found["answer"]["dist"] <= 3.0


@pytest.fixture(scope="module")
def macro(versions):
    """The macro version loaded back, as a search receives it."""
    return load_macro(versions.models, versions.macro, plugins.registry().macro)


def _spec_of(versions):
    """A scenario of the fixture: one support condition and one intervention."""
    variable = load_registry().variable("gl.brent")
    return Scenario.model_validate(
        {
            "narrative": "нефть",
            "approved_by": "И. Иванов",
            "observations": [
                {
                    "kind": "constraint",
                    "terms": [{"quantity": "gl.brent", "month": month(versions, 2), "weight": 1.0}],
                    "unit": "level",
                    "low": float(variable.from_model(anchor(versions) - 2.0)),
                    "high": float(variable.from_model(anchor(versions) + 2.0)),
                    "basis": "assumption",
                }
            ],
            "interventions": [
                {
                    "quantity": "gl.brent",
                    "months": [month(versions, 2)],
                    "kind": "shift",
                    "values": [1.0],
                    "unit": "level",
                    "reason": "шок",
                    "basis": "assumption",
                }
            ],
        }
    )


def test_the_price_of_the_fixture_does_not_fall_as_the_conditions_narrow(versions, macro):
    from mst.scenario.build import dist_of

    family = width_family(_spec_of(versions), macro, versions.satellite)
    prices = [dist_of(family.build(c), macro, draws=60, seed=5).distance for c in (4.0, 1.0, 0.35)]
    assert prices[0] < prices[1] < prices[2]


def test_a_family_whose_interventions_move_is_refused(versions):
    spec = _spec_of(versions)

    def build(c: float):
        return spec.model_copy(
            update={
                "interventions": [
                    spec.interventions[0].model_copy(
                        update={"values": [value * c for value in spec.interventions[0].values]}
                    )
                ]
            }
        )

    family = Family(build=build)
    with pytest.raises(ValueError, match="interventions move with the parameter"):
        family.check()


def test_a_point_of_the_family_is_its_tolerance_at_one_whatever_its_value(versions, macro):
    variable = load_registry().variable("gl.brent")
    spec = _spec_of(versions)
    point = spec.observations[0].model_copy(
        update={
            "low": None,
            "high": None,
            "value": float(variable.from_model(anchor(versions))),
        }
    )
    spec = spec.model_copy(update={"observations": [point]})
    data = context_from_versions(macro, versions.satellite)
    (translated,) = observations_of(spec, data, 12)
    tau = tolerance_of(translated.terms, data)
    family = width_family(spec, macro, versions.satellite)
    own = observations_of(family.build(1.0), data, 12)[0]
    assert (own.low, own.high) == pytest.approx((anchor(versions) - tau, anchor(versions) + tau))
    narrow = observations_of(family.build(0.5), data, 12)[0]
    assert (narrow.low, narrow.high) == pytest.approx(
        (anchor(versions) - tau / 2, anchor(versions) + tau / 2)
    )


def test_the_family_keeps_a_change_or_a_percent_condition_at_one(versions, macro):
    # The bounds move in the model scale, where the price reads them; a unit of the event is
    # written back so that the same coordinate comes out of the translation again.
    data = context_from_versions(macro, versions.satellite)
    for unit, terms, low, high in (
        (
            "change",
            [{"quantity": "gl.brent", "month": month(versions, 2), "weight": 1.0}],
            1.0,
            2.0,
        ),
        (
            "percent",
            [
                {"quantity": "gl.brent", "month": month(versions, 1), "weight": -1.0},
                {"quantity": "gl.brent", "month": month(versions, 2), "weight": 1.0},
            ],
            5.0,
            10.0,
        ),
    ):
        spec = Scenario.model_validate(
            {
                "narrative": "нефть",
                "approved_by": "И. Иванов",
                "observations": [
                    {
                        "kind": "constraint",
                        "terms": terms,
                        "unit": unit,
                        "low": low,
                        "high": high,
                        "basis": "assumption",
                    }
                ],
            }
        )
        (source,) = observations_of(spec, data, 12)
        family = width_family(spec, macro, versions.satellite)
        (own,) = observations_of(family.build(1.0), data, 12)
        assert (own.low, own.high) == pytest.approx((source.low, source.high))
        (narrow,) = observations_of(family.build(0.5), data, 12)
        middle = (source.low + source.high) / 2
        half = (source.high - source.low) / 2
        assert (narrow.low, narrow.high) == pytest.approx((middle - half / 2, middle + half / 2))


def test_the_command_line_takes_the_versions_as_arguments(tmp_path):
    from mst.scenario.__main__ import main

    # The arguments parse: the unknown scenario fails later, not with a usage error.
    with pytest.raises((OSError, KeyError, ValueError)):
        main(
            [
                "reverse",
                "nowhere",
                "--workspace",
                str(tmp_path),
                "--portfolio",
                "p",
                "--draws",
                "2",
                "--seed",
                "1",
                "--kappa",
                "1.0",
            ]
        )
    with pytest.raises((OSError, KeyError, ValueError)):
        main(
            [
                "reverse",
                "nowhere",
                "--workspace",
                str(tmp_path),
                "--portfolio",
                "p",
                "--draws",
                "2",
                "--seed",
                "1",
                "--level",
                "1.0",
                "--macro",
                OLDER,
                "--satellites",
                OTHER,
            ]
        )


def version_folder(directory, name, *, mtime, **meta):
    """A finished version folder: the meta is all ``versions.folders`` reads to find it."""
    folder = directory / name
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    os.utime(folder, (mtime, mtime))
    return folder


def fake_models(tmp_path):
    """Two macro versions and two bundles; the newest bundle belongs to the older macro."""
    models = tmp_path / "models"
    version_folder(models / "macro", OLDER, mtime=1_000, as_of="2026-01-01")
    version_folder(models / "macro", NEWER, mtime=2_000, as_of="2026-05-01")
    version_folder(models / "satellites", SATELLITES, mtime=2_500, macro=NEWER)
    version_folder(models / "satellites", OTHER, mtime=3_000, macro=OLDER)
    return models


def fake_loaders(monkeypatch, *, bundle_macro=NEWER):
    """The two model loaders of the selection, recording the names they receive."""
    calls = []
    macro = SimpleNamespace(name=NEWER)
    bundle = SimpleNamespace(macro=bundle_macro)

    def load_macro(root, name, installed):
        calls.append(("macro", name))
        macro.name = name
        return macro

    def load_satellites(root, name, installed):
        calls.append(("satellites", name))
        return bundle

    monkeypatch.setattr("mst.macro.version.load", load_macro)
    monkeypatch.setattr("mst.drivers.bundle.load", load_satellites)
    return calls, macro, bundle


def test_versions_of_takes_the_newest_finished_pair(tmp_path, monkeypatch):
    from mst.scenario.__main__ import versions_of

    fake_models(tmp_path)
    monkeypatch.setattr(plugins, "registry", lambda: SimpleNamespace(macro={}, satellites={}))
    calls, macro, bundle = fake_loaders(monkeypatch)
    found = versions_of(store.Workspace(tmp_path))
    assert calls == [("macro", NEWER), ("satellites", SATELLITES)]
    assert found == (macro, bundle)


def test_versions_of_honours_explicit_names(tmp_path, monkeypatch):
    from mst.scenario.__main__ import versions_of

    fake_models(tmp_path)
    monkeypatch.setattr(plugins, "registry", lambda: SimpleNamespace(macro={}, satellites={}))
    calls, _, bundle = fake_loaders(monkeypatch, bundle_macro=OLDER)
    found = versions_of(store.Workspace(tmp_path), OLDER, OTHER)
    assert calls == [("macro", OLDER), ("satellites", OTHER)]
    assert found[0].name == OLDER and found[1] is bundle


def test_versions_of_refuses_satellites_on_another_macro(tmp_path, monkeypatch):
    from mst.scenario.__main__ import versions_of

    fake_models(tmp_path)
    monkeypatch.setattr(plugins, "registry", lambda: SimpleNamespace(macro={}, satellites={}))
    fake_loaders(monkeypatch, bundle_macro=OLDER)
    with pytest.raises(ValueError, match="estimated on macro version"):
        versions_of(store.Workspace(tmp_path), NEWER, OTHER)


def test_versions_of_names_a_workspace_without_a_macro_version(tmp_path, monkeypatch):
    from mst.scenario.__main__ import versions_of

    monkeypatch.setattr(plugins, "registry", lambda: SimpleNamespace(macro={}, satellites={}))
    fake_loaders(monkeypatch)
    with pytest.raises(ValueError, match="no finished macro version"):
        versions_of(store.Workspace(tmp_path))

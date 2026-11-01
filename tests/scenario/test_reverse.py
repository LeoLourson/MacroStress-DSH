"""The reverse problem: the worst outcome a given price still allows.

The checks follow the plan: a family with a closed-form answer — a price and a loss both
linear in the width — is searched by both questions and the boundary is found; the family of
the scenario fixture does not get cheaper as its conditions narrow; and a family whose
impacts move with the parameter is refused, because a free impact costs nothing and a bound
on the price would bound nothing.
"""

import pytest

from mst.scenario.reverse import Family, width_family


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
    """A family whose evaluation reads the known table: impacts never move."""

    def build(c: float):
        from mst.scenario.spec import Spec

        return Spec.model_validate(
            {
                "narrative": "test",
                "horizon": 3,
                "macro": "a" * 64,
                "satellites": "b" * 64,
                "approved_by": "test",
                "constraints": [
                    {
                        "terms": [{"variable": "x", "month": 1, "weight": 1.0}],
                        "low": -c,
                        "high": c,
                        "basis": "assumption",
                    }
                ],
            }
        )

    family = Family(build=build)
    return family


def _patch_evaluate(monkeypatch, known):
    """Point the reverse search at the known table instead of a run."""
    from mst.scenario import reverse

    def evaluate(spec, macro, satellites, positions, *, draws, seed, base_currency="RUB"):
        c = spec.constraints[0].high
        row = known.point(c)
        return reverse.Point(c=c, dist=row.dist, rho=row.rho)

    monkeypatch.setattr(reverse, "evaluate", evaluate)


def test_sup_rho_finds_the_boundary_of_a_linear_family(monkeypatch):
    from mst.scenario import reverse

    known = _Known(price=(4.0, 2.0), loss=(10.0, 3.0))
    _patch_evaluate(monkeypatch, known)
    family = _family_of(known)
    found = reverse.sup_rho(
        family, 3.0, macro=None, satellites=None, positions=[], draws=1, seed=1
    )
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
    found = reverse.sup_rho(
        family, 0.1, macro=None, satellites=None, positions=[], draws=1, seed=1
    )
    assert found["answer"] is None
    assert found["points"]


def test_the_price_of_the_fixture_does_not_fall_as_the_conditions_narrow(versions):
    from mst.scenario.build import dist_of

    spec = _spec_of(versions)
    family = width_family(spec, versions.loaded)
    prices = [
        dist_of(family.build(c), versions.loaded, draws=60, seed=5).distance
        for c in (4.0, 1.0, 0.35)
    ]
    assert prices[0] < prices[1] < prices[2]


def test_a_family_whose_impacts_move_is_refused(versions):
    from mst.scenario.reverse import Family

    spec = _spec_of(versions)

    def build(c: float):
        return spec.model_copy(
            update={
                "impacts": [
                    spec.impacts[0].model_copy(update={"delta": spec.impacts[0].delta * c})
                ]
            }
        )

    family = Family(build=build)
    with pytest.raises(ValueError, match="impacts move with the parameter"):
        family.check()


def _spec_of(versions):
    """A scenario of the fixture: one interval condition and one impact."""
    from mst.scenario.spec import Spec

    return Spec.model_validate(
        {
            "narrative": "нефть",
            "horizon": 6,
            "macro": versions.macro,
            "satellites": versions.satellites,
            "approved_by": "И. Иванов",
            "constraints": [
                {
                    "terms": [{"variable": "gl.brent", "month": 2, "weight": 1.0}],
                    "low": -2.0,
                    "high": 2.0,
                    "basis": "assumption",
                }
            ],
            "impacts": [
                {
                    "target": "gl.brent",
                    "start": 1,
                    "end": 3,
                    "delta": 1.0,
                    "reason": "шок",
                    "basis": "assumption",
                }
            ],
        }
    )


def test_a_point_of_the_family_is_its_tolerance_at_one_whatever_its_value(versions):
    from mst.scenario.build import tolerance_of

    spec = _spec_of(versions)
    point = spec.constraints[0].model_copy(update={"low": None, "high": None, "value": 0.0})
    family = width_family(spec.model_copy(update={"constraints": [point]}), versions.loaded)
    tau = tolerance_of(point.terms, versions.loaded)
    own = family.build(1.0).constraints[0]
    assert (own.low, own.high) == pytest.approx((-tau, tau))
    narrow = family.build(0.5).constraints[0]
    assert (narrow.low, narrow.high) == pytest.approx((-tau / 2, tau / 2))


def test_the_command_line_reads_the_reverse_form_it_documents(tmp_path):
    from mst.scenario.__main__ import main

    # The arguments parse: the unknown scenario fails later, not with a usage error.
    with pytest.raises((OSError, KeyError, ValueError)):
        main(
            [
                "reverse", "nowhere", "--workspace", str(tmp_path), "--portfolio", "p",
                "--draws", "2", "--seed", "1", "--kappa", "1.0",
            ]
        )

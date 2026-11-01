"""The reverse problem: the worst outcome a given price still allows.

The canon's section 6 turns the question around: instead of asking what a scenario costs, it
asks either for the worst functional ``rho`` among the scenarios whose price stays under
``kappa``, or for the cheapest scenario whose ``rho`` reaches a level ``L*``. The family of
scenarios is fixed by its versions ``m``, its impacts ``O`` and its causes ``D`` — only the
conditions' width moves with the parameter ``c`` — because a free ``O`` would cost nothing
and the bound on the price would bound nothing.

``Family`` carries the builder ``c -> Spec``; ``width_family`` is the natural one: every
condition of a scenario narrows or widens around its midpoint by the same factor. The search
walks a grid of ``c`` and refines around the boundary with bisection: the streams of stage 4
make the price and the loss smooth in ``c`` up to the Monte-Carlo error, so a handful of
refinements finds the boundary the report states.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .build import build as build_paths
from .spec import Spec

GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0)
"""The coarse grid of the width factor ``c`` the search starts from."""
REFINEMENTS = 5
"""Bisection steps after the grid: the boundary narrows by 2^-5 of the bracketing step."""


@dataclass(frozen=True)
class Family:
    """A one-parameter family of scenarios with fixed versions, impacts and causes.

    ``build`` maps the width factor ``c`` to a scenario: ``c = 1`` is the family's own
    scenario, smaller ``c`` narrows its conditions towards their midpoints and a larger one
    widens them. Everything else — the versions, the impacts ``O``, the causes ``D`` — stays
    put, and the search refuses a family whose impacts move with ``c``: a free impact costs
    nothing, and a bound on the price would bound nothing.
    """

    build: Callable[[float], Spec]

    def check(self) -> list[Spec]:
        """The variants of the coarse grid, refused when their impacts differ.

        The impacts are compared as the scenario contract stores them: any difference in
        target, months or delta breaks the family, whatever the builder meant by it.
        """
        variants = [self.build(c) for c in GRID]
        impacts = [
            [impact.model_dump() for impact in variant.impacts] for variant in variants
        ]
        if any(other != impacts[0] for other in impacts[1:]):
            raise ValueError(
                "the family's impacts move with the parameter: only the conditions may "
                "vary, a free impact O costs nothing and the price bounds nothing"
            )
        return variants


def width_family(spec: Spec) -> Family:
    """The family of a scenario: every condition narrows and widens around its midpoint."""

    def build(c: float) -> Spec:
        constraints = []
        for constraint in spec.constraints:
            terms = constraint.terms
            if constraint.value is not None:
                constraints.append(
                    constraint.model_copy(
                        update={
                            "low": constraint.value - c * abs(constraint.value),
                            "high": constraint.value + c * abs(constraint.value),
                            "value": None,
                        }
                    )
                )
                continue
            middle = (constraint.low + constraint.high) / 2
            half = (constraint.high - constraint.low) / 2
            constraints.append(
                constraint.model_copy(
                    update={"low": middle - c * half, "high": middle + c * half}
                )
            )
        return spec.model_copy(update={"constraints": constraints})

    return Family(build=build)


@dataclass(frozen=True)
class Point:
    """One evaluated scenario of the family: its width, its price and its outcome."""

    c: float
    dist: float | None
    rho: float | None


def evaluate(spec: Spec, macro, satellites, positions, *, draws: int, seed: int, base_currency="RUB") -> Point:
    """``(dist, rho)`` of one scenario: the price and the loss the reverse problem trades.

    ``rho`` is the expected shortfall of the portfolio loss at 95 % — the tail the canon's
    example names — on the paths of the scenario; ``dist`` is the scenario's price, absent
    when the scenario carries nothing to price.
    """
    from ..engine.run import run as calculate
    from ..portfolio.report import losses_of
    from ..runs import readable
    from ..metrics.es import es_block

    paths = build_paths(spec, macro, satellites, draws=draws, seed=seed)
    batch = calculate(
        positions,
        readable(paths),
        drivers=satellites.registry,
        served=satellites.served(),
        base_currency=base_currency,
        as_of=macro.as_of,
    )
    losses = losses_of(batch.results)
    rho = float(es_block(losses, seed=seed)["es95"]) if losses.size else None
    dist = None if paths.dist is None else float(paths.dist.distance)
    return Point(c=float("nan"), dist=dist, rho=rho)


def sup_rho(
    family: Family,
    kappa: float,
    *,
    macro,
    satellites,
    positions,
    draws: int,
    seed: int,
    grid: tuple[float, ...] = GRID,
) -> dict:
    """The worst ``rho`` among the family's scenarios priced at ``kappa`` or less.

    The coarse grid is evaluated first, then the boundary — the largest ``c``-neighbourhood
    where the price stays under ``kappa`` — is bisected. The answer names the point and the
    whole path of the search, so the reader sees the trade the price allows.
    """
    points = _walk(
        family, macro, satellites, positions, draws=draws, seed=seed, grid=grid
    )
    affordable = [point for point in points if point.dist is not None and point.dist <= kappa]
    if not affordable:
        return {"kappa": kappa, "answer": None, "points": _rows(points)}
    best = max(affordable, key=lambda point: (point.rho is not None, point.rho))
    # Refine towards the boundary: the price grows as the conditions narrow, so the
    # affordable region sits at the small-c end of the grid.
    low, high = _bracket(affordable, points)
    for _ in range(REFINEMENTS):
        if high is None:
            break
        middle = (low + high) / 2
        point = _evaluate(family, middle, macro, satellites, positions, draws, seed)
        points.append(point)
        if point.dist is not None and point.dist <= kappa:
            low = middle
            if point.rho is not None and (best.rho is None or point.rho >= best.rho):
                best = point
        else:
            high = middle
    return {
        "kappa": kappa,
        "answer": None if best.rho is None else {"c": best.c, "rho": best.rho, "dist": best.dist},
        "points": _rows(points),
    }


def inf_dist(
    family: Family,
    level: float,
    *,
    macro,
    satellites,
    positions,
    draws: int,
    seed: int,
    grid: tuple[float, ...] = GRID,
) -> dict:
    """The cheapest price among the family's scenarios whose ``rho`` reaches ``level``."""
    points = _walk(
        family, macro, satellites, positions, draws=draws, seed=seed, grid=grid
    )
    reached = [point for point in points if point.rho is not None and point.rho >= level]
    if not reached:
        return {"level": level, "answer": None, "points": _rows(points)}
    best = min(reached, key=lambda point: point.dist)
    # The loss falls as the conditions widen and the price with it: refine towards the
    # boundary where rho stops reaching the level.
    ordered = sorted(reached, key=lambda point: point.c)
    low, high = _bracket(ordered, points)
    for _ in range(REFINEMENTS):
        if high is None:
            break
        middle = (low + high) / 2
        point = _evaluate(family, middle, macro, satellites, positions, draws, seed)
        points.append(point)
        if point.rho is not None and point.rho >= level:
            low = middle
            if point.dist is not None and (best.dist is None or point.dist <= best.dist):
                best = point
        else:
            high = middle
    return {
        "level": level,
        "answer": None if best.dist is None else {"c": best.c, "rho": best.rho, "dist": best.dist},
        "points": _rows(points),
    }


def _walk(family, macro, satellites, positions, *, draws, seed, grid):
    """Every point of the coarse grid, with the family's impacts checked first."""
    variants = family.check()
    points = [
        _evaluate(family, c, macro, satellites, positions, draws, seed)
        for c in grid
    ]
    # The check built the variants; the evaluation builds them again — the same builder,
    # the same scenarios.
    assert len(variants) == len(points)
    return points


def _evaluate(family, c, macro, satellites, positions, draws, seed) -> Point:
    """One point of the family, carrying its own ``c``."""
    point = evaluate(
        family.build(c), macro, satellites, positions, draws=draws, seed=seed
    )
    return Point(c=float(c), dist=point.dist, rho=point.rho)


def _bracket(inside, points):
    """``(low, high)`` around the boundary: the best inside point and the next one out."""
    best = max(inside, key=lambda point: point.c)
    larger = [point.c for point in points if point.c > best.c]
    return best.c, min(larger) if larger else None


def _rows(points) -> list[dict]:
    """The search path as plain rows: the width, the price and the outcome of every point."""
    return [
        {
            "c": point.c,
            "dist": point.dist,
            "rho": point.rho,
        }
        for point in sorted(points, key=lambda point: point.c)
    ]

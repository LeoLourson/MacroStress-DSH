"""The forecast of one portfolio, valued by the position models a run values it with.

The forecast is the empty scenario of a macro and a satellite version, built and valued as a
run builds and values it: the same paths, the same position models, the loss by every month.
The same portfolio held over the past is ``mst.portfolio.history.replay``.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from mst import runs
from mst.engine.run import run as engine_run
from mst.plugin_sdk.paths import FixedPaths
from mst.portfolio.report import losses_of
from mst.scenario import build as scenario_build
from mst.scenario.spec import Spec

EMPTY = Spec(narrative="s0: the empty scenario of the forecast check set")


@dataclass(frozen=True)
class Forecast:
    paths: object
    """What ``mst.scenario.build.build`` returned: macro, shadow and driver paths."""
    returns: np.ndarray
    """``(R, H)`` the portfolio's return by months ``1 .. H``, % of its current value."""
    value: float
    """The portfolio's current value, millions of the base currency."""
    drivers: list[str]


def forecast(macro, satellites, portfolio, *, draws: int, seed: int, horizon: int) -> Forecast:
    request = SimpleNamespace(targets=[])
    needed = runs.needed_drivers(
        portfolio.positions, EMPTY, satellites, request, macro, horizon, portfolio.base_currency
    )
    paths = scenario_build.build(
        EMPTY, macro, satellites, draws=draws, seed=seed, horizon=horizon, drivers=needed
    )
    fixed = runs.readable(paths)

    def valued(found: FixedPaths):
        return engine_run(
            portfolio.positions,
            found,
            drivers=satellites.registry,
            served=satellites.served(),
            base_currency=portfolio.base_currency,
            as_of=macro.as_of,
        ).results

    results = valued(fixed)
    value = float(sum(r.value_current for r in results if r.status != "invalid"))
    if value <= 0:
        raise ValueError("no position of the portfolio could be valued on the forecast")
    monthly = np.column_stack(
        [losses_of(valued(fixed.up_to(t))) for t in range(1, horizon + 1)]
    )
    return Forecast(paths, -100 * monthly / value, value, needed)

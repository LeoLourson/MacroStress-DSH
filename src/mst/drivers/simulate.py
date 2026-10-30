"""Driver paths under a macro simulation: steps 4 and 5 of section 7 of the formalization.

Repetition ``r`` of the macro simulation takes one posterior draw of the satellites and draws the
residuals of all drivers jointly, with standard deviations of that draw and the correlation of
the version. Each driver starts from its last value known at ``t = 0``; when that value is older,
the months up to ``t = 0`` are simulated on the observed macro panel and, for its last ``p``
months, on the history drawn by the repetition; a month whose ``x`` the panel does not know
stops the driver. A driver without history of its own starts from the level of the nearest
object above it and has the residual spread of the nearest one with data, independent of the
others. Overlays add their ``delta`` to the path in the model scale.
"""

from dataclasses import dataclass

import numpy as np

from mst.model_sdk.contracts import MacroSimulation
from mst.scenario.contracts import Overlay

from .estimate import regressors
from .version import Satellites

PARAMETERS_FROM_ABOVE = "parameters_from_above"
"""The object has no history: its parameters and level come from the level above."""
APPROXIMATE = "approximate_data"
"""Some values the satellite was estimated on have an estimated publication date."""
OVERLAY = "overlay"
"""An expert overlay of the scenario moves the path."""


class NotServed(KeyError):
    """A driver the satellite version does not serve, with the reason."""

    def __init__(self, driver: str, reason: str):
        self.driver, self.reason = driver, reason
        super().__init__(f"{driver}: {reason}")


@dataclass(frozen=True)
class DriverPaths:
    ids: list[str]
    paths: np.ndarray
    """(R, H + 1, K) in the model scale of each driver; t = 0 is the calculation month."""
    flags: dict[str, frozenset[str]]


def _x(quantity, ids: list[str], months: np.ndarray) -> np.ndarray:
    """``x`` of a quantity from macro months ``(..., n)`` in the order of ``ids``."""
    index = {id_: i for i, id_ in enumerate(ids)}
    parts = [months[..., index[v]] for v in quantity.regressors]
    parts += [np.maximum(0.0, months[..., index[f.variable]] - f.above) for f in quantity.features]
    return np.stack(parts, axis=-1)


def _nearest(hist, node: int, rows: np.ndarray) -> int:
    """The nearest node above ``node`` that was estimated on data of its own."""
    above = hist.parent[node]
    while above >= 0 and rows[above] == 0:
        above = hist.parent[above]
    if above < 0:
        raise NotServed(hist.driver[node], "no object above it has history")
    return above


def simulate(
    sat: Satellites,
    macro: MacroSimulation,
    drivers: list[str],
    overlays: list[Overlay] = (),
    *,
    seed: int,
) -> DriverPaths:
    if macro.history is None:
        raise ValueError("the macro simulation carries no history months")
    served = set(sat.served())
    for driver in drivers:
        quantity, node = sat.node(driver)
        if driver not in served:
            raise NotServed(driver, sat.checks[quantity][node].reason)
    draws, horizon = macro.paths.shape[0], macro.paths.shape[1] - 1
    p = macro.history.shape[1]
    plan = []
    for driver in drivers:
        quantity, node = sat.node(driver)
        fit_ = sat.fits[quantity]
        hist = fit_.history
        if hist.months[-1] != macro.months[0]:
            raise ValueError(
                f"{driver} was estimated up to {hist.months[-1]}, the simulation starts at "
                f"{macro.months[0]}: take the satellites of the same macro version"
            )
        own = fit_.rows[node] > 0
        source = node if own else _nearest(hist, node, fit_.rows)
        last = int(np.flatnonzero(np.isfinite(hist.d[:, source]))[-1])
        lags, count = hist.quantity.lags, len(hist.months)
        observed = hist.x[max(0, last + 1 - lags) : count - p]
        if not np.all(np.isfinite(observed)):
            raise NotServed(driver, "x is unknown between its last value and t = 0")
        flags = set()
        if not own:
            flags.add(PARAMETERS_FROM_ABOVE)
        if hist.approximate[:, source].any():
            flags.add(APPROXIMATE)
        plan.append((driver, quantity, node, source, own, hist.d[last, source], last, flags))

    joint = [item[0] for item in plan if item[4]]
    position = {d: sat.correlated.index(d) for d in joint}
    corr = sat.correlation[np.ix_([position[d] for d in joint], [position[d] for d in joint])]
    chol = np.linalg.cholesky(corr) if joint else np.zeros((0, 0))
    count = next(iter(sat.fits.values())).gamma.shape[0]
    length = next(iter(sat.fits.values())).history.d.shape[0]
    steps = horizon + length - 1 - min((item[6] for item in plan), default=length - 1)

    paths = np.empty((draws, horizon + 1, len(plan)))
    for r in range(draws):
        rng = np.random.default_rng([seed, 1, r])
        s = int(rng.integers(count))
        shocks = rng.standard_normal((steps, len(joint))) @ chol.T
        single = rng.standard_normal((steps, len(plan)))
        for k, (driver, quantity, node, source, own, start, last, _) in enumerate(plan):
            fit_ = sat.fits[quantity]
            hist = fit_.history
            gamma, lags = fit_.gamma[s, node], hist.quantity.lags
            sigma = fit_.sigma[s, source]
            # x on the panel months, then the drawn history and the paths of repetition r.
            x = np.concatenate(
                [
                    hist.x[: len(hist.months) - p],
                    _x(hist.quantity, macro.ids, macro.history[r]),
                    _x(hist.quantity, macro.ids, macro.paths[r, 1:]),
                ]
            )
            noise = shocks[:, joint.index(driver)] if own else single[:, k]
            value, zero = start, len(hist.months) - 1
            for step, u in enumerate(range(last + 1, zero + horizon + 1)):
                value = gamma @ regressors(lags, x, value, u) + sigma * noise[step]
                if u >= zero:
                    paths[r, u - zero, k] = value
            if last == zero:
                paths[r, 0, k] = start
    flags = {item[0]: item[7] for item in plan}
    for overlay in overlays:
        if overlay.driver not in flags:
            raise ValueError(f"overlay on {overlay.driver}, which the run does not simulate")
        k = drivers.index(overlay.driver)
        paths[:, overlay.start : overlay.end + 1, k] += overlay.delta
        flags[overlay.driver].add(OVERLAY)
    return DriverPaths(
        ids=list(drivers),
        paths=paths,
        flags={driver: frozenset(flags[driver]) for driver in drivers},
    )

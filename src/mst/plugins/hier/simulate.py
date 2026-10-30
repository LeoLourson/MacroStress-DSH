"""Driver paths under a macro simulation: the hierarchical model on the levels above.

Repetition ``r`` of the macro simulation takes one posterior draw of the satellites and draws
the residuals of all drivers jointly, with standard deviations of that draw and the correlation
the platform passes in (measured over every driver of the bundle with history of its own).
Each driver starts from its last value known at ``t = 0``; when that value is older, the months
up to ``t = 0`` are simulated on the observed macro panel and, for its last ``p`` months, on
the history drawn by the repetition; a month whose ``x`` the panel does not know stops the
driver. A driver without history of its own starts from the level of the nearest object above
it and has the residual spread of the nearest one with data, independent of the others.
The impacts of the scenario add their ``delta`` to the path in the model scale.
"""

import numpy as np

from mst.drivers.history import x_from_paths
from mst.model_sdk.contracts import AdditiveImpact, DriverPaths, MacroSimulation

from .estimate import regressors

PARAMETERS_FROM_ABOVE = "parameters_from_above"
"""The object has no history: its parameters and level come from the level above."""
APPROXIMATE = "approximate_data"
"""Some values the satellite was estimated on have an estimated publication date."""
IMPACT = "impact"
"""A direct impact of the scenario moves the path."""


class NotServed(KeyError):
    """A driver the check rejected, with the reason."""

    def __init__(self, driver: str, reason: str):
        self.driver, self.reason = driver, reason
        super().__init__(f"{driver}: {reason}")


def _nearest(hist, node: int, rows: np.ndarray) -> int:
    """The nearest node above ``node`` that was estimated on data of its own."""
    above = hist.parent[node]
    while above >= 0 and rows[above] == 0:
        above = hist.parent[above]
    if above < 0:
        raise NotServed(hist.driver[node], "no object above it has history")
    return above


def simulate(
    model,
    macro: MacroSimulation,
    drivers: list[str],
    impacts: list[AdditiveImpact],
    *,
    correlation: tuple[list[str], np.ndarray] | None = None,
    seed: int,
    stream: int,
) -> DriverPaths:
    """The paths of ``drivers`` on ``macro``, in the model scale of each quantity.

    ``model`` is what ``HierModel.load`` returned: the fits of its quantities and the check of
    every driver. ``correlation`` is the bundle's residual correlation with the driver ids it
    was measured on; the joint noise of this model's drivers with history is drawn from its
    slice.
    """
    if macro.history is None:
        raise ValueError("the macro simulation carries no history months")
    correlated, matrix = correlation if correlation is not None else ([], np.zeros((0, 0)))
    draws, horizon = macro.paths.shape[0], macro.paths.shape[1] - 1
    p = macro.history.shape[1]
    plan = []
    for driver in drivers:
        quantity, node = _node(model, driver)
        fit_ = model.fits[quantity]
        hist = fit_.history
        item = model.checks[quantity].get(node)
        if item is None or not item.served:
            raise NotServed(driver, item.reason if item else "no check of this driver")
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
    picked = [correlated.index(d) for d in joint if d in correlated]
    corr = matrix[np.ix_(picked, picked)] if picked else np.zeros((0, 0))
    chol = np.linalg.cholesky(corr) if picked else np.zeros((0, 0))
    count = next(iter(model.fits.values())).gamma.shape[0]
    length = next(iter(model.fits.values())).history.d.shape[0]
    steps = horizon + length - 1 - min((item[6] for item in plan), default=length - 1)

    paths = np.empty((draws, horizon + 1, len(plan)))
    for r in range(draws):
        rng = np.random.default_rng([seed, stream, r])
        s = int(rng.integers(count))
        shocks = rng.standard_normal((steps, len(joint))) @ chol.T
        single = rng.standard_normal((steps, len(plan)))
        for k, (driver, quantity, node, source, own, start, last, _) in enumerate(plan):
            fit_ = model.fits[quantity]
            hist = fit_.history
            gamma, lags = fit_.gamma[s, node], hist.quantity.lags
            sigma = fit_.sigma[s, source]
            # x on the panel months, then the drawn history and the paths of repetition r.
            x = np.concatenate(
                [
                    hist.x[: len(hist.months) - p],
                    x_from_paths(hist.quantity, macro.ids, macro.history[r]),
                    x_from_paths(hist.quantity, macro.ids, macro.paths[r, 1:]),
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
    for impact in impacts:
        if impact.target not in flags:
            raise ValueError(f"an impact on {impact.target}, which this model does not simulate")
        k = drivers.index(impact.target)
        paths[:, impact.start : impact.end + 1, k] += impact.delta
        flags[impact.target].add(IMPACT)
    return DriverPaths(
        ids=list(drivers),
        paths=paths,
        flags={driver: frozenset(flags[driver]) for driver in drivers},
    )


def _node(model, driver: str) -> tuple[str, int]:
    quantity = driver.split(":", 1)[0]
    fit_ = model.fits.get(quantity)
    if fit_ is None or driver not in fit_.history.driver:
        raise KeyError(driver)
    return quantity, fit_.history.driver.index(driver)

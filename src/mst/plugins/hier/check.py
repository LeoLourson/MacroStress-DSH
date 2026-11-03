"""The out-of-sample check of the satellites (docs/concept/bvar-method-legacy.md, section 9).

The models are estimated on the months before the last ``holdout`` months of the panel. From
every observation in the holdout a driver is forecast, month by month on the realized macro
variables, by its satellite at the posterior mean, and compared with the observations of the
next ``reach`` months. The model without macro variables is ``d_k = alpha + rho d_{k-1}`` over
successive observations, fitted by least squares on the same months, so a quarterly driver is
compared with a quarterly autoregression. A driver is served when its satellite has the smaller
root mean squared error. A driver without history of its own is served when the nearest
driver above it with history is.

A satellite fitted in differences is judged in levels like any other: its forecast accumulates
the changes from the level of the origin, and the autoregression beside it is the same AR(1) on
the levels, so the verdict stays comparable.
"""


import numpy as np

from mst.drivers.history import History
from mst.model_sdk import Check

from .estimate import difference, fit, regressors
from .spec import Spec


def _rmse(errors: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


def _first_change(d: np.ndarray, origin: int, gamma) -> float:
    """The change a differenced forecast starts from at ``origin``.

    The month before the origin is missing now and then; the forecast then starts from the
    stationary mean of the changes of that draw, the model's own unconditional expectation.
    """
    if origin >= 1 and np.isfinite(d[origin - 1]):
        return float(d[origin] - d[origin - 1])
    return float(gamma[0] / (1 - gamma[1]))


def _forecasts(hist: History, node: int, cut: int, reach: int, gamma, alpha, rho, x, differenced):
    """Errors of the satellite and of the autoregression from every origin of the holdout."""
    d, lags = hist.d[:, node], hist.quantity.lags
    observed = np.flatnonzero(np.isfinite(d))
    mine, base = [], []
    for origin in observed[observed >= cut]:
        targets = observed[(observed > origin) & (observed <= origin + reach)]
        if not len(targets):
            continue
        value, change, reached = d[origin], _first_change(d, origin, gamma), set()
        for t in range(origin + 1, int(targets[-1]) + 1):
            z = regressors(lags, x, change if differenced else value, t)
            if not np.all(np.isfinite(z)):
                break
            step = float(gamma @ z)
            if differenced:
                change = step
                value = value + change
            else:
                value = step
            if t in targets:
                reached.add(t)
                mine.append(d[t] - value)
        value = d[origin]
        for t in targets:
            value = alpha + rho * value
            if t in reached:
                base.append(d[t] - value)
    return mine, base


def check(hist: History, spec: Spec, *, progress: str | None = None) -> dict[int, Check]:
    """The check of every node with a driver, by node index; ``progress`` labels the sampler."""
    cut = len(hist.months) - 1 - spec.holdout
    trained = fit(hist, spec, until=cut, progress=progress)
    x = difference(hist.x) if trained.differenced else hist.x
    out = {}
    for node, driver in enumerate(hist.driver):
        d = hist.d[:, node]
        if driver is None or not np.isfinite(d).any():
            continue
        train = np.flatnonzero(np.isfinite(d[: cut + 1]))
        if trained.rows[node] == 0 or len(train) < 4:
            out[node] = Check(None, None, 0, False, "too few observations before the holdout")
            continue
        own = np.column_stack([np.ones(len(train) - 1), d[train[:-1]]])
        alpha, rho = np.linalg.lstsq(own, d[train[1:]], rcond=None)[0]
        gamma = trained.gamma[:, node].mean(axis=0)
        mine, base = _forecasts(
            hist, node, cut, spec.reach, gamma, alpha, rho, x, trained.differenced
        )
        a, b = _rmse(mine), _rmse(base)
        if a is None or b is None:
            out[node] = Check(a, b, len(mine), False, "no observation of the holdout forecast")
        elif a < b:
            out[node] = Check(a, b, len(mine), True, "")
        else:
            out[node] = Check(a, b, len(mine), False, "not more accurate than without macro")
    for node, driver in enumerate(hist.driver):
        if driver is None or node in out:
            continue
        above = hist.parent[node]
        while above >= 0 and above not in out:
            above = hist.parent[above]
        if above < 0:
            out[node] = Check(None, None, 0, False, "no checked driver above it")
        else:
            served = out[above].served
            reason = "" if served else f"the driver above, {hist.driver[above]}, is not served"
            out[node] = Check(None, None, 0, served, reason)
    return out

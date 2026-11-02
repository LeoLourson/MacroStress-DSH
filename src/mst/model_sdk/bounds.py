"""Lower bounds of the macro variables: the shadow value and the observed one.

A linear model does not know that a rate cannot go below zero, so the value it works with
is a shadow: the dynamics, the lags and the conditions all run on it untouched, and what a
simulation reports is the observed value — the shadow read through its floor,
``x_t = max(s_t, bound)``, in the model scale. An exact condition at the bound, such as
"the Fed at zero", means the shadow within its tolerance of the bound; the price of a
scenario stays on the shadow, the gaussian law the machinery knows. Estimation still reads
the history as observed — a proper shadow-rate estimation is another piece of work; the
approximation is stated in the plugin READMEs.

The floors live in the macro registry (``lower`` of a variable, in its own unit); a model
variable the registry does not know — the ids of a test panel, say — carries none.
"""

import numpy as np


def lower_bounds(registry, ids: list[str]) -> np.ndarray | None:
    """The model-scale floor of each named variable; ``None`` when no variable has one.

    An ``identity`` variable's floor is its own value; a ``log100`` variable's floor below
    or at zero maps to minus infinity, which bounds nothing. A variable the registry does
    not know carries no floor, so a panel of test ids stays unbounded.
    """
    known = {variable.id: variable for variable in registry.variables}
    bounds = np.full(len(ids), -np.inf)
    found = False
    for k, id_ in enumerate(ids):
        variable = known.get(id_)
        if variable is None or variable.lower is None:
            continue
        if variable.transform == "log100":
            bounds[k] = -np.inf if variable.lower <= 0 else 100.0 * np.log(variable.lower)
        else:
            bounds[k] = variable.lower
        found = found or np.isfinite(bounds[k])
    return bounds if found else None


def clip_lower(paths: np.ndarray, bounds: np.ndarray | None) -> np.ndarray:
    """The observed paths of the shadow ones: each column read through its floor.

    ``bounds`` is what :func:`lower_bounds` returned for the columns of ``paths``; ``None``
    leaves the paths as they are, floor by floor and bit for bit.
    """
    if bounds is None:
        return paths
    return np.maximum(paths, bounds[None, None, :])

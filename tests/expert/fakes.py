"""Synthetic answers, a series store and a context for the expert plugin tests.

The answers are consistent — they pass ``answers.check`` — and cover the whole registry:
every shock answers about all 23 variables, one shock is one-sided, one has zero
frequency, and every answer carries a non-proportional response so it is not refused as
linear. No LLM is called and no real answer is written.
"""

from datetime import date
from pathlib import Path

import numpy as np

from mst.data.store import Observation, SeriesStore
from mst.macro.panel import months_between, shift
from mst.macro.registry import load_registry
from mst.model_sdk.contracts import MacroEstimateContext
from mst.plugins.expert.model import MARKET, load_shocks

#: The calculation date of the fixture and the first month of its history.
AS_OF = date(2026, 11, 1)
START = date(2012, 1, 1)
#: The fragments of the model corpus the fixture's answers cite; all visible on the date.
FRAGMENTS = ("M1", "M2")
#: The seed of the synthetic history: the same fixture gives the same numbers.
SEED = 11
#: The monthly drift and noise of the synthetic history, in the model scale, by transform.
SCALES = {"log100": (500.0, 0.2, 1.0), "rate": (8.0, 0.005, 0.05), "free": (2.0, 0.01, 0.1)}


def series_store(root) -> SeriesStore:
    """Every registered variable as a random walk with drift, positive where it must be."""
    registry = load_registry()
    rng = np.random.default_rng(SEED)
    store = SeriesStore(root)
    months = months_between(START, shift(START, 177))
    for k, variable in enumerate(registry.variables):
        base, drift, noise = (
            SCALES["log100"]
            if variable.transform == "log100"
            else SCALES["rate"]
            if variable.lower is not None
            else SCALES["free"]
        )
        level = base + drift * np.arange(len(months)) + noise * np.cumsum(
            rng.standard_normal(len(months))
        ) + (k % 7) * 0.5
        if variable.transform == "log100":
            values = variable.from_model(level)
        elif variable.lower is not None:
            values = np.maximum(level, variable.lower + 0.1)
        else:
            values = level
        store.write(
            variable.id,
            [
                Observation(month, float(value), shift(month, 1), False)
                for month, value in zip(months, values, strict=True)
            ],
        )
    return store


def context(root, *, as_of: date = AS_OF, fragment=None, inputs=None) -> MacroEstimateContext:
    """The estimate context of the fixture: the store, the registry and the date."""
    return MacroEstimateContext(
        store=series_store(root),
        registry=load_registry(),
        as_of=as_of,
        fragment=fragment,
        inputs=inputs,
    )


def visible(fragment_id: str) -> bool:
    """The fragment callable of the fixture: its two fragments are visible, others are not."""
    return fragment_id in FRAGMENTS


def _quantiles(centre: float, spread: float) -> dict:
    return {"q5": centre - spread, "q50": centre, "q95": centre + spread}


def _scale_of(variable) -> float:
    """The response scale of a variable: a hundredth of its history's monthly noise."""
    _, _, noise = (
        SCALES["log100"]
        if variable.transform == "log100"
        else SCALES["rate"]
        if variable.lower is not None
        else SCALES["free"]
    )
    return 0.01 * noise


def shock_payload(shock_id: str, **changes) -> dict:
    """One consistent answer about one shock: nonlinear, one-sided or zero-frequency options."""
    one_sided = changes.pop("one_sided", False)
    frequency = changes.pop("frequency", 0.5)
    sizes = (
        {"q5": 0.0, "q50": 2.0, "q95": 5.0}
        if one_sided
        else {"q5": -4.0, "q50": 0.5, "q95": 4.0}
    )
    responses = {}
    for k, variable in enumerate(load_registry().variables):
        by_horizon = {}
        for horizon in (1, 3, 6, 12):
            node = horizon / 12.0 + 0.2
            by_node = {}
            for index, name in enumerate(("q5", "q50", "q95")):
                size = sizes[name]
                # A response deliberately not proportional to the size: the factor
                # 1 + 0.1 * index bends the line through the three nodes.
                centre = (
                    _scale_of(variable)
                    * size
                    * (1 + 0.1 * index)
                    * node
                    * (1.0 + 0.02 * (k % 3))
                )
                by_node[name] = _quantiles(centre, abs(centre) * 0.25 + 0.01)
            by_horizon[str(horizon)] = by_node
        responses[variable.id] = by_horizon
    payload = {
        "shock": shock_id,
        "frequency": frequency,
        "size": sizes,
        "responses": responses,
        "rho": 0.5,
        "nonlinearity": (
            "the response to this shock grows less than proportionally with the size at "
            "the 95 % node: the pass-through saturates"
        ),
        "sources": [],
    }
    payload.update(changes)
    return payload


def answers() -> dict[str, dict]:
    """The 18 consistent answers of the fixture, by key."""
    found = {}
    for index, shock in enumerate(load_shocks()):
        found[shock.id] = shock_payload(
            shock.id,
            one_sided=shock.id == "gl.oil",
            frequency=0.0 if shock.id == "gl.grain" else 0.4 + 0.1 * (index % 5),
        )
    return found


def market_payload(**changes) -> dict:
    """The market answer of the fixture: two variables with their fragments."""
    payload = {
        "values": {
            "gl.brent": {
                "horizons": {"1": 540.0, "3": 543.0, "6": 546.0, "12": 550.0},
                "fragment": "M1",
            },
            "ru.usdrub": {
                "horizons": {"1": 541.0, "3": 544.0, "6": 547.0, "12": 552.0},
                "fragment": "M2",
            },
        }
    }
    payload.update(changes)
    return payload


def full_answers() -> dict[str, dict]:
    """Every answer the market variant needs: the 18 shocks and the market."""
    return {**answers(), MARKET: market_payload()}


def answers_context(root, *, base="rw", with_market=None, changes=None):
    """A context with the fixture's answers written into its elicitation folder.

    ``changes`` maps a key to a function that rewrites that answer's payload; the answers
    are checked and stored the way the server does, in their normal forms.
    """
    from mst import inputs
    from mst.plugins.expert import answers as answers_module

    folder = Path(root) / "elicitation"
    ctx = context(root, fragment=visible, inputs=folder)
    payloads = full_answers() if base == "market" else answers()
    if with_market is False:
        payloads = answers()
    for key, payload in payloads.items():
        if changes and key in changes:
            payload = changes[key](payload)
        normal = answers_module.check(ctx, key, payload)
        inputs.write(folder, AS_OF, key, normal)
    return context(root, fragment=visible, inputs=folder)


def quick_spec(**changes):
    """The spec of the estimate tests: small grids, a short stream, the fixture's dates."""
    from mst.plugins.expert.spec import Spec

    values = {
        "family": "pareto",
        "alpha": 4.0,
        "levels": 129,
        "stream_draws": 12000,
        "candidates": 16,
        "subset_samples": 500,
        "history_start": START,
        "seed": 3,
    }
    values.update(changes)
    return Spec.model_validate(values)

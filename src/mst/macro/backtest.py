"""The check of the macro model on historical episodes (docs/scenario-formalization.md, section 9).

For an episode of the corpus that starts in month ``s`` the calculation date is the day before
``s``. The model is estimated on the panel known that day. Regime dummies of later months are
left out of that estimation: no month of its history identifies them, and nobody knew of them
yet. The realized paths of the constrained variables, as known on the check date, fix the
scenario month by month up to the end of the episode window, and the driving shocks are the
ones the sources of the time name. An unconstrained variable is covered in a month when its
realized value lies in ``[P5, P95]`` of the simulated paths.

Seasonal variables are compared after removing the calendar profile estimated on the
calculation date, as the model saw them. An exclusion of ``backtest.toml`` drops the months of a
variable from its ``start`` on when the history ends before it: the realized level there is not
comparable with that history (an index splice). Such a month is neither constrained nor
compared.
"""

import hashlib
import json
import tomllib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import Field, field_validator

from mst.contracts import Contract, Identifier, Seed
from mst.data.store import SeriesStore
from mst.scenario.contracts import Scenario

from .estimate import Posterior, estimate
from .forecast import Simulation
from .identify import identify
from .panel import Panel, build_panel, seasonal_profile, shift
from .registry import Registry, Shock
from .spec import Count, Spec, first_of_month
from .version import load, save

BACKTEST_PATH = Path(__file__).with_name("backtest.toml")
LOW, HIGH = 5, 95


class Exclusion(Contract):
    variables: list[Identifier] = Field(min_length=1)
    start: date
    reason: Identifier

    _start = field_validator("start")(first_of_month)


class EpisodeCheck(Contract):
    id: Identifier
    constrained: list[Identifier] = Field(min_length=1)
    driving: list[Identifier] = Field(min_length=1)
    reason: Identifier
    """Why these shocks drove the episode, with the corpus keys of the sources of the time."""
    sources: list[Identifier] = Field(min_length=1)


class Backtest(Contract):
    draws: Count
    seed: Seed
    after_months: Annotated[int, Field(strict=True, ge=1)]
    """The window ends this many months after the start month, as in the episode cards."""
    exclusions: list[Exclusion] = []
    episodes: list[EpisodeCheck] = Field(min_length=1)


def load_backtest(path: Path = BACKTEST_PATH) -> Backtest:
    with path.open("rb") as file:
        return Backtest.model_validate(tomllib.load(file))


def calculation_date(start: date) -> date:
    """The day before the episode starts."""
    return start - timedelta(days=1)


def history_spec(spec: Spec, last: date) -> Spec:
    """``spec`` without the regime dummies of months after ``last``."""
    return spec.model_copy(update={"dummies": [d for d in spec.dummies if d.month <= last]})


@dataclass(frozen=True)
class Case:
    """An episode prepared for the check: the scenario and the realized paths to compare."""

    id: str
    as_of: date
    months: list[date]
    """Months t = 1 .. H after the last month of history."""
    realized: np.ndarray
    """(H, n) in the model scale as the model saw it; NaN where unknown or excluded."""
    constrained: list[str]
    scenario: Scenario


def realized_paths(
    store: SeriesStore,
    registry: Registry,
    *,
    ids: list[str],
    months: list[date],
    history_end: date,
    known_on: date,
    profiles: dict[str, np.ndarray],
    exclusions: list[Exclusion],
) -> np.ndarray:
    index = {month: t for t, month in enumerate(months)}
    y = np.full((len(months), len(ids)), np.nan)
    for i, id_ in enumerate(ids):
        variable = registry.variable(id_)
        for row in store.as_of(id_, known_on):
            if row.month in index:
                y[index[row.month], i] = variable.to_model(row.value)
        if id_ in profiles:
            y[:, i] -= profiles[id_][[m.month - 1 for m in months]]
        for exclusion in exclusions:
            if id_ in exclusion.variables and exclusion.start > history_end:
                y[[m >= exclusion.start for m in months], i] = np.nan
    return y


def prepare(
    check: EpisodeCheck,
    start: date,
    backtest: Backtest,
    store: SeriesStore,
    registry: Registry,
    spec: Spec,
    known_on: date,
):
    """The case of an episode and the panel and spec to estimate the model on."""
    as_of = calculation_date(start)
    panel = build_panel(store, registry, start=spec.start, as_of=as_of, seasonal=spec.seasonal)
    raw = build_panel(store, registry, start=spec.start, as_of=as_of)
    profiles = {
        id_: seasonal_profile(raw.y[:, raw.ids.index(id_)], raw.months) for id_ in spec.seasonal
    }
    last = panel.months[-1]
    end = shift(start, backtest.after_months)
    horizon = (end.year - last.year) * 12 + end.month - last.month
    months = [shift(last, k) for k in range(1, horizon + 1)]
    realized = realized_paths(
        store,
        registry,
        ids=panel.ids,
        months=months,
        history_end=last,
        known_on=known_on,
        profiles=profiles,
        exclusions=backtest.exclusions,
    )
    constraints = []
    for id_ in check.constrained:
        i = panel.ids.index(id_)
        for k, value in enumerate(realized[:, i], start=1):
            if not np.isnan(value):
                constraints.append(
                    {
                        "terms": [{"variable": id_, "month": k, "weight": 1.0}],
                        "low": float(value),
                        "high": float(value),
                    }
                )
    scenario = Scenario.model_validate(
        {
            "narrative": check.id,
            "sources": check.sources,
            "horizon": len(months),
            "constraints": constraints,
            "driving_shocks": check.driving,
        }
    )
    registry.check(scenario)
    case = Case(check.id, as_of, months, realized, check.constrained, scenario)
    return case, panel, history_spec(spec, last)


@dataclass(frozen=True)
class Coverage:
    hits: dict[str, int]
    counts: dict[str, int]
    """Per unconstrained variable: months inside ``[P5, P95]`` and months compared."""
    quantiles: np.ndarray
    """(3, H, n): P5, P50, P95 of months t = 1 .. H."""

    @property
    def share(self) -> float | None:
        total = sum(self.counts.values())
        return sum(self.hits.values()) / total if total else None


def coverage(case: Case, simulation: Simulation) -> Coverage:
    quantiles = np.percentile(simulation.paths[:, 1:, :], [LOW, 50, HIGH], axis=0)
    hits, counts = {}, {}
    for i, id_ in enumerate(simulation.ids):
        known = ~np.isnan(case.realized[:, i])
        if id_ in case.constrained or not known.any():
            continue
        value = case.realized[known, i]
        inside = (quantiles[0, known, i] <= value) & (value <= quantiles[2, known, i])
        hits[id_], counts[id_] = int(inside.sum()), int(known.sum())
    return Coverage(hits, counts, quantiles)


def _key(panel: Panel, spec: Spec, as_of: date, shocks: list[Shock]) -> str:
    meta = [
        spec.model_dump(mode="json"),
        as_of.isoformat(),
        panel.ids,
        [s.model_dump(mode="json") for s in shocks],
    ]
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode())
    digest.update(np.ascontiguousarray(panel.y).tobytes())
    digest.update(np.ascontiguousarray(panel.approximate).tobytes())
    return digest.hexdigest()


def models_for(
    panel: Panel, spec: Spec, as_of: date, tables: dict[str, list[Shock]], root: Path
) -> dict[str, tuple[str, Posterior]]:
    """The model version of each sign table on the calculation date, estimated once.

    ``<root>/backtest/<key>`` names the version a table gave on the same panel and spec, so a
    rerun loads it instead of estimating again.
    """
    keys = {name: _key(panel, spec, as_of, shocks) for name, shocks in tables.items()}
    found = {}
    for name, key in keys.items():
        pointer = root / "backtest" / key
        if pointer.exists():
            version = pointer.read_text(encoding="utf-8").strip()
            found[name] = (version, load(root, version)[0])
    missing = [name for name in tables if name not in found]
    if missing:
        posterior = estimate(panel, spec)
        for name in missing:
            model = identify(
                posterior,
                tables[name],
                sign_lag=spec.sign_lag,
                tries=spec.rotation_tries,
                seed=spec.sampler.seed,
            )
            version = save(model, spec, panel, as_of, root)
            pointer = root / "backtest" / keys[name]
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(version + "\n", encoding="utf-8")
            found[name] = (version, model)
    return found


EXPLOSIVE = 1.02
"""A draw is explosive above this modulus; levels near a unit root sit just above 1."""


def explosive_share(model: Posterior, limit: float = EXPLOSIVE) -> float:
    """The share of draws whose companion matrix has an eigenvalue of modulus above ``limit``:
    their paths grow without bound, and a band built on them covers anything."""
    n, p = len(model.ids), model.lags_count
    companion = np.zeros((model.draws, n * p, n * p))
    companion[:, :n] = np.concatenate(list(model.lags.transpose(1, 0, 2, 3)), axis=2)
    companion[:, n:, :-n] = np.eye(n * (p - 1))
    return float((np.abs(np.linalg.eigvals(companion)).max(axis=1) > limit).mean())
